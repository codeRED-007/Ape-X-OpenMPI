"""
replay_mpi.py — Replay buffer rank (rank 1) for Ape-X with OpenMPI.

Replaces replay.py entirely. A single loop replaces the entire ZMQ
architecture that was:

    ┌──────────────────────────────────────────────────────────────────────┐
    │ ZMQ (replay.py)                                                      │
    │                                                                      │
    │  3 proxy Processes (TCP ↔ IPC bridges)                               │
    │    recv_batch_device()   ROUTER:51001  ↔  ipc:///tmp/5101.ipc        │
    │    recv_prios_device()   ROUTER:51002  ↔  ipc:///tmp/5102.ipc        │
    │    send_batch_device()   ROUTER:51003  ↔  ipc:///tmp/5103.ipc        │
    │                                                                      │
    │  asyncio.gather() over N coroutines:                                 │
    │    recv_batch_worker  ×4  — receive actor batches, push to buffer    │
    │    recv_prios_worker  ×4  — receive prio updates from learner        │
    │    send_batch_worker  ×8  — sample + send batches to learner         │
    │                                                                      │
    │  asyncio.Lock to serialise buffer access across coroutines           │
    │  ThreadPoolExecutor to offload CPU-bound buffer ops off event loop   │
    └──────────────────────────────────────────────────────────────────────┘

    ┌──────────────────────────────────────────────────────────────────────┐
    │ MPI (replay_mpi.py)                                                  │
    │                                                                      │
    │  Single while-True loop using comm.iprobe() as a dispatcher:        │
    │                                                                      │
    │   if TAG_BATCH pending   → recv from any actor, push to buffer       │
    │   if TAG_PRIOS pending   → recv from learner, update priorities      │
    │   if TAG_SAMPLE_REQ      → recv request, sample, send back           │
    │                            (only once buffer is warm)                │
    │                                                                      │
    │  No proxy processes. No asyncio. No IPC sockets. No ThreadPool.      │
    │  No pickle calls (mpi4py serialises automatically).                  │
    └──────────────────────────────────────────────────────────────────────┘

Why iprobe() instead of blocking recv():
    MPI's recv() blocks until ONE specific message arrives.  The replay
    rank needs to service three independent message streams concurrently:
    actor batches, learner priority updates, and learner sample requests.
    A naive blocking recv() on one stream would deadlock if a message
    arrives on a different stream first (see detailed explanation in the
    project notes).  iprobe() checks non-blockingly whether a message
    matching (source, tag) is available, then recv() is called only when
    we know it will return immediately.

Rank layout (must match learner_mpi.py / actor_mpi.py):
    Rank 0             → Learner
    Rank 1             → Replay buffer  (this file)
    Ranks 2 .. N-2     → Actors   (actor_id = rank - 2)
    Rank N-1           → Evaluator
"""

import os
import time

from mpi4py import MPI

import utils
from memory import CustomPrioritizedReplayBuffer
from arguments_new import argparser

# ── Rank constants (keep in sync with learner_mpi.py / actor_mpi.py) ─────────
LEARNER_RANK = 0
REPLAY_RANK  = 1   # this rank

# ── Message tags (keep in sync across all MPI files) ──────────────────────────
TAG_BATCH      = 10   # actor  → replay  : (batch, prios) experience bundle
TAG_PARAMS     = 20   # learner → actor  : state_dict with updated weights
TAG_SAMPLE_REQ = 30   # learner → replay : "please send me a training batch"
TAG_SAMPLE     = 31   # replay  → learner: sampled training batch
TAG_PRIOS      = 40   # learner → replay : updated priority scores


# ─────────────────────────────────────────────────────────────────────────────
# Buffer helpers  (mirror the ZMQ version's push_batch / update_prios /
# sample_batch support functions, but without pickle — mpi4py handles that)
# ─────────────────────────────────────────────────────────────────────────────

def _push_batch(buffer: CustomPrioritizedReplayBuffer, payload: tuple) -> None:
    """
    Unpack an (batch, prios) tuple received from an actor and add every
    transition to the replay buffer.

    In the ZMQ version push_batch() received raw bytes and called
    pickle.loads() itself.  Here mpi4py has already deserialised the
    object, so we just unpack and add.
    """
    batch, prios = payload
    for sample in zip(*batch, prios):
        buffer.add(*sample)


def _update_prios(buffer: CustomPrioritizedReplayBuffer, payload: tuple) -> None:
    """
    Apply a (idxes, prios) priority update from the learner to the buffer.
    """
    idxes, prios = payload
    buffer.update_priorities(idxes, prios)


def _sample_batch(buffer: CustomPrioritizedReplayBuffer,
                  batch_size: int, beta: float) -> list:
    """
    Draw a prioritised sample from the buffer and return it as a plain
    Python list.  mpi4py will serialise it automatically on send().
    """
    return buffer.sample(batch_size, beta)


# ─────────────────────────────────────────────────────────────────────────────
# Main replay loop
# ─────────────────────────────────────────────────────────────────────────────

def replay_main(comm: MPI.Comm, args) -> None:
    """
    Full event loop for the replay buffer rank (rank 1).

    Parameters
    ----------
    comm : MPI.Comm
        The global MPI communicator (MPI.COMM_WORLD).
    args : argparse.Namespace
        Parsed arguments from argparser().
    """
    utils.set_global_seeds(args.seed, use_torch=False)

    buffer = CustomPrioritizedReplayBuffer(args.replay_buffer_size, args.alpha)

    # warm_start: once the buffer has enough transitions the learner is
    # allowed to start requesting sample batches.  Mirrors the asyncio
    # Event() that gated recv_prios_worker and send_batch_worker in the
    # ZMQ version.
    warm_start = False

    # ── Logging counters ──────────────────────────────────────────────────────
    batch_recv_cnt = 0          # total actor batches received
    ts             = time.time()

    # ── Startup sync ──────────────────────────────────────────────────────────
    # Every rank calls Barrier in its _main function.  The replay rank
    # reaches here first (it has no model to build), so it just waits.
    print("[Replay] Waiting at barrier for all ranks...", flush=True)
    comm.Barrier()
    print("[Replay] All ranks ready. Starting replay loop.", flush=True)

    # ── Main dispatch loop ────────────────────────────────────────────────────
    #
    # Design: poll each message type in priority order using iprobe().
    # Priority order matters:
    #   1. Actor batches first  — filling the buffer is the highest priority;
    #                             we want to reach warm_start as fast as
    #                             possible.
    #   2. Priority updates     — keeping priorities accurate is important for
    #                             training quality; process these promptly.
    #   3. Sample requests last — only once warm, and only one per loop
    #                             iteration so we don't starve the other two.
    #
    # All three branches call recv() only after iprobe() returns True, so
    # recv() is always instantaneous — no branch ever blocks the loop.
    #
    # Idle sleep: if none of the three probes fire we sleep for 100 µs to
    # avoid burning a full CPU core in a hot spin-loop.
    while True:

        did_work = False

        # ── 1. Receive experience batch from any actor ────────────────────────
        # MPI.ANY_SOURCE accepts from any actor rank (or any rank, but only
        # actors send TAG_BATCH so this is safe).
        status = MPI.Status()
        if comm.iprobe(source=MPI.ANY_SOURCE, tag=TAG_BATCH, status=status):
            source = status.Get_source() # 1. Get the exact actor ID
            payload = comm.recv(source=source, tag=TAG_BATCH) # 2. Force recv to only pull from that actor
            _push_batch(buffer, payload)
            payload = None
            did_work = True
            batch_recv_cnt += 1

            # ── Logging (mirrors recv_batch_worker's cnt % 100 check) ─────────
            if batch_recv_cnt % 100 == 0:
                elapsed = time.time() - ts
                # each batch contains args.send_interval transitions
                fps = (args.send_interval * batch_recv_cnt) / elapsed
                print(
                    "[Replay] Buffer: {:,} transitions / Intake FPS: {:.1f}".format(
                        len(buffer), fps
                    ), flush=True
                )
                # reset the window so FPS reflects recent throughput
                batch_recv_cnt = 0
                ts = time.time()

            # ── Warm-start gate ───────────────────────────────────────────────
            # In the ZMQ version asyncio.Event was set here to unblock the
            # send_batch_worker and recv_prios_worker coroutines.  We use a
            # plain boolean flag instead.
            if not warm_start and len(buffer) >= args.threshold_size:
                warm_start = True
                print(
                    "[Replay] Buffer warm ({:,} transitions). "
                    "Ready to serve learner.".format(len(buffer)),
                    flush=True
                )

        # ── 2. Receive priority updates from learner ──────────────────────────
        # Only meaningful once training has started (warm_start), but it
        # costs nothing to check — and if somehow a prio update arrived
        # before we're warm we should still drain it so it doesn't pile up
        # in the MPI buffer.
        if comm.iprobe(source=LEARNER_RANK, tag=TAG_PRIOS):
            payload = comm.recv(source=LEARNER_RANK, tag=TAG_PRIOS)
            _update_prios(buffer, payload)
            payload = None
            did_work = True

        # ── 3. Serve a sample batch to the learner ────────────────────────────
        # The learner sends a tiny TAG_SAMPLE_REQ message to request a batch.
        # We respond with TAG_SAMPLE.  We only do this once warm so the
        # learner never receives an under-filled sample.
        #
        # Unlike branches 1 and 2 (which can fire multiple times per loop
        # iteration via back-to-back iprobe calls), we handle at most ONE
        # sample request per loop iteration.  Sampling is the most expensive
        # buffer operation (O(log N) segment-tree traversal), so we don't
        # want it to starve batch ingestion.
        if warm_start and comm.iprobe(source=LEARNER_RANK, tag=TAG_SAMPLE_REQ):
            comm.recv(source=LEARNER_RANK, tag=TAG_SAMPLE_REQ)   # consume request
            batch = _sample_batch(buffer, args.batch_size, args.beta)
            comm.send(batch, dest=LEARNER_RANK, tag=TAG_SAMPLE)
            batch = None
            did_work = True

        # ── Idle back-off ─────────────────────────────────────────────────────
        # If nothing arrived this iteration, yield the CPU briefly rather
        # than spinning at 100%.  100 µs is small enough not to add noticeable
        # latency — the learner's training step takes far longer than this.
        if not did_work:
            time.sleep(0.0001)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    os.environ["OMP_NUM_THREADS"] = "1"
    comm = MPI.COMM_WORLD
    args = argparser()
    replay_main(comm, args)


if __name__ == "__main__":
    main()