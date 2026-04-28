"""
learner_mpi.py — Learner rank (rank 0) implementation for Ape-X with OpenMPI.

Replaces learner.py. The learner rank:
  1. Builds the DuelingDQN model and target network on GPU (or CPU if no CUDA).
  2. Waits at a Barrier for all ranks to be ready (replaces check_connection /
     the REQ-ROUTER handshake in the ZMQ version).
  3. Pushes initial weights to every actor and evaluator rank via point-to-point
     isend (replaces the PUB socket on port 52001).
  4. Enters the training loop:
       a. Requests a sample batch from replay (TAG_SAMPLE_REQ → replay).
       b. Blocks on recv for the sample batch (TAG_SAMPLE ← replay).
       c. Moves the batch to GPU, computes Double-DQN loss, updates weights.
       d. Sends updated priorities back to replay (TAG_PRIOS → replay).
       e. Periodically broadcasts updated weights to all actors + evaluator
          via individual isend (replaces the PUB socket broadcast).
       f. Periodically updates the frozen target network.
       g. Periodically saves the model to disk.

ZMQ roles removed and their MPI replacements:
  ┌─────────────────────────────┬──────────────────────────────────────────┐
  │ ZMQ (learner.py)            │ MPI (learner_mpi.py)                     │
  ├─────────────────────────────┼──────────────────────────────────────────┤
  │ check_connection()          │ comm.Barrier()                           │
  │   ROUTER socket :52002      │                                          │
  ├─────────────────────────────┼──────────────────────────────────────────┤
  │ send_param() Process        │ _broadcast_params() helper               │
  │   PUB socket :52001         │   comm.isend() to each actor + eval rank │
  │   multiprocessing.Queue     │   (no queue, no subprocess)              │
  ├─────────────────────────────┼──────────────────────────────────────────┤
  │ recv_batch() Process        │ comm.send(TAG_SAMPLE_REQ) +              │
  │   DEALER socket :51003      │   comm.recv(TAG_SAMPLE)                  │
  │   threading.Thread          │   (inline in train loop, no thread)      │
  │   multiprocessing.Queue     │                                          │
  ├─────────────────────────────┼──────────────────────────────────────────┤
  │ send_prios() Process        │ comm.isend(TAG_PRIOS)                    │
  │   DEALER socket :51002      │   (inline in train loop, no subprocess)  │
  │   multiprocessing.Queue     │                                          │
  └─────────────────────────────┴──────────────────────────────────────────┘

Rank layout (must match actor_mpi.py / replay_mpi.py):
    Rank 0             → Learner  (this file)
    Rank 1             → Replay buffer
    Ranks 2 .. N-2     → Actors   (actor_id = rank - 2)
    Rank N-1           → Evaluator
"""

import time
import os

import torch
import numpy as np
from mpi4py import MPI
from tensorboardX import SummaryWriter

import utils
import wrapper
from model import DuelingDQN
from arguments_new import argparser

# ── Rank constants (keep in sync with actor_mpi.py / replay_mpi.py) ──────────
LEARNER_RANK = 0
REPLAY_RANK  = 1
# Actor ranks are 2 .. (size - 2); evaluator is (size - 1).

# ── Message tags (keep in sync across all MPI files) ─────────────────────────
TAG_BATCH      = 10   # actor  → replay  : (batch, prios) experience bundle
TAG_PARAMS     = 20   # learner → actor  : state_dict with updated weights
TAG_SAMPLE_REQ = 30   # learner → replay : "please send me a training batch"
TAG_SAMPLE     = 31   # replay  → learner: sampled training batch
TAG_PRIOS      = 40   # learner → replay : updated priority scores


# ─────────────────────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────────────────────

def _actor_and_eval_ranks(comm: MPI.Comm) -> list[int]:
    """Return all ranks that should receive parameter updates."""
    size = comm.Get_size()
    # Ranks 2..(size-2) are actors; rank (size-1) is evaluator.
    return list(range(2, size))


def _broadcast_params(comm: MPI.Comm, model: torch.nn.Module) -> list[MPI.Request]:
    """
    Send the current model weights to every actor and evaluator rank.

    Replaces the ZMQ PUB socket.  Uses isend (non-blocking) so the learner
    never stalls waiting for a slow actor to call recv().  Returns the list
    of Request handles; the caller can call MPI.Request.Waitall(reqs) if it
    needs to confirm delivery, but in practice we just fire-and-forget.

    The ZMQ version used CONFLATE=1, meaning actors only ever saw the latest
    weights if the queue backed up.  We replicate that by not waiting for
    acknowledgements — if an actor hasn't consumed its previous params message
    yet, MPI buffers the new one.  Actors drain it with iprobe (see actor_mpi).
    """
    # Move all tensors to CPU before sending: actors run on CPU and
    # serialising CUDA tensors across processes is fragile.
    params = {k: v.cpu() for k, v in model.state_dict().items()}
    reqs = []
    for dest in _actor_and_eval_ranks(comm):
        req = comm.isend(params, dest=dest, tag=TAG_PARAMS)
        reqs.append(req)
    return reqs


def _batch_to_device(batch: list, device: torch.device) -> tuple:
    states, actions, rewards, next_states, dones, weights, idxes = batch

    # ── FIX: Explicitly cast to float32 to prevent 64-bit memory spikes ───────
    states      = torch.from_numpy(np.stack([np.asarray(s, dtype=np.float32) for s in states])).to(device)
    actions     = torch.LongTensor(actions).to(device)
    rewards     = torch.FloatTensor(rewards).to(device)
    next_states = torch.from_numpy(np.stack([np.asarray(s, dtype=np.float32) for s in next_states])).to(device)
    dones       = torch.FloatTensor(dones).to(device)
    weights     = torch.FloatTensor(weights).to(device)

    return states, actions, rewards, next_states, dones, weights, idxes


# ─────────────────────────────────────────────────────────────────────────────
# Main learner loop
# ─────────────────────────────────────────────────────────────────────────────

def learner_main(comm: MPI.Comm, args) -> None:
    """
    Full training loop for the learner rank (rank 0).

    Parameters
    ----------
    comm : MPI.Comm
        The global MPI communicator (MPI.COMM_WORLD).
    args : argparse.Namespace
        Parsed arguments from argparser().
    """
    # ── Model setup ───────────────────────────────────────────────────────────
    env = wrapper.make_atari(args.env)
    env = wrapper.wrap_atari_dqn(env, args)
    utils.set_global_seeds(args.seed, use_torch=True)

    model     = DuelingDQN(env).to(args.device)
    tgt_model = DuelingDQN(env).to(args.device)
    tgt_model.load_state_dict(model.state_dict())

    optimizer = torch.optim.RMSprop(
        model.parameters(), args.lr,
        alpha=0.95, eps=1.5e-7, centered=True
    )
    writer = SummaryWriter(comment="-{}-learner".format(args.env))

    # ── Startup sync ──────────────────────────────────────────────────────────
    # Replaces check_connection() + the ROUTER socket on port 52002.
    # Every rank (learner, replay, all actors, evaluator) calls comm.Barrier()
    # in its own _main function.  Only when all ranks have reached their
    # Barrier does any of them proceed.  This guarantees every rank is alive
    # and initialised before the first weight broadcast goes out.
    print("[Learner] Waiting at barrier for all ranks...", flush=True)
    comm.Barrier()
    print("[Learner] All ranks ready. Broadcasting initial parameters.", flush=True)

    # ── Initial weight broadcast ──────────────────────────────────────────────
    # In the ZMQ version, train() put the initial state_dict into param_queue,
    # and the send_param() process picked it up and published via PUB socket.
    # Here we push directly to every actor + evaluator rank with isend.
    # We call Waitall so the first training step doesn't start before every
    # actor has at least been offered the weights (they may not have recv()'d
    # yet, but the messages are in flight).
    init_reqs = _broadcast_params(comm, model)
    MPI.Request.Waitall(init_reqs)
    print("[Learner] Initial parameters sent to all actors.", flush=True)

    # ── Training loop ─────────────────────────────────────────────────────────
    learn_idx = 0
    ts        = time.time()

    # Track outstanding isend requests for params so we don't leak handles.
    param_reqs: list[MPI.Request] = []

    while True:
        # ── 1. Request a training batch from replay ───────────────────────────
        # Sends a tiny sentinel message to replay (TAG_SAMPLE_REQ) so replay
        # knows to prepare and send a batch back.  Replaces the DEALER socket
        # on port 51003 and the recv_batch() Process + Thread + Queue chain.
        comm.send(None, dest=REPLAY_RANK, tag=TAG_SAMPLE_REQ)

        # ── 2. Receive the sample batch ───────────────────────────────────────
        # Blocks until replay responds with TAG_SAMPLE.  This is intentional:
        # the learner should not proceed without data.  If replay isn't ready
        # yet (buffer still warming up) it simply won't respond to
        # TAG_SAMPLE_REQ until it has enough data — the learner waits.
        raw_batch = comm.recv(source=REPLAY_RANK, tag=TAG_SAMPLE)

        # ── 3. Move batch to GPU ──────────────────────────────────────────────
        *batch, idxes = _batch_to_device(raw_batch, args.device)
        raw_batch = None

        # ── 4. Compute loss and update weights ────────────────────────────────
        loss, prios = utils.compute_loss(model, tgt_model, batch, args.n_steps, args.gamma)
        grad_norm   = utils.update_parameters(loss, model, optimizer, args.max_norm)
        batch       = None

        # ── 5. Send updated priorities back to replay ─────────────────────────
        # isend so we don't block the training loop waiting for replay to recv.
        # We track the request and free it next iteration to avoid handle leaks.
        prio_req = comm.isend((idxes, prios.tolist()), dest=REPLAY_RANK, tag=TAG_PRIOS)
        prio_req.Free()  # fire-and-forget: MPI manages the buffer lifetime

        idxes = prios = None

        learn_idx += 1

        # ── 6. Logging ────────────────────────────────────────────────────────
        writer.add_scalar("learner/loss",      loss,      learn_idx)
        writer.add_scalar("learner/grad_norm", grad_norm, learn_idx)

        # ── 7. Periodic: update frozen target network ─────────────────────────
        if learn_idx % args.target_update_interval == 0:
            print("[Learner] Updating target network..", flush=True)
            tgt_model.load_state_dict(model.state_dict())

        # ── 8. Periodic: save model checkpoint ───────────────────────────────
        if learn_idx % args.save_interval == 0:
            print("[Learner] Saving model..", flush=True)
            torch.save(model.state_dict(), "model.pth")

        # ── 9. Periodic: broadcast updated weights to actors ─────────────────
        # Replaces the send_param() Process + PUB socket + param_queue.
        # Clean up any completed handles from the last broadcast first to
        # avoid accumulating thousands of Request objects in memory.
        if learn_idx % args.publish_param_interval == 0:
            # Drain handles that are already done (non-blocking test).
            param_reqs = [r for r in param_reqs if not r.Test()]
            new_reqs   = _broadcast_params(comm, model)
            param_reqs.extend(new_reqs)

        # ── 10. Periodic: log batches-per-second ─────────────────────────────
        if learn_idx % args.bps_interval == 0:
            bps = args.bps_interval / (time.time() - ts)
            print("[Learner] Step: {:8} / BPS: {:.2f}".format(learn_idx, bps), flush=True)
            writer.add_scalar("learner/BPS", bps, learn_idx)
            ts = time.time()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point (used when running learner_mpi.py standalone for testing)
# In the full Ape-X MPI job this is called from apex_mpi.py by rank 0.
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    os.environ["OMP_NUM_THREADS"] = "1"
    comm = MPI.COMM_WORLD
    args = argparser()
    learner_main(comm, args)


if __name__ == "__main__":
    main()