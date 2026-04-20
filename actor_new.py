"""
actor_mpi.py — Actor rank implementation for Ape-X with OpenMPI.

Replaces actor.py. Each actor rank:
  - Receives initial model weights from learner via MPI_Barrier + bcast-style isend
  - Plays the Atari game using a local DuelingDQN (inference only, CPU)
  - Sends experience batches to the replay rank (TAG_BATCH)
  - Polls for updated weights from learner (TAG_PARAMS) every update_interval steps

No ZMQ. No subprocesses. No multiprocessing.Queue.
The two roles that were split into two Processes in the ZMQ version
(exploration + recv_param) are now interleaved in a single loop using
comm.iprobe() for non-blocking param checks.

Rank layout (must match across all files):
    Rank 0           → Learner
    Rank 1           → Replay buffer
    Ranks 2..(N-2)   → Actors  (actor_id = rank - 2)
    Rank N-1         → Evaluator
"""

import os
import torch
import numpy as np
from mpi4py import MPI
from tensorboardX import SummaryWriter

import utils
from memory import BatchStorage
from wrapper import make_atari, wrap_atari_dqn
from model import DuelingDQN
from arguments_new import argparser

# ── Rank constants (keep in sync with learner_mpi.py / replay_mpi.py) ────────
LEARNER_RANK = 0
REPLAY_RANK  = 1

# ── Message tags (keep in sync across all MPI files) ─────────────────────────
TAG_BATCH        = 10   # actor  → replay  : (batch, prios) experience bundle
TAG_PARAMS       = 20   # learner → actor  : state_dict with updated weights
TAG_SAMPLE_REQ   = 30   # learner → replay : request for a training batch
TAG_SAMPLE       = 31   # replay  → learner: sampled training batch
TAG_PRIOS        = 40   # learner → replay : updated priority scores


def actor_main(comm: MPI.Comm, args) -> None:
    """
    Main loop for a single actor rank.

    Parameters
    ----------
    comm : MPI.Comm
        The global MPI communicator (MPI.COMM_WORLD).
    args : argparse.Namespace
        Parsed arguments from argparser().
    """
    rank     = comm.Get_rank()
    n_actors = comm.Get_size() - 3          # total ranks minus learner, replay, evaluator
    actor_id = rank - 2                     # rank 2 → actor_id 0, rank 3 → actor_id 1, …

    # ── Environment + model setup ─────────────────────────────────────────────
    env = make_atari(args.env)
    env = wrap_atari_dqn(env, args)

    seed = args.seed + actor_id
    utils.set_global_seeds(seed, use_torch=True)
    if hasattr(env, "seed"):
        env.seed(seed)

    model   = DuelingDQN(env)               # CPU-only; actors never need GPU
    storage = BatchStorage(args.n_steps, args.gamma)
    writer  = SummaryWriter(comment="-{}-actor{}".format(args.env, actor_id))

    # Each actor uses a different epsilon so the replay buffer sees diverse
    # exploration behaviour.  Actor 0 is near-greedy; the last actor explores
    # almost randomly.
    epsilon = args.eps_base ** (1 + actor_id / max(n_actors - 1, 1) * args.eps_alpha)

    # ── Startup sync: wait for ALL ranks to be ready ──────────────────────────
    # This replaces the ZMQ REQ/ROUTER handshake in connect_param_socket().
    comm.Barrier()

    # ── Receive initial weights from learner ──────────────────────────────────
    # The learner does comm.send(params, dest=rank, tag=TAG_PARAMS) for every
    # actor and evaluator rank right after the Barrier.
    params = comm.recv(source=LEARNER_RANK, tag=TAG_PARAMS)
    model.load_state_dict(params)
    print(f"[Actor {actor_id}] Received initial parameters from learner.", flush=True)

    # ── Main game loop ────────────────────────────────────────────────────────
    episode_reward = 0
    episode_length = 0
    episode_idx    = 0
    actor_idx      = 0

    state, _ = env.reset()

    while True:
        # ── Step the environment ──────────────────────────────────────────────
        action, q_values = model.act(torch.FloatTensor(np.array(state)), epsilon)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        storage.add(state, reward, action, done, q_values)

        state           = next_state
        episode_reward += reward
        episode_length += 1
        actor_idx      += 1

        # ── Episode bookkeeping ───────────────────────────────────────────────
        if done or episode_length == args.max_episode_length:
            state, _ = env.reset()
            writer.add_scalar("actor/episode_reward", episode_reward, episode_idx)
            writer.add_scalar("actor/episode_length", episode_length, episode_idx)
            episode_reward = 0
            episode_length = 0
            episode_idx   += 1

        # ── Poll for updated weights (non-blocking) ───────────────────────────
        # In the ZMQ version this was a separate Process (recv_param) passing
        # weights through a multiprocessing.Queue.  Here we fold it into the
        # main loop with iprobe so we never block the game loop.
        #
        # iprobe returns True if a message is waiting but does NOT consume it.
        # We only call recv() when we know something is there, so this is
        # always instant.
        if actor_idx % args.update_interval == 0:
            if comm.iprobe(source=LEARNER_RANK, tag=TAG_PARAMS):
                params = comm.recv(source=LEARNER_RANK, tag=TAG_PARAMS)
                model.load_state_dict(params)
                print(f"[Actor {actor_id}] Updated parameters.", flush=True)

        # ── Send experience batch to replay ───────────────────────────────────
        # In the ZMQ version an "outstanding" counter enforced backpressure
        # (don't queue more than max_outstanding unacknowledged sends).
        # With MPI isend the message goes into MPI's internal buffer and
        # returns immediately; we just fire and forget here.  If you need
        # strict backpressure you can track Request objects and call
        # req.Wait() before issuing the next send.
        if len(storage) == args.send_interval:
            batch, prios = storage.make_batch()
            storage.reset()

            # isend: non-blocking — does not wait for replay to recv()
            req = comm.isend((batch, prios), dest=REPLAY_RANK, tag=TAG_BATCH)
            # print(f"[Actor {actor_id}] Sent batch to replay.", flush=True)


def main() -> None:
    comm = MPI.COMM_WORLD
    args = argparser()
    os.environ["OMP_NUM_THREADS"] = "1"
    actor_main(comm, args)


if __name__ == "__main__":
    main()