"""
eval_mpi.py — Evaluator rank implementation for Ape-X with OpenMPI.

Replaces eval.py. Evaluates the learner's performance with epsilon=0.
"""

import os
import torch
import numpy as np
from mpi4py import MPI
from tensorboardX import SummaryWriter

import utils
from wrapper import make_atari, wrap_atari_dqn
from model import DuelingDQN
from arguments_new import argparser

LEARNER_RANK = 0
TAG_PARAMS   = 20

def eval_main(comm: MPI.Comm, args) -> None:
    rank = comm.Get_rank()
    
    # ── Environment setup ─────────────────────────────────────────────────────
    args.clip_rewards = False  # We want the true game score for logging
    env = make_atari(args.env)
    env = wrap_atari_dqn(env, args)

    # Use a fixed seed for the evaluator to ensure consistent test conditions
    seed = args.seed + 999 
    utils.set_global_seeds(seed, use_torch=True)
    if hasattr(env, "seed"):
        env.seed(seed)

    model = DuelingDQN(env)
    writer = SummaryWriter(comment="-{}-eval".format(args.env))

    # ── Startup sync ──────────────────────────────────────────────────────────
    comm.Barrier()

    # ── Receive initial weights ───────────────────────────────────────────────
    params = comm.recv(source=LEARNER_RANK, tag=TAG_PARAMS)
    model.load_state_dict(params)
    print(f"[Evaluator] Received initial parameters from learner.", flush=True)

    episode_reward = 0
    episode_length = 0
    episode_idx    = 0

    state, _ = env.reset()

    while True:
        # Epsilon is 0.0 -> Pure exploitation
        action, _ = model.act(torch.FloatTensor(np.array(state)), 0.0)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        state = next_state
        episode_reward += reward
        episode_length += 1

        if done or episode_length == args.max_episode_length:
            state, _ = env.reset()
            writer.add_scalar("evaluator/episode_reward", episode_reward, episode_idx)
            writer.add_scalar("evaluator/episode_length", episode_length, episode_idx)
            
            print(f"[Evaluator] Episode {episode_idx} | Score: {episode_reward} | Length: {episode_length}")
            
            episode_reward = 0
            episode_length = 0
            episode_idx   += 1

            # Drain any accumulated parameter updates from the network buffer
            # and load the absolute latest weights before starting the next episode
            has_new_params = False
            while comm.iprobe(source=LEARNER_RANK, tag=TAG_PARAMS):
                params = comm.recv(source=LEARNER_RANK, tag=TAG_PARAMS)
                has_new_params = True
            
            if has_new_params:
                model.load_state_dict(params)
                print(f"[Evaluator] Updated parameters for next episode.", flush=True)


def main() -> None:
    os.environ["OMP_NUM_THREADS"] = "1"
    comm = MPI.COMM_WORLD
    args = argparser()
    eval_main(comm, args)


if __name__ == "__main__":
    main()