import argparse
import torch

def argparser():
    parser = argparse.ArgumentParser(description='Ape-X MPI')

    # Common Arguments
    parser.add_argument('--seed', type=int, default=1122)
    parser.add_argument('--n_steps', type=int, default=3)
    parser.add_argument('--gamma', type=float, default=0.99)

    # Environment Arguments
    parser.add_argument('--env', type=str, default="SeaquestNoFrameskip-v4")
    parser.add_argument('--episode_life', type=int, default=1)
    parser.add_argument('--clip_rewards', type=int, default=1)
    parser.add_argument('--frame_stack', type=int, default=1)
    parser.add_argument('--scale', type=int, default=0)

    # Arguments for Actor
    parser.add_argument('--send_interval', type=int, default=50)
    parser.add_argument('--update_interval', type=int, default=400)
    parser.add_argument('--max_episode_length', type=int, default=50000)
    parser.add_argument('--eps_base', type=float, default=0.4)
    parser.add_argument('--eps_alpha', type=float, default=7.0)

    # Arguments for Replay
    parser.add_argument('--alpha', type=float, default=0.6)
    parser.add_argument('--beta', type=float, default=0.4)
    parser.add_argument('--replay_buffer_size', type=int, default=2000000)
    parser.add_argument('--threshold_size', type=int, default=50000)
    parser.add_argument('--batch_size', type=int, default=512)

    # Arguments for Learner
    parser.add_argument('--lr', type=float, default=6.25e-5)
    parser.add_argument('--max_norm', type=float, default=40.0)
    parser.add_argument('--cuda', action='store_true', default=False)
    parser.add_argument('--target_update_interval', type=int, default=2500)
    parser.add_argument('--publish_param_interval', type=int, default=25)
    parser.add_argument('--save_interval', type=int, default=5000)
    parser.add_argument('--bps_interval', type=int, default=100)

    # Arguments for Evaluation
    parser.add_argument('--render', action='store_true', default=False)

    args = parser.parse_args()
    args.cuda = args.cuda and torch.cuda.is_available()
    args.device = torch.device("cuda" if args.cuda else "cpu")

    return args