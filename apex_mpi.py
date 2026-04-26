import os
from mpi4py import MPI
from arguments_new import argparser

# Import the core loops from the files you provided
from learner_new import learner_main
from replay_new import replay_main
from actor_new import actor_main
from eval_new import eval_main

# Assuming you have a translated eval script
# from eval_new import eval_main 

def main():
    # Crucial to prevent CPU thrashing when running many actors on one machine
    os.environ["OMP_NUM_THREADS"] = "1"
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    args = argparser()

    # Minimum: learner(1) + replay(1) + actor(1) + evaluator(1) = 4 ranks
    assert size >= 4, (
        f"Need at least 4 ranks, got {size}. "
        f"Run with: mpirun -n 11 --oversubscribe python apex_mpi.py"
    )

    # The rank topology defined in your files
    if rank == 0:
        learner_main(comm, args)
    elif rank == 1:
        replay_main(comm, args)
    elif rank == size - 1:
        # Last rank is always the evaluator
        eval_main(comm, args)
    else:
        # Ranks 2 through (size - 2) become actors
        actor_main(comm, args)

if __name__ == "__main__":
    main()