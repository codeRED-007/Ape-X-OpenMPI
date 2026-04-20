#!/bin/bash

SESSION="apex"

# Create a new tmux session detached, naming the first window 'replay'
tmux new-session -s $SESSION -n replay -d
tmux send-keys -t $SESSION:replay ' python replay.py; read' C-m

# Create a new window for the learner and send the command
tmux new-window -t $SESSION -n learner
tmux send-keys -t $SESSION:learner ' REPLAY_IP="127.0.0.1" N_ACTORS=8 python learner.py --cuda; read' C-m

# Loop to create windows for all 8 actors
for i in {0..7}; do
    tmux new-window -t $SESSION -n "actor${i}"
    tmux send-keys -t $SESSION:"actor${i}" " REPLAY_IP=\"127.0.0.1\" LEARNER_IP=\"127.0.0.1\" ACTOR_ID=${i} N_ACTORS=8 python actor.py; read" C-m
done

# Create a new window for the evaluator
tmux new-window -t $SESSION -n evaluator
tmux send-keys -t $SESSION:evaluator ' LEARNER_IP="127.0.0.1" python eval.py; read' C-m

# Finally, attach to the tmux session
tmux attach-session -t $SESSION