#!/bin/sh

# Number of runs per profile
NUM_RUNS=5

# Model to use
MODEL="deepseek"

# Run Nordic profile experiments
for i in $(seq 1 $NUM_RUNS); do
    echo "Running Nordic profile experiment $i of $NUM_RUNS..."
    python run_experiment.py \
        --model $MODEL \
        --profile-type nordic \
        --num-agents 4 \
        --num-generations 10 \
        --run-index $i &
done

# Run East Asian profile experiments
for i in $(seq 1 $NUM_RUNS); do
    echo "Running East Asian profile experiment $i of $NUM_RUNS..."
    python run_experiment.py \
        --model $MODEL \
        --profile-type east_asian \
        --num-agents 4 \
        --num-generations 10 \
        --run-index $((i + NUM_RUNS)) &
done

# Wait for all background processes to complete
wait

echo "All experiments completed!" 