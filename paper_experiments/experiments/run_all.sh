#!/bin/bash

TASKS=("docstring" "greater-than" "induction" "ioi" "tracr-proportion" "tracr-reverse")
TESTS=("minimality" "non-equivalence" "non-independence" "partial-necessity" "sufficiency")
SEED=1

len=${#TASKS[@]}
test_len=${#TESTS[@]}

for (( i=0; i<$len; i++ )); do
    TASK=${TASKS[$i]}
    for (( j=0; j<$test_len; j++ )); do
        TEST=${TESTS[$j]}
        echo "Task: $TASK | Test: $TEST | Seed: $SEED"
        sbatch run_job.sh $TASK $TEST $SEED
    done
done
