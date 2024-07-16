#!/bin/bash

# File to store parameter combinations
PARAM_FILE="euc_combinations.txt"

# Clear the file if it exists
> $PARAM_FILE

# Generate all combinations
for hyperbolic in true; do
    for depth in 9; do
        for embedding_dim in 4 8; do
            echo "$hyperbolic $depth $embedding_dim" >> $PARAM_FILE
        done
    done
done

# Count total number of combinations
TOTAL_JOBS=$(wc -l < $PARAM_FILE)
echo "Total number of jobs: $TOTAL_JOBS"