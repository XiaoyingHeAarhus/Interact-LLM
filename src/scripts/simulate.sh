#!/bin/bash

# default number of runs
DEFAULT_RUNS=10

# use first argument as N_RUNS, or default to DEFAULT_RUNS if not provided
N_RUNS=${1:-$DEFAULT_RUNS}

models=("mlx-community/meta-Llama-3.1-8B-Instruct-4bit" "mlx-community/Qwen2.5-7B-Instruct-1M-4bit")
prompt_ids=("A1" "B1" "C1")

for model in "${models[@]}"; do
    for prompt_id in "${prompt_ids[@]}"; do
        for ((run=1; run<=N_RUNS; run++)); do
            uv run python src/scripts/simulate.py --model "$model" --prompt_id "$prompt_id"
        done
    done
done
