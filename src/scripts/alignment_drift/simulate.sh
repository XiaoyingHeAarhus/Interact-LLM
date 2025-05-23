#!/bin/bash

models=("qwen2.5:7b" "llama3.1:8b" "mistral:7b" "gemma3:12b")
prompt_ids=("A1" "B1" "C1")
backend="hf"

for model in "${models[@]}"; do
    for prompt_id in "${prompt_ids[@]}"; do
            uv run python src/scripts/alignment_drift/simulate.py --model_name "$model" --prompt_id "$prompt_id" --backend "$backend"
    done
done