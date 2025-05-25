#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)" # get dir of this script to make it runable from everywhere

models=("qwen2.5:7b" "llama3.1:8b" "mistral:7b" "gemma3:12b")
prompt_ids=("A1" "B1" "C1")
backend="hf"

for model in "${models[@]}"; do
    for prompt_id in "${prompt_ids[@]}"; do
        uv run python "$SCRIPT_DIR/simulate.py" \
            --model_name "$model" --prompt_id "$prompt_id" --backend "$backend"
    done
done
