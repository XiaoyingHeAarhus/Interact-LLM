for run in {1..10}; do
    uv run python src/scripts/simulate.py --prompt_id "A1"
done

for run in {1..10}; do
    uv run python src/scripts/simulate.py --prompt_id "B1"
done

for run in {1..10}; do
    uv run python src/scripts/simulate.py --prompt_id "C1"
done
