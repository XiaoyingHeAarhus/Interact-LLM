# Alignment Drift in CEFR-prompted LLMs for Interactive Spanish Tutoring (Almasi & Kristensen-McLachlan, 2025)
This folder contains the scripts used to simulate teacher-student dialogues with a single LLM, featuring interchangeable "student" and "teacher" roles, as described in the paper *"Alignment Drift in CEFR-prompted LLMs for Interactive Spanish Tutoring"* [(Almasi & Kristensen-McLachlan, 2025)](https://arxiv.org/abs/2505.08351). 

> Note: This experiment only works with XX.version of this repository. Ensure you clone it with the right tag.

For the dataset and analysis of the resulting simulations, see [INTERACT-LLM/alignment-drift-llms](https://github.com/INTERACT-LLM/alignment-drift-llms).

# 🚀️ Overview
| File                   | Description                                                                 |
|------------------------------|-----------------------------------------------------------------------------|
| `detect_lang.py`          | Util script. Simple detection of string containing English or Mandarin Chinese. Used to re-generate responses if they are not purely in Spanish in the dialogue simulations (`simulate.py`). |
| `simulate.py`         | Script to simulate teacher-student dialogues with a single LLM for a single prompt-id (see also [configs/prompts/v3.0.toml](/configs/prompts/v3.0.toml)).       |
| `simulate.sh`                   | Bash script to run `simulate.py` with all `model` and `prompt_id` combinations (30 dialogues for each combination). |

> Note: The particular LLMs that are supported and can be run through `simulate.py` are defined in [configs/models.toml](/configs/models.toml). You can define additional models there, but they are not guaraneteed to work.

# ⚙️ Usage 
Prior to running any code, follow the technical requirements and setup described in the [main README](/README.md).

## Reproducing paper 
Run `simulate.sh` in the terminal:
```
bash simulate.sh
```
## Running the code
You can also run the code seperately e.g., from root with default arguments:
```
uv run python src/scripts/alignment_drift/simulate.py 
```

Or specific arguments:
```
uv run python src/scripts/alignment_drift/simulate.py --model_name {} --prompt_id {} --prompt_version {} --backend {}
```
Where `--model_name` needs to be specified in [configs/models.toml](/configs/models.toml) and `--prompt_id` + `--prompt_version` in [configs/prompts](/configs/prompts/). 

`--backend` can be either `'mlx'`or `'hf'` (`'mlx'` only if the model is supported in the backend and the code is run on a `macOS` system).

# 🧪 Analysis 
Refer to the paper repository [INTERACT-LLM/alignment-drift-llms](https://github.com/INTERACT-LLM/alignment-drift-llms) for the dataset and analysis of the simulations.

# ✨️ Acknowledgements
Simulations in [(Almasi & Kristensen-McLachlan, 2025)](https://arxiv.org/abs/2505.08351) were run entirely using the Huggingface's [transformers](https://github.com/huggingface/transformers) library (`hf`).

Language detection was made possible through the Python library [lingua-py](https://github.com/pemistahl/lingua-py).