# Alignment Drift in CEFR-prompted LLMs for Interactive Spanish Tutoring (Almasi & Kristensen-McLachlan, 2025)
This folder contains the scripts used to simulate teacher-student dialogues with a single LLM, featuring interchangeable "student" and "teacher" roles, as described in the paper *"Alignment Drift in CEFR-prompted LLMs for Interactive Spanish Tutoring"* [(Almasi & Kristensen-McLachlan, 2025)](https://arxiv.org/abs/2505.08351). 

| ⚠️ **IMPORTANT** |
|------------------|
| The scripts are only guaranteed to work with repository versions that include `"alignment-drift"` in the tag (e.g., `v1.0.3-alignment-drift`). Be sure to clone `Interact-LLM` using the correct tag.|

For the dataset and analysis of the resulting simulations, see [INTERACT-LLM/alignment-drift-llms](https://github.com/INTERACT-LLM/alignment-drift-llms).

# 🚀️ Overview
| File                   | Description                                                                 |
|------------------------------|-----------------------------------------------------------------------------|
| `detect_lang.py`          | Util script. Simple detection of string containing English or Mandarin Chinese. Used to re-generate responses if they are not purely in Spanish in the dialogue simulations (`simulate.py`). |
| `simulate.py`         | Script to simulate teacher-student dialogues with a single LLM for a single prompt-id (see also [configs/prompts/v3.0.toml](/configs/prompts/v3.0.toml)).       |
| `simulate.sh`                   | Bash script to run `simulate.py` with all `model` and `prompt_id` combinations (30 dialogues for each combination). |

>Note: The particular LLMs that are supported and can be run through `simulate.py` are defined in [configs/models.toml](/configs/models.toml). You can define additional models in the toml file, but they are not guaranteed to work.

# ⚙️ Usage 
Prior to running any code, follow the technical requirements and setup described in the [main README](/README.md).

>Note: Some models are gated, and you will therefore need a file `hf_token.txt` in the [tokens](/tokens) folder that contains a HuggingFace token with read access.

## Running the Entire Experiment
From root, change directory to the `alignment-drift` folder:
```bash
cd src/scripts/alignment_drift
```

Run `simulate.sh` in the terminal:
```bash
bash simulate.sh
```

If you wish to run from root and not change directory, type:
```bash
bash src/scripts/alignment_drift/simulate.sh
```

## Running `simulate.py`
You can also run the 'simulate.py' which will only run a single model (`qwen2.5:7b`) with a single prompt level (`A1`) as default.


After having navigated to the folder (`cd src/scripts/alignment_drift`), type:
```bash
uv run python simulate.py 
```

If you do not want to run with the default settings, you can specify arguments:
```
uv run python simulate.py --model_name {} --prompt_id {} --prompt_version {} --backend {}
```
Where `--model_name` refers to models such as `gemma3:12b`. These names need to already be specified in [configs/models.toml](/configs/models.toml). Similarly, the script only accepts values for `--prompt_id` and `--prompt_version` that exist in [configs/prompts](/configs/prompts/) in the desired format.

`--backend` can be either `'mlx'`for Apple Silicon optimisation or `'hf'` to rely on the [transformers](https://github.com/huggingface/transformers) library. [(Almasi & Kristensen-McLachlan, 2025)](https://arxiv.org/abs/2505.08351) used only `'hf'`. 

> Note: `'mlx'` can only be used if the model is supported in the backend and the code is run on a `macOS` system with Apple Silicon hardware.

# 🧪 Analysis 
Refer to the paper repository [INTERACT-LLM/alignment-drift-llms](https://github.com/INTERACT-LLM/alignment-drift-llms) for the dataset and analysis of the simulations.

# 📝 Citation 
Refer to the paper repository for [how to cite us](https://github.com/INTERACT-LLM/alignment-drift-llms?tab=readme-ov-file#-citation) !

# ✨️ Acknowledgements
Simulations in [Almasi & Kristensen-McLachlan (2025)](https://arxiv.org/abs/2505.08351) were run entirely using Huggingface's [transformers](https://github.com/huggingface/transformers) library (specified as a `hf`).

Language detection was made possible through the Python library [lingua-py](https://github.com/pemistahl/lingua-py).