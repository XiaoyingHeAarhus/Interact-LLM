# Interact-LLM: Inference Experiments
This repository contains code for experiments exploring how large language models (LLMs) perform as cognitive tutors for language learning. All experiments are connected to the `INTERACT-LLM` project. See also the [Research](#-research) section.

>Note: The code is currently *only* intended for internal use and is not production-ready.

🔗 [Read about the INTERACT-LLM project (in Danish)](https://cc.au.dk/aktuelt/nyheder/nyhed/artikel/forskere-fra-arts-vil-udvikle-en-chatbot-der-kan-fremme-sprogindlaering),

# 🚀 Overview
The `src` folder contains:
| Folder | Description |
|--------|-------------|
| [`interact_llm`](src/interact_llm/) | Inference engine and terminal-based chatbot |
| [`scripts`](src/scripts/) | Experimental setups using the backend, including [alignment-drift-simulation](src/scripts/alignment-drift-simulation) created for [Almasi & Kristensen-McLachlan (2025)](https://arxiv.org/abs/2505.08351).|

# 📝 Research 
The `Interact-LLM` repository is version-tagged with distinct releases pertaining to particular publications; 
| Paper | `Interact-LLM` version| `Interact-LLM` experiment | Paper Repo (Analysis) |
|--------|---------------------|------------| ------ |
|[Almasi & Kristensen-McLachlan (2025)](https://arxiv.org/abs/2505.08351)| `v1.0.0-alignment-drift` | [scripts/alignment-drift-simulation]() | [INTERACT-LLM/alignment-drift-llms](https://github.com/INTERACT-LLM/alignment-drift-llms)|

# 🛠️ Technical Requirements
The code was run on `Python 3.12.3` on both a macOS (`15.3.1`) and Ubuntu system (`24.04`). The project also requires:
| Tool     | Installation                                                                 |
|----------|--------------------------------------------------------------------------------------|
| [make](https://www.gnu.org/software/make/manual/make.html) | Installed via [Homebrew](https://formulae.brew.sh/formula/make)                  |
| [uv](https://docs.astral.sh/uv/)                         | Installed through this project's `makefile` (see [Usage](#usage))                 |


# ⚙️ Usage 

## Setup project 
To install [`uv`](https://docs.astral.sh/uv/) on macOS/Linux and set up a virtual environment with the required Python dependencies, run in the terminal:
```bash
make setup
```

## Run Chatbot in Terminal App
To experiment with interacting with a chatbot (prompted to act as a Spanish language tutor currently), run in the terminal 
```bash
uv run python -m interact_llm 
```

## Reproduce Experiments 
Refer to the individual READMEs in the folders in `scripts` e.g., [alignment-drift-simulation](src/scripts/alignment-drift-simulation).