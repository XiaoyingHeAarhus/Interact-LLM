"""
Simulate two chat LLMs talking to each other

Model tested currently: 
- mlx-community/Qwen2.5-7B-Instruct-1M-4bit
- mlx-community/meta-Llama-3.1-8B-Instruct-4bit

"""
from datetime import datetime
import json
from pathlib import Path
import argparse

from interact_llm.llm.mlx_wrapper import ChatMLX
from interact_llm.data_models.chat import ChatMessage, ChatHistory
from interact_llm.data_models.prompt import load_prompt_by_id

from tqdm import tqdm

DEFAULT_PROMPT_VERSION = 3.0

def input_parse():
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument(
        "--prompt_id", help="id of prompt in toml", type=str, default="A1"
    )
    parser.add_argument(
        "--prompt_version", help="version of prompt toml file", type=float, default=DEFAULT_PROMPT_VERSION
    )

    parser.add_argument(
        "--model_id", help="model id as it is specified on hugging face", type=str, 
        default = "mlx-community/Qwen2.5-7B-Instruct-1M-4bit"
    )

    # save arguments to be parsed from the CLI
    args = parser.parse_args()

    return args

def main():
    # init cli args
    args = input_parse()

    # PROMPT FORMATTING
    prompt_version = args.prompt_version
    prompt_id = args.prompt_id
    prompt_file = (
        Path(__file__).parents[2]
        / "configs"
        / "prompts"
        / f"v{str(prompt_version)}.toml"
    )

    print(
        f"[INFO]: Formatting prompts using toml file version {prompt_version} and prompt id {prompt_id}"
    )

    system_prompt = load_prompt_by_id(
        toml_path=prompt_file, prompt_id=prompt_id, system_prompt=True
    )

    # define chat histories 
    student_history = ChatHistory(messages=[ChatMessage(role = "system", content="You are a student learning Spanish, responding to a teacher who is facilitating a natural dialogue with you.")])
    
    teacher_history = ChatHistory(
        messages=[ChatMessage(role=system_prompt.role, content=system_prompt.content), 
                  ChatMessage(role = "user", content="Hola"),
                  ]
    )

    # load model with MLX
    sampling_params = {"temp": 0.8, "top_p": 0.95, "min_p": 0.95, "top_k": 40}
    penality_params = {"repetition_penalty": 1.1}

    model_id = args.model_id
    model = ChatMLX(
            model_id=model_id,
            sampling_params=sampling_params,
            penalty_params=penality_params,
        )
    print(f"[INFO]: Loading model {model_id} ... please wait")
    model.load()

    # how many responses in total
    n_total_rounds = 9
    
    for _ in tqdm(range(n_total_rounds)):
        # assistant responds to user
        teacher_message = model.generate(teacher_history) # assistant
        teacher_history.messages.append(teacher_message) 

        # format as a user msg for assistant
        teacher_as_user = ChatMessage(role = "user", content = teacher_message.content)
        student_history.messages.append(teacher_as_user)
        
        # get student response, append to teacher chat history
        student_message = model.generate(student_history) # assistant 
        student_history.messages.append(student_message)

        student_as_user = ChatMessage(role = "user", content = student_message.content)
        teacher_history.messages.append(student_as_user)

    # save chat
    chat_json = json.dumps(
                        [msg.dict() for msg in teacher_history.messages],
                        indent=3,
                        ensure_ascii=False,
                    )
    
    save_dir = (
        Path(__file__).parents[3]
        / "simulated_data"
        / model_id.replace("/", "--")
        / f"v{str(prompt_version)}"
        / prompt_id
    )

    save_dir.mkdir(exist_ok=True, parents=True)

    save_file_name = datetime.now().strftime("%Y%m%d-%H%M%S")
    with open(save_dir / f"{save_file_name}.json", "w") as outfile:
        outfile.write(chat_json)

if __name__ == "__main__":
    main()