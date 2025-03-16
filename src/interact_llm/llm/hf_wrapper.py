"""
Chat Model
"""

from pathlib import Path
from typing import Optional

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import logging

from interact_llm.data_models.chat import ChatMessage

# supress attention mask warning - only show errors (warning not important, it sets it automatically)
logging.set_verbosity_error()


class ChatHF:
    """
    Model wrapper for loading and using a HuggingFace causal language model with HF's own libraries
    """

    def __init__(
        self,
        model_id: str,
        cache_dir: Optional[Path] = None,
        sampling_params: Optional[dict] = None,
        penalty_params: Optional[dict] = None,
    ):
        self.model_id = model_id
        self.cache_dir = cache_dir
        self.tokenizer = None
        self.model = None
        self.sampling_params = sampling_params
        self.penalty_params = penalty_params

    def load(self) -> None:
        """
        Lazy-loading (loads model and tokenizer if not already loaded)
        """
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_id, cache_dir=self.cache_dir
            )

        if self.model is None:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                cache_dir=self.cache_dir,
                torch_dtype="auto",
                device_map="auto",
            )

    def format_params(self):
        if self.sampling_params:
            # normalise "temp" to "temperature" (ensures you can pass temp to the model as this is how MLX/HF defines it)
            if "temp" in self.sampling_params:
                self.sampling_params["temperature"] = self.sampling_params.pop("temp")

            kwargs = self.sampling_params
        else:
            kwargs = {}

        if self.penalty_params:
            kwargs.update(self.penalty_params)

        return kwargs

    def generate(self, chat: list[ChatMessage], max_new_tokens: int = 200):
        kwargs = self.format_params()

        if len(kwargs) > 0:
            do_sample = True
        else:
            do_sample = False
            print(
                "[INFO:] No sampling parameters nor penalty parameters were passed. Setting do_sample to 'False'"
            )

        prompt = self.tokenizer.apply_chat_template(
            chat,
            tokenize=False,
            add_generation_prompt=True,
        )

        # tokenized inputs and outputs
        token_inputs = self.tokenizer.encode(
            prompt, add_special_tokens=False, return_tensors="pt"
        ).to(self.model.device)

        token_outputs = self.model.generate(
            input_ids=token_inputs.to(self.model.device),
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            **kwargs,
        )

        # chat (decoded output)
        response = self.tokenizer.decode(
            (token_outputs[:, token_inputs.shape[1] :])[0], skip_special_tokens=True
        )

        chat_message = ChatMessage(role="assistant", content=response)

        return chat_message
