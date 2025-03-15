"""
Chat Model
"""

from datetime import datetime
from pathlib import Path
from typing import Optional

from interact_llm.data_models.chat import ChatMessage
from transformers import AutoModelForCausalLM, AutoTokenizer


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
                device_map=self.device_map if self.device_map else None,
                cache_dir=self.cache_dir,
                torch_dtype="auto",
                device_map="auto"
            )

    def format_params(self):
        if self.sampling_params:         
            kwargs = self.sampling_params
        else:
            kwargs = {}

        if self.penalty_params:           
            kwargs.update(self.penalty_params)

        return kwargs

    def generate(self, chat: list[ChatMessage], max_new_tokens: int = 200):
        kwargs = self.format_params()

        prompt = self.tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True,
        )

        # tokenized inputs and outputs
        token_inputs = self.tokenizer.encode(
            prompt, add_special_tokens=False, return_tensors="pt"
        ).to(self.model.device)

        token_outputs = self.model.generate(
            input_ids=token_inputs.to(self.model.device), max_new_tokens=max_new_tokens, 
            **kwargs
        )

        # chat (decoded output)
        response = self.tokenizer.decode((token_outputs[:, token_inputs.shape[1] :])[0])

        chat_message = ChatMessage(role="assistant", content=response)

        return chat_message
