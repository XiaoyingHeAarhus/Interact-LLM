"""
HF wrapper for Gemma 
"""

from pathlib import Path
from typing import Optional

from transformers import AutoProcessor, Gemma3ForConditionalGeneration

from interact_llm.data_models.chat import ChatMessage


class ChatGemma:
    """
    Model wrapper for loading and using a HuggingFace causal language model with HF's own libraries
    """

    def __init__(
        self,
        model_id: str = "google/gemma-3-12b-it",
        cache_dir: Optional[Path] = None,
        sampling_params: Optional[dict] = None,
        penalty_params: Optional[dict] = None,
    ):
        self.model_id = model_id
        self.cache_dir = cache_dir
        self.processor = None # multi-modal model does not have a tokenizer, but a processor 
        self.model = None
        self.sampling_params = sampling_params
        self.penalty_params = penalty_params

    def load(self) -> None:
        """
        Lazy-loading (loads model and tokenizer if not already loaded)
        """
        if self.processor is None:
            self.processor = AutoProcessor.from_pretrained(
                self.model_id, cache_dir=self.cache_dir
            )

        if self.model is None:
            self.model = Gemma3ForConditionalGeneration.from_pretrained(
                self.model_id, device_map="auto"
            ).eval() #.eval comes from gemma's model card (https://huggingface.co/google/gemma-3-12b-it)

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

    def format_chat_for_gemma(self, chat: list[ChatMessage]) -> list[dict]:
        formatted_chat = []

        for msg in chat:
            formatted_chat.append({
                "role": msg.role,
                "content": [{"type": "text", "text": msg.content}]
            })

        return formatted_chat

    def generate(self, chat: list[ChatMessage], max_new_tokens: int = 3000):
        kwargs = self.format_params()

        if len(kwargs) > 0:
            do_sample = True
        else:
            do_sample = False
            print(
                "[INFO:] No sampling parameters nor penalty parameters were passed. Setting do_sample to 'False'"
            )

        formatted_chat = self.format_chat_for_gemma(chat)

        self.processor.use_default_system_prompt = False # ensure no system prompt is there
        
        text = self.processor.apply_chat_template(
            formatted_chat,
            tokenize=False,
            add_generation_prompt=True,
        )

        # tokenized inputs and outputs
        model_inputs = self.tokenizer(
            text, return_tensors="pt"
        ).to(self.model.device)

        input_len = model_inputs["input_ids"].shape[-1]
        
        output = self.model.generate(
            **model_inputs,

            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            **kwargs,
        )

        # chat (decoded output)
        response = self.processor.decode(output[0][input_len:], skip_special_tokens=True)

        chat_message = ChatMessage(role="assistant", content=response)

        return chat_message
