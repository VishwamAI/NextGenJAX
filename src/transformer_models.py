import os
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch


class TransformerModel:
    def __init__(self, model_name: str, device: str = "cpu"):
        self.model_name = model_name
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(
            self.device
        )

    def save_model(self, save_directory: str):
        os.makedirs(save_directory, exist_ok=True)
        self.tokenizer.save_pretrained(os.path.join(save_directory, "tokenizer"))
        self.model.save_pretrained(os.path.join(save_directory, "model"))

    def load_model(self, load_directory: str):
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                os.path.join(load_directory, "tokenizer")
            )
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                os.path.join(load_directory, "model")
            ).to(self.device)
        except Exception as e:
            print(f"Error loading model or tokenizer: {e}")

    def generate_text(self, input_text: str, max_length: int = 50) -> str:
        if not isinstance(input_text, str):
            raise ValueError("input_text must be a string")
        if not isinstance(max_length, int) or max_length <= 0:
            raise ValueError("max_length must be a positive integer")

        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            inputs["input_ids"], max_length=max_length
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
