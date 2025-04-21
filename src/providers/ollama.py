from ollama import chat

from .base import LLMProvider


class OllamaProvider(LLMProvider):
    def __init__(
        self,
        model_name="ollama run llama2-uncensored:7b",
        temperature=0.3,
        max_tokens=4280,
    ):
        self.model = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

    def format_prompt(self, text: str) -> str:
        return text

    def generate(self, prompt: str, max_tokens: int = None) -> str:
        if max_tokens is None:
            max_tokens = self.max_tokens

        response = chat(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        )
        return response["message"]["content"].strip()

    def batch_generate(self, prompts: list, max_tokens: int = None) -> list:
        return [self.generate(p, max_tokens=max_tokens) for p in prompts]
