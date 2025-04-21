# providers/openai_provider.py
import openai

from .base import LLMProvider


class OpenAIProvider(LLMProvider):
    def __init__(self, model="gpt-4", temperature=0.3, max_tokens=512):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def format_prompt(self, text: str) -> str:
        return text

    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        response = openai.ChatCompletion.create(
            model=self.model,
            temperature=self.temperature,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content.strip()

    def batch_generate(self, prompts: list, max_tokens: int = 512) -> list:
        return [self.generate(p, max_tokens=max_tokens) for p in prompts]
