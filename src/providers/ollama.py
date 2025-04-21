from concurrent.futures import ThreadPoolExecutor, as_completed

from ollama import chat

from .base import LLMProvider


class OllamaProvider(LLMProvider):
    def __init__(
        self,
        model_name="llama2-uncensored:7b",
        temperature=0.3,
        max_tokens=4280,
        concurrent_batch_size=5,
    ):
        self.model = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.concurrent_batch_size = concurrent_batch_size

    def format_prompt(self, text: str) -> str:
        return text

    def generate(self, prompt: str, max_tokens: int = None) -> str:
        if max_tokens is None:
            max_tokens = self.max_tokens

        response = chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
        )
        return response["message"]["content"].strip()

    def batch_generate(self, prompts: list, max_tokens: int = None) -> list:
        results = [None] * len(prompts)

        def task(i, prompt):
            return i, self.generate(prompt, max_tokens=max_tokens)

        with ThreadPoolExecutor(max_workers=self.concurrent_batch_size) as executor:
            futures = [
                executor.submit(task, i, prompt) for i, prompt in enumerate(prompts)
            ]
            for future in as_completed(futures):
                i, result = future.result()
                results[i] = result

        return results
