# providers/unsloth_provider.py
from unsloth import FastLanguageModel
from vllm import SamplingParams

from .base import LLMProvider


class UnslothProvider(LLMProvider):
    def __init__(self, model_name="meta-llama/meta-Llama-3.1-8B-Instruct"):
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=4096,
            load_in_4bit=True,
            fast_inference=True,
            gpu_memory_utilization=0.6,
        )
        self.default_sampling = SamplingParams(
            temperature=0.3,
            top_p=0.95,
            max_tokens=512,
        )

    def format_prompt(self, text: str) -> str:
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=False,
            add_generation_prompt=True,
        )

    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        params = self.default_sampling
        params.max_tokens = max_tokens
        formatted = self.format_prompt(prompt)
        return (
            self.model.fast_generate([formatted], sampling_params=params)[0]
            .outputs[0]
            .text
        )

    def batch_generate(self, prompts: list, max_tokens: int = 512) -> list:
        params = self.default_sampling
        params.max_tokens = max_tokens
        formatted = [self.format_prompt(p) for p in prompts]
        return [
            output.outputs[0].text
            for output in self.model.fast_generate(formatted, sampling_params=params)
        ]
