from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)

from .base import LLMProvider


class HuggingFaceProvider(LLMProvider):
    def __init__(
        self,
        model_name="georgesung/llama3_8b_chat_uncensored",
        temperature=0.7,
        max_tokens=4288,
        use_8bit=True,
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

        quant_config = BitsAndBytesConfig(load_in_8bit=True) if use_8bit else None

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            quantization_config=quant_config,
        )
        self.pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_length=self.max_tokens,
            temperature=self.temperature,
            top_p=0.95,
            repetition_penalty=1.15,
        )

    def format_prompt(self, text: str) -> str:
        return f"### HUMAN:\n{text}\n\n### RESPONSE:\n"

    def generate(self, prompt: str, max_tokens: int = None) -> str:
        max_tokens = max_tokens or self.max_tokens
        output = self.pipeline(self.format_prompt(prompt), max_length=max_tokens)
        text = output[0]["generated_text"]
        return self._extract_response(text)

    def batch_generate(self, prompts: list, max_tokens: int = None) -> list:
        return [self.generate(p, max_tokens=max_tokens) for p in prompts]

    def _extract_response(self, text):
        # Extract text after "### RESPONSE:"
        split_key = "### RESPONSE:"
        if split_key in text:
            return text.split(split_key)[-1].strip()
        return text.strip()
