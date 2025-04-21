# providers/base.py
from abc import ABC, abstractmethod
from typing import List


class LLMProvider(ABC):
    @abstractmethod
    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        pass

    @abstractmethod
    def batch_generate(self, prompts: List[str], max_tokens: int = 512) -> List[str]:
        pass

    @abstractmethod
    def format_prompt(self, text: str) -> str:
        pass
