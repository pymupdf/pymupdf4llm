"""Token counting utilities for chunk size management."""

from typing import Callable, Optional

# Approximate tokens per character ratio for fallback estimation
_CHARS_PER_TOKEN_APPROX = 4.0


class TokenCounter:
    """Counts tokens using tiktoken or a custom function.

    Falls back to character-based estimation if tiktoken is unavailable.
    """

    def __init__(self, tokenizer: Optional[str | Callable] = None):
        self._count_fn = None

        if callable(tokenizer):
            self._count_fn = tokenizer
        elif isinstance(tokenizer, str):
            self._count_fn = self._make_tiktoken_counter(tokenizer)

        if self._count_fn is None:
            self._count_fn = self._char_estimate

    @staticmethod
    def _make_tiktoken_counter(encoding_name: str) -> Optional[Callable]:
        try:
            import tiktoken
        except ImportError:
            # tiktoken is optional; fall back to character estimation
            return None
        # An unknown encoding name is a caller error and must be loud —
        # a silent fallback would skew every budget decision.
        enc = tiktoken.get_encoding(encoding_name)
        return lambda text: len(enc.encode(text))

    @staticmethod
    def _char_estimate(text: str) -> int:
        return max(1, int(len(text) / _CHARS_PER_TOKEN_APPROX))

    def count(self, text: str) -> int:
        """Count tokens in text."""
        if not text:
            return 0
        return self._count_fn(text)
