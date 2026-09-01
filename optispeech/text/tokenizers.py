import re
from abc import ABC, abstractmethod

from . import mantoq
from .mantoq import MANTOQ_SYMBOLS, MANTOQ_SPECIAL_SYMBOLS


# tokenizer registry
_TOKENIZERS = {}


class BaseTokenizer(ABC):
    name: str
    input_symbols: dict[str, int]
    special_symbols: dict[str, int]

    def __init_subclass__(cls, /, **kwargs):
        _TOKENIZERS.setdefault(cls.name, cls)

    @classmethod
    def get_tokenizer_by_name(cls, name):
        try:
            return _TOKENIZERS[name]
        except KeyError:
            raise ValueError(f"Tokenizer `{name}` does not exist.")

    def __init__(
        self,
        add_blank: bool,
        add_bos_eos: bool,
        normalize_text: bool,
    ):
        self.add_blank = add_blank
        self.add_bos_eos = add_bos_eos
        self.normalize_text = normalize_text

    @abstractmethod
    def __call__(
        self, text: str, language: str, *, split_sentences: bool = True
    ) -> tuple[list[int] | list[list[int]], str]:
        """Return input IDs."""

    def preprocess_text(self, text: str, language: str = None) -> str:
        return text



class MantoqTokenizer(BaseTokenizer):
    name = "mantoq"
    input_symbols = MANTOQ_SYMBOLS
    special_symbols = MANTOQ_SPECIAL_SYMBOLS

    def __call__(
        self, text: str, language: str, *, split_sentences: bool = True
    ) -> tuple[list[int] | list[list[int]], str]:
        assert language == "ar", "MantoqTokenizer only supports Arabic language"
        sequence = []
        clean_text, natural_text = self.phonemize_text(text)
        for symbol in clean_text:
            symbol_id = self.input_symbols[symbol]
            sequence += [symbol_id]
        if split_sentences:
            sequence = [sequence]
        return sequence, clean_text

    def phonemize_text(self, text: str, language: str=None) -> str:
        texts, phonemes = mantoq.g2p(text, add_tashkeel=False)
        return phonemes, texts

    def get_pause_separator_stretch_mask(self, text: str) -> list[bool]:
        """Return whether each word separator may be stretched."""
        text = mantoq.normalize_input_text(text, add_tashkeel=False)
        standalone_punctuation = {".", ",", "?", "!"}
        words = [word for word in text.split() if word not in standalone_punctuation]
        return [not word.startswith(("ا", "ٱ")) for word in words[1:]]
