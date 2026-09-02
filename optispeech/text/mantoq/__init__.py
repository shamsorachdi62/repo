import sys, os
import importlib.util
from pathlib import Path

_mantoq_dir = Path(__file__).resolve().parent
_lib_dir = _mantoq_dir / "lib"
_buck_dir = _lib_dir / "buck"
_pyarabic_dir = _lib_dir / "pyarabic"
_pylibtashkeel_dir = _lib_dir / "pylibtashkeel"

for _p in [_mantoq_dir, _lib_dir, _buck_dir, _pyarabic_dir, _pylibtashkeel_dir]:
    _ds = str(_p)
    if os.path.isdir(_ds) and _ds not in sys.path:
        sys.path.insert(0, _ds)

for _p in [_lib_dir, _buck_dir, _pyarabic_dir, _pylibtashkeel_dir]:
    if os.path.isdir(str(_p)):
        _init_file = _p / "__init__.py"
        if not _init_file.exists():
            try:
                _init_file.write_text("# __init__\n", encoding="utf-8")
            except Exception:
                pass


def _load_mod(mod_name, file_path):
    spec = importlib.util.spec_from_file_location(mod_name, str(file_path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


try:
    from .lib.buck import symbols
    from .lib.buck.tokenization import (arabic_to_phonemes, phon_to_id_,
                                        phonemes_to_tokens, simplify_phonemes)
    from .lib.buck.tokenization import tokens_to_ids as _tokens_to_id
except Exception:
    try:
        import symbols
        import tokenization
        arabic_to_phonemes = tokenization.arabic_to_phonemes
        phon_to_id_ = tokenization.phon_to_id_
        phonemes_to_tokens = tokenization.phonemes_to_tokens
        simplify_phonemes = tokenization.simplify_phonemes
        _tokens_to_id = tokenization.tokens_to_ids
    except Exception:
        symbols = _load_mod("symbols", _buck_dir / "symbols.py")
        phonetise = _load_mod("phonetise_buckwalter", _buck_dir / "phonetise_buckwalter.py")
        tokenization = _load_mod("tokenization", _buck_dir / "tokenization.py")
        arabic_to_phonemes = tokenization.arabic_to_phonemes
        phon_to_id_ = tokenization.phon_to_id_
        phonemes_to_tokens = tokenization.phonemes_to_tokens
        simplify_phonemes = tokenization.simplify_phonemes
        _tokens_to_id = tokenization.tokens_to_ids

from .num2words import num2words
from .tashkeel import tashkeel

MANTOQ_SYMBOLS = dict(phon_to_id_)
MANTOQ_SPECIAL_SYMBOLS = dict(
    pad=MANTOQ_SYMBOLS[symbols.PADDING_TOKEN],
    eos=MANTOQ_SYMBOLS[symbols.EOS_TOKEN],
)
AR_SPECIAL_PUNCS_TABLE = str.maketrans("،؟؛", ",?;")


def normalize_input_text(
    text: str,
    add_tashkeel: bool = True,
    process_numbers: bool = True,
) -> str:
    text = text.translate(AR_SPECIAL_PUNCS_TABLE)
    if add_tashkeel:
        text = tashkeel(text)
    if process_numbers:
        text = num2words(text)
    return text


def g2p(
    text: str,
    add_tashkeel: bool = True,
    process_numbers: bool = True,
    append_eos: bool = False,
) -> list[str]:
    normalized_text = normalize_input_text(text, add_tashkeel, process_numbers)
    text = normalized_text
    phones = arabic_to_phonemes(text)
    phones = simplify_phonemes(phones)
    tokens = phonemes_to_tokens(phones)
    if not append_eos:
        tokens = tokens[:-1]
    return normalized_text, tokens


def tokens2ids(tokens: list[str]) -> list[int]:
    return _tokens_to_id(tokens)
