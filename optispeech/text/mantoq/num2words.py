import re
import sys
import importlib.util
from pathlib import Path
from functools import partial

_mantoq_dir = Path(__file__).resolve().parent
_pyarabic_dir = _mantoq_dir / "lib" / "pyarabic"


def _load_mod(mod_name, file_path):
    spec = importlib.util.spec_from_file_location(mod_name, str(file_path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


try:
    from .lib.pyarabic import araby
    from .lib.pyarabic import number as arnum
    from .lib.pyarabic.trans import normalize_digits
except Exception:
    for p in [_pyarabic_dir, _mantoq_dir / "lib"]:
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    try:
        import araby
        import number as arnum
        from trans import normalize_digits
    except Exception:
        araby = _load_mod("araby", _pyarabic_dir / "araby.py")
        arnum = _load_mod("arnum", _pyarabic_dir / "number.py")
        trans = _load_mod("trans", _pyarabic_dir / "trans.py")
        normalize_digits = trans.normalize_digits

NUM_REGEX = re.compile(r"\d+")
PERCENT_NO_DIAC = "بالمئة"
PERCENT_DIAC = "بِالْمِئَة"


def _convert_num2words(m: re.Match, *, apply_tashkeel):
    number = m.group(0)
    word_representation = arnum.number2text(number)
    if apply_tashkeel:
        return " ".join(arnum.pre_tashkeel_number(word_representation.split(" ")))
    return word_representation


def num2words(text: str, handle_percent=True, apply_tashkeel: bool = True) -> str:
    """
    Converts numbers in `text` to Arabic words.
    Simple conversion. Does not check if the number is date/currency...etc.

    Args:
        text: input text that may contain numbers
        apply_tashkeel: diacritize added words
    """
    text = normalize_digits(text)
    output = NUM_REGEX.sub(
        partial(_convert_num2words, apply_tashkeel=apply_tashkeel), text
    )
    if handle_percent:
        replacement = PERCENT_DIAC if apply_tashkeel else PERCENT_NO_DIAC
        output = output.replace("%", f" {replacement}")
    return araby.fix_spaces(output)
