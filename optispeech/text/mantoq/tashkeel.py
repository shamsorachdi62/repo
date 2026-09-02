import sys
import warnings
import importlib.util
from pathlib import Path

_mantoq_dir = Path(__file__).resolve().parent
_pylibtashkeel_dir = _mantoq_dir / "lib" / "pylibtashkeel"


def _load_mod(mod_name, file_path):
    spec = importlib.util.spec_from_file_location(mod_name, str(file_path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


try:
    from .lib.pylibtashkeel import LibtashkeelDiacritizer
except Exception:
    if str(_pylibtashkeel_dir) not in sys.path:
        sys.path.insert(0, str(_pylibtashkeel_dir))
    try:
        from text_encoder import LibtashkeelDiacritizer
    except Exception:
        te = _load_mod("text_encoder", _pylibtashkeel_dir / "text_encoder.py")
        LibtashkeelDiacritizer = te.LibtashkeelDiacritizer

try:
    import onnxruntime

    _TASHKEEL_AVAILABLE = True
except ImportError:
    _TASHKEEL_AVAILABLE = False

_DIACRITIZER_INST = None


def tashkeel(text: str) -> str:
    global _DIACRITIZER_INST
    if not _TASHKEEL_AVAILABLE:
        warnings.warn(
            "Warning: The Tashkeel feature will not be available. Please re-install with the `libtashkeel` extra.",
            UserWarning,
        )
        return text
    if _DIACRITIZER_INST is None:
        _DIACRITIZER_INST = LibtashkeelDiacritizer()
    return _DIACRITIZER_INST([text])[0]
