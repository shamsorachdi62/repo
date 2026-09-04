import os
import sys
import warnings
from pathlib import Path

try:
    import onnxruntime
    _TASHKEEL_AVAILABLE = True
except ImportError:
    _TASHKEEL_AVAILABLE = False

_DIACRITIZER_INST = None

def _get_diacritizer():
    global _DIACRITIZER_INST
    if _DIACRITIZER_INST is not None:
        return _DIACRITIZER_INST

    curr_dir = Path(__file__).resolve().parent
    diac_dir = curr_dir / "diacritizer"
    model_path = diac_dir / "diacritizer.onnx"
    vocab_path = diac_dir / "input_vocab_to_int.json"
    output_vocab_path = diac_dir / "output_int_to_vocab.json"

    if not model_path.exists() or not vocab_path.exists():
        os.makedirs(diac_dir, exist_ok=True)
        zip_path = diac_dir / "onnx_infer.zip"
        file_id = "1YWPDvrd_N4SawN8gV_XuVljrtlEfpGNn"
        try:
            import gdown
            gdown.download(id=file_id, output=str(zip_path), quiet=False)
        except Exception:
            import requests
            url = f"https://drive.google.com/uc?id={file_id}&export=download"
            session = requests.Session()
            res = session.get(url, stream=True)
            for k, v in res.cookies.items():
                if k.startswith("download_warning"):
                    url = f"{url}&confirm={v}"
                    res = session.get(url, stream=True)
                    break
            with open(zip_path, "wb") as f:
                for chunk in res.iter_content(chunk_size=32768):
                    if chunk:
                        f.write(chunk)

        import zipfile
        with zipfile.ZipFile(str(zip_path), "r") as z:
            z.extractall(str(diac_dir))
        if zip_path.exists():
            try:
                os.remove(str(zip_path))
            except Exception:
                pass

    from .diacritizer.infer_onnx import OnnxArabicDiacritizer
    _DIACRITIZER_INST = OnnxArabicDiacritizer(
        model_path=str(model_path),
        vocab_path=str(vocab_path),
        output_vocab_path=str(output_vocab_path),
    )
    return _DIACRITIZER_INST

def tashkeel(text: str) -> str:
    if not _TASHKEEL_AVAILABLE:
        warnings.warn("Warning: onnxruntime not available, skipping tashkeel.", UserWarning)
        return text
    try:
        inst = _get_diacritizer()
        return inst.diacritize(text)
    except Exception as e:
        warnings.warn(f"Warning: ONNX diacritizer failed with {e}, fallback to text", UserWarning)
        return text
