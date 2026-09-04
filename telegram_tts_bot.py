import os
import sys
import subprocess
import shutil
import glob
import time
import re
import json
import zipfile
import urllib.request
from pathlib import Path

if os.path.exists("/kaggle/working"):
    WORKSPACE_DIR = "/kaggle/working"
elif os.path.exists("/content"):
    WORKSPACE_DIR = "/content"
else:
    WORKSPACE_DIR = os.getcwd()

LOCAL_REPO = os.path.join(WORKSPACE_DIR, "repo")
LOCAL_CHECKPOINTS = os.path.join(WORKSPACE_DIR, "checkpoints")
LOCAL_ONNX_EXPORT = os.path.join(WORKSPACE_DIR, "onnx_exports")
LOCAL_AUDIO_OUT = os.path.join(WORKSPACE_DIR, "generated_audio")

for d in [LOCAL_CHECKPOINTS, LOCAL_ONNX_EXPORT, LOCAL_AUDIO_OUT]:
    os.makedirs(d, exist_ok=True)

if os.path.isdir(LOCAL_REPO) and LOCAL_REPO not in sys.path:
    sys.path.insert(0, LOCAL_REPO)

HF_BACKUP_REPO = "Mohamad-I8/tts-training-backup"
HF_TOKEN = "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"
HF_CHECKPOINTS_PREFIX = "checkpoints"
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "8616385226:AAHgVZtivsJHWC6HypQUVS5OxqGcgbgQZVc")

def ensure_dependencies():
    packages = []
    try:
        import pyworld
    except ImportError:
        packages.extend(["pyworld", "librosa"])
    try:
        import telebot
    except ImportError:
        packages.append("pyTelegramBotAPI")
    try:
        import pyloudnorm
    except ImportError:
        packages.append("pyloudnorm")
    try:
        from huggingface_hub import HfApi
    except ImportError:
        packages.append("huggingface_hub")
    try:
        import onnxruntime
    except ImportError:
        packages.append("onnxruntime")
    try:
        import gdown
    except ImportError:
        packages.append("gdown")

    if packages:
        print(f"Installing missing dependencies: {packages}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q"] + packages)

ensure_dependencies()

from huggingface_hub import HfApi, hf_hub_download
import telebot
from telebot.types import Message
import numpy as np
import soundfile as sf
import torch
import pyloudnorm as pyln

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
hf_api = HfApi(token=HF_TOKEN)

def apply_compatibility_patches():
    nano_file = os.path.join(LOCAL_REPO, "optispeech", "onnx", "nano_streaming.py")
    if os.path.exists(nano_file):
        with open(nano_file, "r", encoding="utf-8") as f:
            n_code = f.read()
        if "_torch_load_compat" not in n_code:
            patch_str = (
                "import torch\n"
                "_orig_torch_load = torch.load\n"
                "def _torch_load_compat(*args, **kwargs):\n"
                "    kwargs['weights_only'] = False\n"
                "    return _orig_torch_load(*args, **kwargs)\n"
                "torch.load = _torch_load_compat\n"
                "if hasattr(torch.serialization, 'add_safe_globals'):\n"
                "    try:\n"
                "        import hydra._internal.target_policy\n"
                "        torch.serialization.add_safe_globals([hydra._internal.target_policy._DeferredTarget])\n"
                "    except Exception:\n"
                "        pass\n\n"
            )
            with open(nano_file, "w", encoding="utf-8") as f:
                f.write(patch_str + n_code)

    modules_init = os.path.join(LOCAL_REPO, "optispeech", "model", "generator", "modules", "__init__.py")
    if os.path.exists(os.path.dirname(modules_init)):
        with open(modules_init, "w", encoding="utf-8") as f:
            f.write("from .core import *\nfrom .layers import *\nfrom .nano_layers import *\nfrom .pqmf import *\n")

    datamodule_file = os.path.join(LOCAL_REPO, "optispeech", "dataset", "text_wav_datamodule.py")
    if os.path.exists(datamodule_file):
        with open(datamodule_file, "r", encoding="utf-8") as f:
            dm_code = f.read()
        if "import pyworld as pw" in dm_code and "except ImportError:" not in dm_code:
            dm_code = dm_code.replace("import pyworld as pw", "try:\n    import pyworld as pw\nexcept ImportError:\n    pw = None")
            with open(datamodule_file, "w", encoding="utf-8") as f:
                f.write(dm_code)

    pitch_file = os.path.join(LOCAL_REPO, "optispeech", "dataset", "feature_extractors", "pitch_extractors.py")
    if os.path.exists(pitch_file):
        with open(pitch_file, "r", encoding="utf-8") as f:
            p_code = f.read()
        if "import pyworld as pw" in p_code and "except ImportError:" not in p_code:
            p_code = p_code.replace(
                "import torchcrepe\nimport penn\nimport pyworld as pw",
                "try:\n    import torchcrepe\nexcept ImportError:\n    torchcrepe = None\ntry:\n    import penn\nexcept ImportError:\n    penn = None\ntry:\n    import pyworld as pw\nexcept ImportError:\n    pw = None"
            )
            with open(pitch_file, "w", encoding="utf-8") as f:
                f.write(p_code)

def ensure_mantoq_library():
    mantoq_dir = os.path.join(LOCAL_REPO, "optispeech", "text", "mantoq")
    mantoq_lib = os.path.join(mantoq_dir, "lib")
    buck_dir = os.path.join(mantoq_lib, "buck")
    if not os.path.isdir(buck_dir) or not os.path.exists(os.path.join(buck_dir, "symbols.py")):
        print("Restoring mantoq linguistic phonemizer library from Hugging Face...")
        os.makedirs(mantoq_lib, exist_ok=True)
        z_path = hf_hub_download(
            repo_id=HF_BACKUP_REPO,
            filename="assets/mantoq_lib.zip",
            repo_type="model",
            token=HF_TOKEN
        )
        with zipfile.ZipFile(z_path, "r") as zf:
            zf.extractall(mantoq_lib)
        print("Mantoq linguistic library restored successfully.")

def ensure_diacritizer_model():
    diac_dir = os.path.join(LOCAL_REPO, "optispeech", "text", "mantoq", "diacritizer")
    model_file = os.path.join(diac_dir, "diacritizer.onnx")
    vocab_file = os.path.join(diac_dir, "input_vocab_to_int.json")
    if not os.path.exists(model_file) or not os.path.exists(vocab_file):
        print("Downloading ONNX Arabic Diacritizer from Google Drive...")
        os.makedirs(diac_dir, exist_ok=True)
        zip_path = os.path.join(diac_dir, "onnx_infer.zip")
        file_id = "1YWPDvrd_N4SawN8gV_XuVljrtlEfpGNn"
        try:
            import gdown
            gdown.download(id=file_id, output=zip_path, quiet=False)
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
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(diac_dir)
        if os.path.exists(zip_path):
            try:
                os.remove(zip_path)
            except Exception:
                pass
        print("ONNX Arabic Diacritizer downloaded and extracted successfully.")

apply_compatibility_patches()
ensure_mantoq_library()
ensure_diacritizer_model()

from optispeech.model import OptiSpeech

def pick_latest_checkpoint(file_list):
    if not file_list:
        return None
    def sort_key(f):
        fname = os.path.basename(f)
        m_ep = re.search(r"epoch[=_]?(\d+)", fname, re.IGNORECASE)
        m_st = re.search(r"step[=_]?(\d+)", fname, re.IGNORECASE)
        ep = int(m_ep.group(1)) if m_ep else 0
        st = int(m_st.group(1)) if m_st else 0
        return (st, ep)
    return max(file_list, key=sort_key)

def sync_latest_checkpoint():
    existing_locals = [
        f for f in glob.glob(os.path.join(LOCAL_CHECKPOINTS, "**", "*.ckpt"), recursive=True)
        if os.path.basename(f) != "resume_target.ckpt"
    ]
    latest_local = pick_latest_checkpoint(existing_locals)

    try:
        repo_files = [f.rfilename for f in hf_api.list_repo_tree(repo_id=HF_BACKUP_REPO, repo_type="model", recursive=True) if hasattr(f, "rfilename")]
    except Exception as e:
        print(f"Warning: Failed to connect to {HF_BACKUP_REPO}: {e}")
        repo_files = []

    ckpt_files = [f for f in repo_files if f.startswith(HF_CHECKPOINTS_PREFIX) and f.endswith(".ckpt")]

    target_repo = HF_BACKUP_REPO
    if not ckpt_files:
        alt_repo = "Mohamad-I8/arabic-tts-chechpoints"
        try:
            alt_files = [f.rfilename for f in hf_api.list_repo_tree(repo_id=alt_repo, repo_type="model", recursive=True) if hasattr(f, "rfilename")]
            ckpt_files = [f for f in alt_files if f.endswith(".ckpt")]
            if ckpt_files:
                target_repo = alt_repo
        except Exception:
            pass

    if not ckpt_files:
        if latest_local:
            print(f"No remote checkpoints available. Using local checkpoint: {latest_local}")
            return latest_local, False
        raise FileNotFoundError("No checkpoint (.ckpt) files found on Hugging Face or locally.")

    target_hf_file = pick_latest_checkpoint(ckpt_files)
    target_hf_basename = os.path.basename(target_hf_file)
    expected_local_path = os.path.join(LOCAL_CHECKPOINTS, target_hf_basename)

    is_new = False
    if latest_local is None:
        is_new = True
        print(f"Preparing to download latest remote checkpoint: {target_hf_basename}")
    else:
        local_basename = os.path.basename(latest_local)
        if local_basename != target_hf_basename:
            comp = pick_latest_checkpoint([latest_local, target_hf_file])
            if comp == target_hf_file:
                is_new = True
                print(f"Newer checkpoint detected on Hugging Face: {target_hf_basename} (Local: {local_basename})")
            else:
                return latest_local, False
        else:
            return latest_local, False

    if is_new:
        print(f"Downloading newest checkpoint: {target_hf_file} from {target_repo}...")
        local_download = hf_hub_download(
            repo_id=target_repo,
            filename=target_hf_file,
            repo_type="model",
            local_dir=WORKSPACE_DIR,
            token=HF_TOKEN,
        )

        if os.path.abspath(local_download) != os.path.abspath(expected_local_path):
            shutil.copy2(local_download, expected_local_path)
            try:
                os.remove(local_download)
            except Exception:
                pass

        for old_c in existing_locals:
            if os.path.abspath(old_c) != os.path.abspath(expected_local_path):
                try:
                    os.remove(old_c)
                    print(f"Deleted obsolete local checkpoint: {os.path.basename(old_c)}")
                except Exception as e:
                    print(f"Could not delete old checkpoint {old_c}: {e}")

    size_mb = os.path.getsize(expected_local_path) / (1024 * 1024)
    print(f"Active Checkpoint is: {expected_local_path} ({size_mb:.2f} MB)")
    return expected_local_path, is_new

def export_to_onnx(checkpoint_path=None, output_dir=LOCAL_ONNX_EXPORT, quantize=False):
    ckpt = checkpoint_path or ACTIVE_CHECKPOINT_PATH
    test_sentence = "مرحبا، هذا اختبار سريع لقياس أداء النموذج الصوتي."
    os.makedirs(output_dir, exist_ok=True)
    for old_onnx in glob.glob(os.path.join(output_dir, "*.onnx")):
        try:
            os.remove(old_onnx)
        except Exception:
            pass

    cmd = [
        sys.executable, "-m", "optispeech.onnx.nano_streaming",
        "--checkpoint", ckpt,
        "--output-dir", output_dir,
        "--text", test_sentence
    ]
    if quantize:
        cmd.append("--quantize-encoder-decoder")

    res = subprocess.run(cmd, cwd=LOCAL_REPO)
    if res.returncode == 0:
        files = list(Path(output_dir).glob("*.onnx"))
        print(f"Exported {len(files)} ONNX files to {output_dir}.")
        return True
    return False

class ArabicTTSEngine:
    def __init__(self, checkpoint_path, device=DEVICE):
        self.device = torch.device(device)
        print(f"Loading OptiSpeech from {checkpoint_path} onto {self.device}...")
        self.model = OptiSpeech.load_from_checkpoint(checkpoint_path, map_location="cpu")
        self.model.to(self.device)
        self.model.eval()
        self.sample_rate = self.model.sample_rate
        self.meter = pyln.Meter(self.sample_rate)
        print(f"OptiSpeech Engine loaded successfully. Sample Rate: {self.sample_rate} Hz")

    def synthesize(self, text: str, d_factor: float = 1.0, p_factor: float = 1.0, output_path: str = None) -> dict:
        t0 = time.time()
        inputs = self.model.prepare_input(text, d_factor=d_factor, p_factor=p_factor, split_sentences=True)
        with torch.inference_mode():
            outputs = self.model.synthesise(inputs)

        wav_parts = []
        for w in outputs.unbatched_wavs():
            w_np = w.squeeze().float().detach().cpu().numpy()
            wav_parts.append(w_np)
            wav_parts.append(np.zeros(int(self.sample_rate * 0.15), dtype=np.float32))

        full_wav = np.concatenate(wav_parts) if len(wav_parts) > 1 else wav_parts[0]

        try:
            loudness = self.meter.integrated_loudness(full_wav)
            full_wav = pyln.normalize.loudness(full_wav, loudness, -20.0)
        except Exception:
            max_val = np.max(np.abs(full_wav))
            if max_val > 0:
                full_wav = full_wav / max_val * 0.95

        duration_sec = len(full_wav) / self.sample_rate
        latency_sec = time.time() - t0
        rtf = latency_sec / duration_sec if duration_sec > 0 else 0

        if output_path is None:
            output_path = os.path.join(LOCAL_AUDIO_OUT, f"speech_{int(time.time() * 1000)}.wav")

        sf.write(output_path, full_wav, self.sample_rate, subtype="PCM_16")

        return {
            "wav_path": output_path,
            "duration": duration_sec,
            "latency": latency_sec,
            "rtf": rtf,
            "clean_text": inputs.clean_text
        }

ACTIVE_CHECKPOINT_PATH, _ = sync_latest_checkpoint()
tts = ArabicTTSEngine(ACTIVE_CHECKPOINT_PATH)

bot = telebot.TeleBot(TELEGRAM_BOT_TOKEN, parse_mode="Markdown")
user_settings = {}

def get_user_settings(user_id):
    if user_id not in user_settings:
        user_settings[user_id] = {"speed": 1.0, "pitch": 1.0}
    return user_settings[user_id]

@bot.message_handler(commands=['start', 'help'])
def send_welcome(message: Message):
    uid = message.from_user.id
    settings = get_user_settings(uid)
    welcome_text = (
        "*أهلاً بك في بوت توليد الصوت العربي (OptiSpeech TTS)*\n\n"
        "أرسل لي أي نص عربي (مشكول أو غير مشكول)، وسأقوم بتحويله إلى رسالة صوتية بجودة عالية فوراً.\n\n"
        "*الأوامر المتاحة:*\n"
        f"• `/speed <قيمة>`: ضبط سرعة الكلام (الحالية: `{settings['speed']}`)\n"
        f"• `/pitch <قيمة>`: ضبط نبرة الصوت (الحالية: `{settings['pitch']}`)\n"
        "• `/update`: فحص وتحديث النموذج تلقائياً إذا توفرت نقطة فحص أحدث من التدريب\n"
        "• `/onnx`: تصدير نقطة الفحص النشطة إلى صيغة ONNX\n"
        "• `/info`: معلومات عن نقطة الفحص والنموذج الحالي\n\n"
        "*جرب الآن:* أرسل أي جملة باللغة العربية."
    )
    bot.reply_to(message, welcome_text)

@bot.message_handler(commands=['speed'])
def change_speed(message: Message):
    uid = message.from_user.id
    parts = message.text.strip().split()
    if len(parts) < 2:
        bot.reply_to(message, "يرجى تحديد السرعة بين 0.5 و 2.0 (مثال: `/speed 1.1`)")
        return
    try:
        val = float(parts[1])
        if 0.5 <= val <= 2.0:
            get_user_settings(uid)["speed"] = val
            bot.reply_to(message, f"تم ضبط سرعة الكلام على: `{val}`")
        else:
            bot.reply_to(message, "يرجى اختيار قيمة بين 0.5 و 2.0")
    except ValueError:
        bot.reply_to(message, "قيمة غير صالحة. يرجى إدخال رقم عشري مثل `1.1`")

@bot.message_handler(commands=['pitch'])
def change_pitch(message: Message):
    uid = message.from_user.id
    parts = message.text.strip().split()
    if len(parts) < 2:
        bot.reply_to(message, "يرجى تحديد النبرة بين 0.5 و 2.0 (مثال: `/pitch 1.0`)")
        return
    try:
        val = float(parts[1])
        if 0.5 <= val <= 2.0:
            get_user_settings(uid)["pitch"] = val
            bot.reply_to(message, f"تم ضبط نبرة الصوت على: `{val}`")
        else:
            bot.reply_to(message, "يرجى اختيار قيمة بين 0.5 و 2.0")
    except ValueError:
        bot.reply_to(message, "قيمة غير صالحة. يرجى إدخال رقم عشري مثل `1.0`")

@bot.message_handler(commands=['update', 'sync'])
def handle_update_cmd(message: Message):
    global ACTIVE_CHECKPOINT_PATH, tts
    bot.send_chat_action(message.chat.id, 'typing')
    bot.reply_to(message, "جاري فحص Hugging Face للبحث عن نقطة فحص أحدث وحذف القديمة...")
    try:
        new_path, is_new = sync_latest_checkpoint()
        if is_new:
            ACTIVE_CHECKPOINT_PATH = new_path
            tts = ArabicTTSEngine(new_path)
            bot.reply_to(message, f"تم العثور على نقطة فحص جديدة وتحديث النموذج بنجاح:\n`{os.path.basename(new_path)}`")
        else:
            bot.reply_to(message, f"النموذج يعمل بالفعل على أحدث نقطة فحص متوفرة:\n`{os.path.basename(new_path)}`")
    except Exception as e:
        bot.reply_to(message, f"حدث خطأ أثناء التحديث: `{e}`")

@bot.message_handler(commands=['onnx'])
def handle_onnx_cmd(message: Message):
    bot.send_chat_action(message.chat.id, 'typing')
    bot.reply_to(message, "جاري تصدير نقطة الفحص النشطة إلى صيغة ONNX...")
    try:
        success = export_to_onnx(ACTIVE_CHECKPOINT_PATH)
        if success:
            bot.reply_to(message, f"تم تصدير ملفات ONNX بنجاح في مجلد:\n`{LOCAL_ONNX_EXPORT}`")
        else:
            bot.reply_to(message, "فشل تصدير ONNX، يرجى مراجعة سجلات التشغيل.")
    except Exception as e:
        bot.reply_to(message, f"حدث خطأ أثناء تصدير ONNX: `{e}`")

@bot.message_handler(commands=['info'])
def send_info(message: Message):
    ckpt_name = os.path.basename(ACTIVE_CHECKPOINT_PATH)
    info_text = (
        "*معلومات النموذج الصوتي:*\n\n"
        f"• *نقطة الفحص:* `{ckpt_name}`\n"
        f"• *معدل العينات:* `{tts.sample_rate} Hz`\n"
        f"• *المعالج اللغوي:* `Mantoq + Libtashkeel`\n"
        f"• *الجهاز:* `{DEVICE.upper()}`\n"
    )
    bot.reply_to(message, info_text)

@bot.message_handler(func=lambda msg: True, content_types=['text'])
def handle_text_to_speech(message: Message):
    uid = message.from_user.id
    text = message.text.strip()
    if not text:
        return

    settings = get_user_settings(uid)
    bot.send_chat_action(message.chat.id, 'record_voice')

    try:
        audio_file = os.path.join(LOCAL_AUDIO_OUT, f"voice_{uid}_{int(time.time() * 1000)}.wav")
        res = tts.synthesize(
            text=text,
            d_factor=settings["speed"],
            p_factor=settings["pitch"],
            output_path=audio_file
        )

        caption = (
            f"زمن المعالجة: `{res['latency']:.2f}s` | "
            f"المدة: `{res['duration']:.2f}s` | "
            f"RTF: `{res['rtf']:.3f}`"
        )

        with open(res["wav_path"], "rb") as voice_data:
            bot.send_voice(
                chat_id=message.chat.id,
                voice=voice_data,
                caption=caption,
                reply_to_message_id=message.message_id
            )

    except Exception as e:
        bot.reply_to(message, f"حدث خطأ أثناء توليد الصوت: `{e}`")

if __name__ == "__main__":
    print(f"Starting Telegram Bot for token: {TELEGRAM_BOT_TOKEN[:10]}... (Press Ctrl+C to terminate)")
    bot.infinity_polling(timeout=60, long_polling_timeout=60)
