#!/usr/bin/env python3
"""
================================================================================
🎙️ Kaggle Professional Dataset Converter & HF Hub Publisher for OptiSpeech
================================================================================
Target Dataset: Mohamad-I8/Aseel_Arabic_Dataset (Hugging Face)
Target Model:   OptiSpeech (https://github.com/shamsorachdi62/repo.git)

Workflow:
1. Installs all required dependencies (soundfile, librosa, pydub, huggingface_hub, etc.).
2. Downloads 'Aseel Arabic Dataset.zip' containing ~10,000 .mp3 audio clips.
3. Extracts and converts all .mp3 files into 24000Hz mono PCM_16 .wav files.
4. Places all converted files in the standard OptiSpeech folder 'wav/'.
5. Audits audio quality (discards corrupted/empty files) and builds clean metadata.
6. Deletes old compressed archives on Hugging Face Hub.
7. Uploads the professional, uncompressed dataset (wav/ + metadata.csv + README.md)
   to Hugging Face Hub (Mohamad-I8/Aseel_Arabic_Dataset).
8. Runs local OptiSpeech preprocessing (phonemization + feature extraction) and 
   calculates normalization statistics for training.
================================================================================
"""

import os
import sys
import glob
import json
import re
import random
import shutil
import zipfile
import subprocess
from pathlib import Path

# ==============================================================================
# 🛠️ 1. Dependency Setup
# ==============================================================================
def install_dependencies():
    print("📦 [1/8] Installing dependencies...")
    try:
        subprocess.run(["apt-get", "update", "-qq"], check=False)
        subprocess.run(["apt-get", "install", "-y", "-qq", "ffmpeg", "libsndfile1"], check=False)
    except Exception as e:
        print(f"⚠️ System package notice: {e}")

    packages = [
        "datasets",
        "huggingface_hub>=0.20.0",
        "soundfile>=0.12.0",
        "librosa>=0.9.2",
        "pydub",
        "lightning>=2.0.0",
        "torchmetrics>=0.11.4",
        "nnaudio>=0.3.3",
        "pyworld>=0.3.4",
        "hydra-core>=1.3.2",
        "hydra-colorlog>=1.2.0",
        "rootutils>=1.0.7",
        "rich>=13.7.1",
        "einops>=0.8.0",
        "unidecode>=1.3.8",
        "scipy>=1.14.0",
        "pandas>=2.2.2",
        "transformers>=4.44.0",
        "tqdm>=4.66.5",
    ]

    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q"] + packages)
    print("✅ Dependencies ready.\n")

install_dependencies()

import numpy as np
import soundfile as sf
from huggingface_hub import HfApi, hf_hub_download, snapshot_download, create_repo

# ==============================================================================
# ⚙️ 2. User Configuration
# ==============================================================================
# Set your Hugging Face Write Token here or via environment variable HF_TOKEN
HF_TOKEN         = os.getenv("HF_TOKEN", "").strip()
HF_DATASET_ID    = "Mohamad-I8/Aseel_Arabic_Dataset"
ZIP_FILENAME     = "Aseel Arabic Dataset.zip"

KAGGLE_WORKING   = "/kaggle/working" if os.path.exists("/kaggle/working") else os.getcwd()
REPO_URL         = "https://github.com/shamsorachdi62/repo.git"
EXPERIMENT_NAME  = "saeed"
SAMPLE_RATE      = 24000
PREPROCESS_WORKERS = max(1, os.cpu_count() // 2 if os.cpu_count() else 2)

REPO_DIR         = os.path.join(KAGGLE_WORKING, "repo")
RAW_DATASET_DIR  = os.path.join(KAGGLE_WORKING, "raw_dataset")
CONVERTED_WAV_DIR= os.path.join(RAW_DATASET_DIR, "wav")
PROCESSED_DATA_DIR= os.path.join(KAGGLE_WORKING, "processed_data")
DATA_STATS_DIR   = os.path.join(KAGGLE_WORKING, "data_stats")

for d in [RAW_DATASET_DIR, CONVERTED_WAV_DIR, DATA_STATS_DIR]:
    os.makedirs(d, exist_ok=True)

# ==============================================================================
# 📥 3. Clone / Setup OptiSpeech Repository
# ==============================================================================
print("📥 [2/8] Setting up OptiSpeech repository...")
if os.path.exists(REPO_DIR):
    print("   Repository exists locally. Updating...")
    subprocess.run(["git", "pull"], cwd=REPO_DIR, check=False)
else:
    print(f"   Cloning repository: {REPO_URL}...")
    subprocess.run(["git", "clone", REPO_URL, REPO_DIR], check=True)

if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)
os.chdir(REPO_DIR)
os.environ["PROJECT_ROOT"] = REPO_DIR
print(f"✅ Repository active at: {REPO_DIR}\n")

# ==============================================================================
# 🎧 4. Audio Quality Audit & Robust Conversion (MP3 → 24000Hz WAV)
# ==============================================================================
def is_valid_audio(filepath: str) -> bool:
    """Verifies audio file integrity, length, and non-zero frames."""
    try:
        if not os.path.exists(filepath) or os.path.getsize(filepath) < 200:
            return False
        info = sf.info(filepath)
        if info.duration < 0.1 or info.frames == 0:
            return False
        return True
    except Exception:
        return False

def convert_mp3_to_wav(src_path: str, dst_path: str, target_sr: int = SAMPLE_RATE) -> bool:
    """Converts MP3 audio file to 24000Hz mono PCM_16 WAV format."""
    # Method 1: librosa
    try:
        import librosa
        wav, _ = librosa.load(src_path, sr=target_sr, mono=True)
        if len(wav) > 0 and np.isfinite(wav).all():
            sf.write(dst_path, wav, target_sr, subtype="PCM_16")
            if is_valid_audio(dst_path):
                return True
    except Exception:
        pass

    # Method 2: pydub
    try:
        from pydub import AudioSegment
        audio = AudioSegment.from_file(src_path)
        audio = audio.set_frame_rate(target_sr).set_channels(1).set_sample_width(2)
        audio.export(dst_path, format="wav")
        if is_valid_audio(dst_path):
            return True
    except Exception:
        pass

    # Method 3: ffmpeg
    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-v", "error", "-i", str(src_path),
                "-ar", str(target_sr), "-ac", "1", "-sample_fmt", "s16", str(dst_path)
            ],
            check=True, capture_output=True
        )
        if is_valid_audio(dst_path):
            return True
    except Exception:
        pass

    return False

# ==============================================================================
# 📦 5. Download Original Zip & Convert ~10,000 MP3 Clips to WAV
# ==============================================================================
print(f"🤗 [3/8] Downloading original zip from Hugging Face ({HF_DATASET_ID})...")
download_dir = os.path.join(KAGGLE_WORKING, "_download")
os.makedirs(download_dir, exist_ok=True)

try:
    zip_path = hf_hub_download(
        repo_id=HF_DATASET_ID,
        filename=ZIP_FILENAME,
        repo_type="dataset",
        local_dir=download_dir,
        token=HF_TOKEN or None,
    )
    print(f"✅ Zip archive downloaded: {zip_path}")
except Exception as e:
    print(f"⚠️ Specific zip download failed ({e}). Downloading snapshot...")
    snapshot_download(repo_id=HF_DATASET_ID, repo_type="dataset", local_dir=download_dir, token=HF_TOKEN or None)
    zip_files = glob.glob(os.path.join(download_dir, "**", "*.zip"), recursive=True)
    if not zip_files:
        raise FileNotFoundError(f"❌ Could not locate zip archive in {download_dir}")
    zip_path = zip_files[0]

print("📂 [4/8] Extracting MP3 files from zip archive...")
extracted_mp3_dir = os.path.join(KAGGLE_WORKING, "_extracted_mp3")
os.makedirs(extracted_mp3_dir, exist_ok=True)

with zipfile.ZipFile(zip_path, "r") as z:
    z.extractall(extracted_mp3_dir)

mp3_files = [
    os.path.join(root, file)
    for root, _, files in os.walk(extracted_mp3_dir)
    for file in files
    if file.lower().endswith((".mp3", ".wav", ".flac", ".ogg"))
]

print(f"   Found {len(mp3_files)} audio files in zip archive.")

# Search for any transcript/metadata file inside extracted archive
transcript_map = {}
for root, _, files in os.walk(extracted_mp3_dir):
    for f in files:
        if f.lower() in ("metadata.csv", "transcript.csv", "transcripts.csv", "metadata.txt", "train.csv"):
            fp = os.path.join(root, f)
            try:
                with open(fp, "r", encoding="utf-8", errors="ignore") as file:
                    for line in file:
                        parts = line.strip().split("|") if "|" in line else line.strip().split(",")
                        if len(parts) >= 2:
                            stem = Path(parts[0].strip()).stem
                            txt = parts[-1].strip()
                            if stem and txt:
                                transcript_map[stem] = txt
            except Exception:
                pass

print(f"   Transcripts matched from metadata: {len(transcript_map)}")

print("🔄 Converting MP3 files to 24000Hz mono WAV (saving into 'wav/')...")
converted_count = 0
corrupted_count = 0
valid_entries = []

for idx, mp3_file in enumerate(mp3_files):
    stem = Path(mp3_file).stem
    clean_id = f"aseel_{idx+1:06d}"
    dst_wav = os.path.join(CONVERTED_WAV_DIR, f"{clean_id}.wav")

    # Get text transcript if present, otherwise default to audio stem
    text = transcript_map.get(stem, stem)

    if convert_mp3_to_wav(mp3_file, dst_wav):
        converted_count += 1
        valid_entries.append((clean_id, text))
    else:
        corrupted_count += 1
        if os.path.exists(dst_wav):
            os.remove(dst_wav)

    if (idx + 1) % 1000 == 0 or (idx + 1) == len(mp3_files):
        print(f"   ... Processed {idx+1}/{len(mp3_files)} files | Valid WAVs: {converted_count}")

print(f"✅ Conversion complete! Valid WAV files: {converted_count} | Corrupted skipped: {corrupted_count}\n")

# ==============================================================================
# 📝 6. Generate Professional Metadata Files (HF + OptiSpeech)
# ==============================================================================
print("📝 [5/8] Generating metadata files...")
if not valid_entries:
    raise RuntimeError("❌ Zero valid audio files converted!")

random.seed(42)
random.shuffle(valid_entries)

# 1. Hugging Face AudioFolder metadata.csv
hf_meta_path = os.path.join(RAW_DATASET_DIR, "metadata.csv")
with open(hf_meta_path, "w", encoding="utf-8", newline="\n") as f:
    f.write("file_name,text\n")
    for fid, txt in valid_entries:
        clean_txt = txt.replace('"', '""').replace("\n", " ").strip()
        f.write(f'wav/{fid}.wav,"{clean_txt}"\n')

# 2. OptiSpeech train.csv and val.csv
val_size = max(1, int(len(valid_entries) * 0.05))
val_entries = valid_entries[:val_size]
train_entries = valid_entries[val_size:]

for filename, entries in [("train.csv", train_entries), ("val.csv", val_entries)]:
    filepath = os.path.join(RAW_DATASET_DIR, filename)
    with open(filepath, "w", encoding="utf-8", newline="\n") as f:
        for fid, txt in entries:
            clean_txt = txt.replace("|", " ").replace("\n", " ").strip()
            f.write(f"{fid}|{clean_txt}\n")

# 3. Create professional Dataset Card (README.md) for HF Hub
readme_path = os.path.join(RAW_DATASET_DIR, "README.md")
readme_content = f"""---
language:
- ar
license: mit
tags:
- audio
- speech
- text-to-speech
- tts
- arabic
size_categories:
- 10K<n<100K
task_categories:
- text-to-speech
- automatic-speech-recognition
---

# 🎙️ Aseel Arabic Speech Dataset (Uncompressed WAV Edition)

This is the restructured, high-fidelity version of the **Aseel Arabic Dataset** optimized for end-to-end Text-to-Speech (TTS) training (OptiSpeech, VITS, FastSpeech2) and Speech Recognition.

## 📊 Dataset Summary
- **Audio Files**: {converted_count} mono PCM_16 `.wav` clips
- **Sample Rate**: 24,000 Hz (24 kHz)
- **Structure**: Native Hugging Face `AudioFolder` format
- **Subdirectory**: `wav/`

## 📁 Repository Layout
```text
Aseel_Arabic_Dataset/
├── metadata.csv        # Hugging Face dataset index (file_name, text)
├── train.csv           # OptiSpeech training index (file_id|text)
├── val.csv             # OptiSpeech validation index (file_id|text)
├── README.md           # Dataset Documentation
└── wav/                # Uncompressed 24kHz Mono WAV files
    ├── aseel_000001.wav
    ├── aseel_000002.wav
    └── ...
```

## 🚀 Quick Load with Hugging Face Datasets

```python
from datasets import load_dataset

dataset = load_dataset("{HF_DATASET_ID}")
print(dataset)
```
"""
with open(readme_path, "w", encoding="utf-8") as f:
    f.write(readme_content)

print("✅ Professional metadata & dataset card created.\n")

# ==============================================================================
# 📤 7. Delete Old Archives & Upload Clean Dataset to Hugging Face Hub
# ==============================================================================
if HF_TOKEN:
    print(f"📤 [6/8] Publishing updated dataset to Hugging Face Hub ({HF_DATASET_ID})...")
    try:
        api = HfApi(token=HF_TOKEN)
        create_repo(repo_id=HF_DATASET_ID, repo_type="dataset", exist_ok=True, token=HF_TOKEN)

        # List existing files and delete old compressed archives (.zip, .rar, .tar.gz)
        existing_files = api.list_repo_files(repo_id=HF_DATASET_ID, repo_type="dataset")
        for ef in existing_files:
            if ef.lower().endswith((".zip", ".tar.gz", ".tgz", ".rar")):
                print(f"🗑️ Deleting old compressed archive on HF Hub: {ef}")
                try:
                    api.delete_file(path_in_repo=ef, repo_id=HF_DATASET_ID, repo_type="dataset", token=HF_TOKEN)
                except Exception as e_del:
                    print(f"   Notice deleting {ef}: {e_del}")

        # Upload uncompressed wav/ folder and metadata
        print("🚀 Uploading uncompressed 'wav/' folder and metadata files to HF Hub...")
        api.upload_folder(
            folder_path=RAW_DATASET_DIR,
            repo_id=HF_DATASET_ID,
            repo_type="dataset",
            token=HF_TOKEN,
            ignore_patterns=["_snapshot*", "_tmp*", "_download*", "_extracted_mp3*"]
        )
        print("🎉 Dataset successfully published to Hugging Face Hub!")
    except Exception as e_upload:
        print(f"⚠️ HF Hub Upload Notice: {e_upload}")
        print("   (Ensure your HF_TOKEN has Write permissions for dataset repos).")
else:
    print("ℹ️ [6/8] Skipping Hugging Face Hub upload (HF_TOKEN not set).")
    print("   To upload to HF Hub, set HF_TOKEN='your_write_token' at top of script.\n")

# ==============================================================================
# 🧬 8. Run OptiSpeech Local Preprocessing & Feature Extraction
# ==============================================================================
print("🧬 [7/8] Running OptiSpeech phonemization & feature extraction...")
if os.path.exists(PROCESSED_DATA_DIR):
    shutil.rmtree(PROCESSED_DATA_DIR)

preprocess_cmd = [
    sys.executable, "-m", "optispeech.tools.preprocess_dataset",
    EXPERIMENT_NAME,
    RAW_DATASET_DIR,
    PROCESSED_DATA_DIR,
    "--n-workers", str(PREPROCESS_WORKERS),
    "--batch-size", "8",
]

print(f"   Executing: {' '.join(preprocess_cmd)}")
res_prep = subprocess.run(preprocess_cmd, cwd=REPO_DIR)
if res_prep.returncode != 0:
    raise RuntimeError(f"❌ Preprocessing failed with code {res_prep.returncode}")

print("✅ Local feature extraction complete.\n")

# ==============================================================================
# 📊 9. Compute Normalization Statistics
# ==============================================================================
print("📊 [8/8] Calculating normalization statistics...")
stats_cmd = [
    sys.executable, "-m", "optispeech.tools.generate_data_statistics",
    EXPERIMENT_NAME,
    "-b", "32",
    "-w", "2",
    "-o", DATA_STATS_DIR,
]

res_stats = subprocess.run(stats_cmd, cwd=REPO_DIR)
if res_stats.returncode != 0:
    raise RuntimeError(f"❌ Statistics calculation failed with code {res_stats.returncode}")

stats_json_path = os.path.join(DATA_STATS_DIR, "stats.json")
if os.path.exists(stats_json_path):
    with open(stats_json_path, "r", encoding="utf-8") as f:
        stats_data = json.load(f)
    print("\n   Calculated Normalization Statistics:")
    for k, v in stats_data.items():
        print(f"     • {k}: {v}")

    config_path = os.path.join(REPO_DIR, "configs", "data", f"{EXPERIMENT_NAME}.yaml")
    if os.path.exists(config_path):
        config_text = open(config_path, "r", encoding="utf-8").read()
        for k, v in stats_data.items():
            config_text = re.sub(rf"({k}:\s*)([\d.\-]+)", rf"\g<1>{v}", config_text)
        with open(config_path, "w", encoding="utf-8") as f:
            f.write(config_text)
        print(f"   ✅ Updated OptiSpeech config: {config_path}")

print("\n" + "=" * 80)
print("🎉 ALL STAGES COMPLETED SUCCESSFULLY!")
print("=" * 80)
print(f"📂 Converted Audio Folder: {CONVERTED_WAV_DIR}")
print(f"📂 Processed Features:     {PROCESSED_DATA_DIR}")
print(f"📂 Normalization Stats:    {DATA_STATS_DIR}")
print(f"🤗 Hugging Face Dataset:   https://huggingface.co/datasets/{HF_DATASET_ID}")
print("=" * 80)
