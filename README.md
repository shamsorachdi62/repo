# Text-to-Speech Engine (OptiSpeech)

A lightweight, efficient, and fast End-to-End Text-to-Speech (TTS) engine designed for high performance, low-latency, and on-device streaming inference.

---

## 🚀 Quick Start in Google Colab & Linux

To import and use this repository directly in **Google Colab** or any Linux environment, execute the following commands:

```bash
# Clone the private repository (replace PAT with your Personal Access Token if private)
!git clone https://github.com/shamsorachdi62/repo.git
%cd repo

# Install dependencies and package
!pip install -e .
```

Alternatively, install directly via `pip`:

```bash
!pip install git+https://github.com/shamsorachdi62/repo.git
```

---

## 🛠️ Local Installation

### Prerequisites
- Python 3.10+
- PyTorch 2.0+

```bash
git clone https://github.com/shamsorachdi62/repo.git
cd repo
pip install -e .
```

---

## 💻 Usage

### Python API

```python
import torch
import soundfile as sf
from optispeech.model import OptiSpeech

# 1. Load trained model from checkpoint
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ckpt_path = "path/to/checkpoint.ckpt"
model = OptiSpeech.load_from_checkpoint(ckpt_path, map_location=device)
model = model.to(device).eval()

# 2. Prepare text input (supports Arabic Tashkeel & Phonetic G2P)
text = "مرحبا بكم في نظام توليد الكلام المباشر."
inference_inputs = model.prepare_input(text, d_factor=1.0, p_factor=1.0)

# 3. Synthesize audio
outputs = model.synthesise(inference_inputs)
wav = outputs.wav.squeeze().numpy()

# 4. Save output waveform
sf.write("output.wav", wav, model.sample_rate)
```

### Command Line Interface (CLI)

```bash
python3 -m optispeech.infer checkpoint.ckpt "نص تجريبي للتحويل إلى صوت" ./output_dir --cuda
```

---

## 🏋️ Training

### 1. Dataset Preparation

Organize your dataset in LJSpeech format:

```text
dataset_dir/
├── train/
│   ├── metadata.csv
│   └── wav/
│       └── sample_001.wav
└── val/
    ├── metadata.csv
    └── wav/
        └── sample_002.wav
```

`metadata.csv` format (`|` separated):
- 2 columns: `file_id|text`
- 3 columns: `file_id|speaker_id|text`
- 4 columns: `file_id|speaker_id|language_id|text`

Preprocess dataset:
```bash
python3 -m optispeech.tools.preprocess_dataset saeed ./input_raw ./processed_output
```

Generate Pitch/F0 statistics:
```bash
python3 -m optispeech.tools.generate_data_statistics saeed
```

### 2. Run Training

```bash
python3 -m optispeech.train experiment=saeed
```

---

## ⚡ ONNX & Low-Latency Streaming

### Export ONNX Model

```bash
python3 -m optispeech.onnx.export checkpoint.ckpt ./exported_model.onnx
```

### Run ONNX Streaming Inference

```bash
python3 -m optispeech.onnx.infer ./exported_model.onnx "نص تجريبي للبث المباشر" ./output_dir
```

---

## 📄 License

Distributed under the MIT License. See [LICENSE](./LICENSE) for details.
