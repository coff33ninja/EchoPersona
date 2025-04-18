---

# 🎙️ EchoPersona
**Proof of Concept: Custom TTS/STT with Character Voices**

## 🚀 Overview
EchoPersona is a modular toolkit for enthusiasts of Text-to-Speech (TTS) and Speech-to-Text (STT) systems. It enables you to:

- 🎧 **Download and Structure Voice Data**: Use a proof-of-concept downloader to fetch character voice data from public web sources¹ (e.g., Genshin Impact Fandom Wiki).
- 🗂️ **Manage Datasets**: Organize, transcribe, and preprocess voice data for individual characters.
- 🧠 **Train Custom TTS Models**: Fine-tune TTS models using Coqui TTS for character-specific voices.
- 🗣️ **Generate Speech**: Synthesize dialogue in a character’s voice using trained models.
- 🔍 **Inspect TTS Framework**: Analyze Coqui TTS modules to understand training-related components.

> ⚠️ **Note**: The downloader (`genshin_voice_downloader1.py`) is a **proof-of-concept** and may be removed in future updates. Use responsibly and comply with source licensing.

---

## 🧩 Key Concepts

- **Character**: A unique voice persona with dedicated audio, metadata, and model files stored in a folder (e.g., `voice_datasets/Hu Tao/`).
- **Base Directories**:
  - `voice_datasets/`: Stores raw audio, metadata (`metadata.csv`, `valid.csv`), and configuration files per character.
  - `tts_output/`: Stores trained model checkpoints, final models (`final_model.pth`), and configuration files.

---

## 🌟 Features

### 🔁 Character-Specific Workflow
- Download voice data for characters (e.g., Hu Tao, Xiao) from public sources.
- Transcribe audio using Whisper for TTS-ready datasets.
- Train and test character-specific TTS models.
- Inspect Coqui TTS internals for debugging and customization.

### 🎧 Voice Data Downloader (`genshin_voice_downloader1.py`)
- Fetches audio files from Genshin Impact Fandom Wiki and `genshin.jmp.blue` API.
- Converts OGG to WAV format (22kHz, mono).
- Transcribes audio using Whisper (base, small, medium, large-v2 models).
- Supports audio segmentation (requires Hugging Face token).
- Generates phonemes using Gruut for enhanced TTS training.
- Creates `metadata.csv` and `valid.csv` for training and validation.
- Produces Coqui TTS configuration files (`<character>_config.json`).
- GUI and CLI modes for flexible operation.

### 🧠 Model Training (`train_tts_model.py`)
- Trains TTS models using Coqui TTS (e.g., Tacotron2, VITS) on character datasets.
- Supports GPU (CUDA, MPS) and CPU training.
- Resumes training from checkpoints (`--restore-path`).
- Configurable batch size, epochs, and evaluation settings.
- Tests trained models by synthesizing sample text.
- GUI for interactive training and monitoring.
- CLI for batch processing multiple characters.

### 🔍 TTS Framework Inspection (`list_tts_functions.py`)
- Analyzes Coqui TTS modules for training-related classes (e.g., `Trainer`, `TTS`) and methods (e.g., `train`, `tts`).
- Logs detailed signatures of relevant functions and classes.
- Helps debug compatibility issues with Coqui TTS versions.
- Outputs results to `tts_inspection.log`.

### 💬 Speech Synthesis
- Generate WAV files from text using trained models.
- Supports custom test text for model evaluation.

### 🎙️ Speech-to-Text (STT)
- Transcribes audio files using Whisper during dataset creation.
- Filters silent or low-quality audio to improve dataset quality.

### 🎵 Audio Preprocessing
- Converts audio to 22kHz mono WAV format.
- Detects and moves silent audio to a separate folder.
- Supports optional audio segmentation for precise transcription.

---

## 🛠️ Installation

```bash
git clone https://github.com/coff33ninja/EchoPersona.git
cd EchoPersona
pip install -r requirements.txt
```

> ⚠️ **Notes**:
- Install `ffmpeg` and ensure it’s in your system PATH (`ffmpeg -version` to verify).
- Coqui TTS (`TTS`) requires PyTorch and may need CUDA for GPU training. See [Coqui TTS documentation](https://github.com/coqui-ai/TTS).
- Whisper models are downloaded to `~/.cache/whisper` on first use.
- Audio segmentation requires a Hugging Face token and `pyannote.audio`.
- GUI requires `tkinter` (included with Python on most systems).

---

## 📁 Folder Structure

```
EchoPersona/
├── voice_datasets/
│   ├── Hu Tao/              # Character dataset folder
│   │   ├── wavs/            # WAV audio files
│   │   ├── metadata.csv     # Format: text|audio_file|phonemes
│   │   ├── valid.csv        # Validation split
│   │   ├── Hu Tao_config.json # Coqui TTS config
│   │   ├── silent_files/    # Silent or failed audio
│   │   └── tts_output/      # Training outputs
├── tts_output/
│   ├── Hu Tao/              # Character model output folder
│   │   ├── checkpoints/     # Training checkpoints
│   │   ├── final_model.pth  # Final model weights
│   │   ├── config.json      # Training configuration
│   │   └── hu_tao_test.wav  # Test audio output
├── logs/
│   ├── train_tts_model.log  # Training logs
│   ├── tts_inspection.log   # TTS module inspection logs
├── genshin_voice_downloader1.py # Voice data downloader
├── train_tts_model.py          # TTS model training
├── list_tts_functions.py       # TTS framework inspection
├── requirements.txt            # Dependencies
├── Readme.md                   # This file
└── logo.png
```

---

## 🧪 Usage Workflow

### 1. 🔻 Download and Structure Voice Data

Use `genshin_voice_downloader1.py` to fetch and prepare voice data.

- **CLI Example**:
  ```bash
  python genshin_voice_downloader1.py process \
      --character "Hu Tao" \
      --output-dir voice_datasets \
      --whisper-model base \
      --tts-model "Fast Tacotron2"
  ```
  - Creates `voice_datasets/Hu Tao/` with `wavs/`, `metadata.csv`, `valid.csv`, and `Hu Tao_config.json`.
  - Downloads audio from Genshin Impact Fandom Wiki, converts to WAV, transcribes with Whisper, and generates phonemes.

- **GUI Example**:
  ```bash
  python genshin_voice_downloader1.py
  ```
  - Select a character (e.g., Hu Tao), output directory, and TTS model.
  - Configure Whisper model, segmentation, and CSV headers.
  - Monitor progress in the Status tab and review `metadata.csv` in the Transcriptions tab.

> **Alternative**: Manually create `voice_datasets/Hu Tao/` with `wavs/` (containing WAV files), `metadata.csv` (format: `text|audio_file|phonemes`), and `valid.csv`.

---

### 2. 🧠 Train the TTS Model

Use `train_tts_model.py` to train a TTS model on the prepared dataset.

- **CLI Example**:
  ```bash
  # Basic training
  python train_tts_model.py \
      --base-dir voice_datasets \
      --character "Hu Tao" \
      --output-dir tts_output \
      --test-output hu_tao_test.wav

  # Resume training
  python train_tts_model.py \
      --base-dir voice_datasets \
      --character "Hu Tao" \
      --output-dir tts_output \
      --restore-path tts_output/Hu Tao/checkpoints/checkpoint.pth
  ```
  - Trains a model using the configuration in `voice_datasets/Hu Tao/Hu Tao_config.json`.
  - Saves checkpoints and `final_model.pth` to `tts_output/Hu Tao/`.
  - Generates a test WAV file (`hu_tao_test.wav`).

- **GUI Example**:
  ```bash
  python train_tts_model.py --gui
  ```
  - Select a character, output directory, and optional restore path.
  - Enable GPU training and choose CUDA or MPS.
  - Specify test text and output WAV path.
  - Monitor training progress in the status window.

**Training Parameters**:
- `--base-dir` (Default: `voice_datasets`): Dataset directory.
- `--character`: Character name (e.g., Hu Tao).
- `--output-dir` (Default: `tts_output`): Model output directory.
- `--use-gpu`: Enable GPU training.
- `--gpu-type` (Default: `cuda`): GPU type (`cuda` or `mps`).
- `--restore-path`: Path to a checkpoint for resuming training.
- `--test-text` (Default: "Hiya, I’m Hu Tao..."): Text for testing.
- `--test-output` (Default: `hu_tao_test.wav`): Test audio output path.

**Troubleshooting**:
- **KeyError: 'formatter'**: Add `"formatter": "ljspeech"` to `Hu Tao_config.json` under `datasets`.
- **TypeError: 'output_path'**: Update `train_tts_model.py` to remove `output_path` from `TrainerArgs`.
- **RAdam Error**: Allowlist `RAdam` in `initialize_model` (see provided fixes).
- **CUDA Out of Memory**: Reduce batch size in the config (e.g., `"batch_size": 8`).
- **Metadata Errors**: Verify `metadata.csv` and `valid.csv` format and paths.
- **Logs**: Check `train_tts_model.log` for detailed errors.

---

### 3. 🔍 Inspect Coqui TTS Framework

Use `list_tts_functions.py` to analyze Coqui TTS modules.

```bash
python list_tts_functions.py
```
- Inspects modules like `TTS.api`, `TTS.bin.train_tts`, and `TTS.utils.synthesizer`.
- Logs classes (e.g., `Trainer`, `TTS`) and methods (e.g., `train`, `tts`) with signatures to `tts_inspection.log`.
- Useful for debugging compatibility issues or understanding Coqui TTS internals.

---

### 4. 🗣️ Test the Trained Model

Test the model by synthesizing audio from text.

- **CLI Example**:
  ```bash
  python train_tts_model.py \
      --base-dir voice_datasets \
      --character "Hu Tao" \
      --test-text "Hello, this is Hu Tao!" \
      --test-output hu_tao_test.wav
  ```
- **GUI Example**: Use the GUI to specify test text and output path during training.

**Output**: A WAV file (e.g., `hu_tao_test.wav`) with synthesized audio.

---

## ⚠️ Notes and Best Practices

- **Paths**: Use absolute paths on Windows (e.g., `C:\EchoPersona\voice_datasets`).
- **Config Files**: Ensure `<character>_config.json` includes `datasets`, `model`, and `output_path`.
- **Metadata Format**: `metadata.csv` and `valid.csv` must have `text|audio_file|phonemes` columns.
- **Logs**: Check `train_tts_model.log` and `tts_inspection.log` for errors.
- **Hardware**: Adjust batch size and GPU settings based on your system.
- **Dataset Quality**: Use `--min-silence-duration` (e.g., 0.5s) to filter silent audio.
- **Coqui TTS Version**: Ensure compatibility with PyTorch 2.6+ (update `TTS` if needed).

---

## 🧾 Dependencies (`requirements.txt`)

- `TTS` (Coqui TTS)
- `openai-whisper`
- `pydub`, `pandas`, `numpy`, `requests`, `tqdm`, `gruut`
- `pygame` (for GUI audio playback)
- `pyannote.audio` (optional, for segmentation)
- `torch` (with CUDA support for GPU)
- `ffmpeg` (external, required)

---

## 🤝 Contributing

Pull requests and issues are welcome! Fork the repository to experiment.

---

## ⚠️ Disclaimer

The voice downloader (`genshin_voice_downloader1.py`) is a **technical proof-of-concept** for fetching data from public sources¹ (Genshin Impact Fandom Wiki, `genshin.jmp.blue`). It may be removed in future updates. Users must:

- ❌ Avoid copyright infringement.
- ✅ Use for **personal, educational, non-commercial** purposes under fair use.
- 📦 Note: No pre-trained models are included.
- ⚠️ Avoid misrepresenting or impersonating with generated audio.

¹ Thanks to Genshin Impact Fandom Wiki and `genshin.jmp.blue` for public data access.

## 📝 License

MIT License — hack, build, remix responsibly.

---

Troubleshooting
Load with weights_only=False If you trust the model checkpoint source (e.g., it’s downloaded from Coqui TTS’s official repository), you can bypass the weights_only restriction by setting weights_only=False in torch.load. Modify the TTS library’s utils/io.py file (around line 54) or patch it in your script. However, this is less secure and not recommended unless necessary:
# In venv/.venv/python-path 
TTS/utils/io.py, line 54 in my case set this to False
return torch.load(f, map_location=map_location, weights_only=False, **kwargs)
# Warning: Setting weights_only=False can lead to arbitrary code execution if the checkpoint is from an untrusted source. Use this only if you’re certain of the checkpoint’s origin.