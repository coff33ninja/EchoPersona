# 🎙️ EchoPersona
**Proof of Concept: Custom TTS/STT with Character Voices**

## 🚀 Overview
This project is a full-stack playground for Text-to-Speech (TTS) and Speech-to-Text (STT) enthusiasts. It empowers you to:

- 🎧 **Structure Voice Data**: Includes a **proof-of-concept** custom downloader (`genshin_voice_downloader.py`) to fetch and structure character voice data from publicly available web sources¹. This component may be removed in future versions.
- 🗂️ **Manage Datasets**: Organize and preprocess voice data per character.
- 🧠 **Train Custom TTS Models**: Build personalized TTS models for each character using their voice samples, with options to resume training from checkpoints.
- 🗣️ **Generate Speech**: Use trained models to synthesize dialogue in a character’s voice.
- 🛠️ **Standard Tools Built-In**: Includes classic TTS (pyttsx3, gTTS), STT (Whisper, Google), and voice cloning (XTTS).

---

## 🧩 Key Concepts

**Character**: Represents a unique voice persona. All associated audio, metadata, and models live in folders named after them.
**Base Directories**:
- `voice_datasets/` – Raw audio data per character (e.g., `voice_datasets/Arlecchino/`).
- `trained_models/` – Output from model training (e.g., `trained_models/Arlecchino/`).

---

## 🌟 Features

### 🔁 Character-Specific Workflow
- Manage and train voice models per character.
- Includes the proof-of-concept downloader to help structure initial data.

### 🧰 Dataset Management (`voice_trainer_cli.py`)
- Record new samples.
- Add/annotate existing WAV/MP3 files.
- Validate dataset metadata.
- View stats: duration, sample count, format consistency.
- Augment audio (pitch, speed, noise).
- Trim silence and check audio quality.

### 🎓 Model Training
Train VITS-based models with `voice_clone_train.py` via the CLI or GUI. Supports:
- Custom epochs, batch size, learning rate.
- Resuming from checkpoints (`--continue_path`).
- Phoneme-based training (`--use_phonemes`, `--phoneme_language`).
- Adjustable sample rate (`--sample_rate`).

### 💬 Generate TTS
Use trained models to create character-specific speech outputs.

### 🎙️ STT Support
Use Whisper (recommended) or other engines to transcribe any audio file.

### 🎤 Pre-Trained Voice Cloning
Use XTTSv2 and other tools to clone voices from reference samples.

---

## 🎵 Audio Playback
- Play audio files using `pydub` or `simpleaudio` via helper functions.
- Supports WAV and MP3 formats.

---

## 🎛️ Audio Preprocessing
- Trim silence from audio files.
- Augment audio with background noise or pitch/speed adjustments.
- Validate audio quality for consistency.

---

## 🛠️ Installation

```bash
git clone https://github.com/USER/EchoPersona.git
cd EchoPersona
pip install -r requirements.txt
```

> ⚠️ **Note**:
- `openai-whisper` requires `ffmpeg` installed and accessible in your system's PATH.
- Coqui TTS (`TTS` package) may have specific OS/CUDA requirements. Check their [documentation](https://github.com/coqui-ai/TTS).
- Whisper models download on first use to `~/.cache/whisper`.
- For Vosk STT, download models manually from the [Vosk website](https://alphacephei.com/vosk/models).

---

## 📁 Folder Structure

```
EchoPersona/
├── voice_datasets/
│   ├── Arlecchino/           # Character dataset folder
│   │   ├── metadata.csv      # Format: audio_file|text|speaker
│   │   ├── sample_Arlecchino_001.wav
│   │   └── ...
├── trained_models/
│   ├── Arlecchino/           # Character model output folder
│   │   ├── training.log      # Training log file
│   │   ├── best_model.pth    # Trained model weights
│   │   ├── final_model.pth   # Final model weights
│   │   ├── config.json       # Model configuration
│   │   ├── final_config.json # Final configuration
│   │   ├── run-YYYYMMDD/    # Checkpoint folder (e.g., run-20250415)
│   │   │   ├── checkpoint.pth
│   │   │   └── ...
│   │   └── vocabulary.txt    # Phoneme vocabulary
├── logs/                      # Secondary logs (if used)
├── enhanced_logger.py         # Optional logging script
├── genshin_voice_downloader.py # Proof-of-concept downloader
├── genshin_voice_retranscriber.py # Optional re-transcription script
├── voice_trainer_cli.py       # CLI for dataset and training
├── voice_trainer_gui.py       # GUI application
├── voice_tools.py             # Core utilities
├── voice_clone_train.py       # Core training script
├── test_trained_model.py      # Test script for trained models
├── test_voice_tools.py        # Unit tests
├── requirements.txt           # Dependencies
├── background_noise.mp3       # Example noise for augmentation
├── Readme.md                  # This file
└── logo.png
```

---

## 🧪 Usage Workflow

### 1. 🔻 Structure Initial Voice Data (Optional)

```bash
python genshin_voice_downloader.py --character "Arlecchino" --output_dir voice_datasets
```
> **Note**: `genshin_voice_downloader.py` is a **proof-of-concept** to demonstrate structuring data from web sources¹ and may be removed in the future. Use responsibly.

Creates `voice_datasets/Arlecchino/` with WAVs and `metadata.csv`. The script uses Whisper for transcription.

> **Alternative**: Manually create `voice_datasets/Arlecchino/` with `metadata.csv` (format: `filename.wav|Transcription text|Arlecchino`) and audio files.

---

### 2. 🧰 Manage Dataset

Use `--character <Name>` with all `voice_trainer_cli.py` actions:

- **Record New Samples**:
  ```bash
  python voice_trainer_cli.py --character "Arlecchino" --action record
  ```

- **Add Existing Audio** (Prompts for transcription):
  ```bash
  python voice_trainer_cli.py --character "Arlecchino" --action provide --file "/path/to/audio.wav"
  ```

- **Validate Metadata** (Checks format & file existence):
  ```bash
  python voice_trainer_cli.py --character "Arlecchino" --action validate
  ```

- **View Stats** (Duration, file count, format warnings):
  ```bash
  python voice_trainer_cli.py --character "Arlecchino" --action stats
  ```

- **Augment File** (Adds pitch/speed variation or noise):
  ```bash
  python voice_trainer_cli.py --character "Arlecchino" --action augment --file "sample_Arlecchino_001.wav"
  ```

- **Trim Silence**:
  ```bash
  python voice_trainer_cli.py --character "Arlecchino" --action trim --file "sample_Arlecchino_001.wav"
  ```

---

### 3. 🧠 Train the Model

Training uses `voice_clone_train.py` internally. Run via CLI or GUI.

- **Via CLI**:
  ```bash
  # Basic command
  python voice_trainer_cli.py --character "Arlecchino" --action train

  # With custom parameters
  python voice_trainer_cli.py --character "Arlecchino" --action train --epochs 500 --batch_size 8 --learning_rate 0.0001 --base_dataset_dir "voice_datasets" --base_model_dir "trained_models" --sample_rate 22050 --use_phonemes true --phoneme_language en-us
  ```

  - **Resuming Training**:
    ```bash
    python voice_trainer_cli.py --character "Arlecchino" --action train --continue_path "trained_models/Arlecchino/run-YYYYMMDD"
    ```
    > **Note**: Ensure the `--continue_path` folder (e.g., `run-YYYYMMDD`) exists and contains a valid checkpoint (e.g., `checkpoint.pth`). Check `trained_models/Arlecchino/` for available runs.

- **Via GUI (`voice_trainer_gui.py`)**:
  - Select the character and `train` action.
  - Adjust sliders for epochs, batch size, learning rate, sample rate.
  - Specify `--continue_path` if resuming.
  - Click "Start Training".

**Output**: Model files (`best_model.pth`, `final_model.pth`, `config.json`, `final_config.json`), logs (`training.log`), and checkpoints (`run-YYYYMMDD/`) in `trained_models/Arlecchino/`.

#### **Training Parameters**:
- **`--epochs`** (Default: 100): Number of dataset passes. Higher values improve quality but risk overfitting.
- **`--batch_size`** (Default: 16): Samples per batch. Reduce (e.g., 8, 4) for memory errors.
- **`--learning_rate`** (Default: 0.001): Weight adjustment step. Lower (e.g., 0.0001) for stability.
- **`--base_dataset_dir`** (Default: `voice_datasets`): Path to dataset folder.
- **`--base_model_dir`** (Default: `trained_models`): Path for model outputs.
- **`--continue_path`** (Optional): Path to checkpoint folder (e.g., `trained_models/Arlecchino/run-YYYYMMDD`) to resume training.
- **`--sample_rate`** (Default: 22050): Audio sample rate (Hz). Match your dataset’s rate.
- **`--use_phonemes`** (Default: true): Use phonemes for better pronunciation.
- **`--phoneme_language`** (Default: en-us): Language for phonemes (required if `--use_phonemes true`).

#### **Troubleshooting Training**:
- **Error: "No models found in continue path"**:
  - Verify the `--continue_path` exists (e.g., `trained_models/Arlecchino/run-YYYYMMDD`).
  - Check for `checkpoint.pth` or similar files in the folder.
  - If incorrect or missing, remove `--continue_path` to start fresh or use the correct run folder.
  - Example: List runs with `dir trained_models\Arlecchino\run-*` (Windows).
- **CUDA Out of Memory**: Reduce `--batch_size` (e.g., 8 or 4).
- **Metadata Errors**: Run `--action validate` to check `metadata.csv`.
- **Training Fails**: Check `trained_models/Arlecchino/training.log` for details (e.g., file not found, phoneme issues).
- **Slow Data Loading**: Set `--num_loader_workers 0` in `voice_clone_train.py`.
- **Test Subset**: Try `--epochs 10` with a few files to debug.
- **Dependencies**: Ensure `ffmpeg`, PyTorch, and CUDA (if using GPU) are compatible.

---

### 4. 🗣️ Use the Trained Voice

- **Quick Test**:
  ```bash
  python voice_trainer_cli.py --character "Arlecchino" --action use --text "This is a test."
  ```

- **Manual Test**:
  ```bash
  python test_trained_model.py --character "Arlecchino" --text "Testing directly." --output_file "output.wav"
  ```

---

### 5. 🔄 Re-attempt Transcription (Optional)

For files with `<transcription_failed>` in `metadata.csv`:
- Use `genshin_voice_retranscriber.py` (if available).
- Or manually edit `metadata.csv` and use `--action provide`.

---

### 6. 🔎 General Speech-to-Text (STT)

```bash
python voice_tools.py
# Select character for structure
# Choose Option 11 for STT on any file
```

---

### 7. 🧪 Interactive Menu

```bash
python voice_tools.py
```

Guided menu for training, testing, cloning, and transcription.

---

## ⚠️ Notes and Best Practices

- **Paths**: Use absolute paths for `--base_dataset_dir`, `--base_model_dir`, `--continue_path` on Windows (e.g., `C:\Users\USER\Documents\GitHub\EchoPersona\voice_datasets`).
- **Integers**: Ensure `--epochs`, `--batch_size`, `--sample_rate` are integers.
- **Logs**: Always check `trained_models/<Character>/training.log` for errors.
- **Dataset**: Validate `metadata.csv` format (`audio_file|text|speaker`) before training.
- **Hardware**: Adjust `--batch_size` and `--num_loader_workers` based on your CPU/GPU.

---

## 🧪 Testing

```bash
python test_voice_tools.py
python test_trained_model.py
```

---

## 🧾 Dependencies (`requirements.txt`)

- `TTS` (Coqui TTS)
- `SpeechRecognition`
- `openai-whisper`
- `pydub`, `sounddevice`, `librosa`, `numpy`, `scipy`, `torch`
- `ffmpeg` (external, install separately)

---

## 🤝 Contributing

Pull requests welcome! Open an issue or fork to experiment.

---

## ⚠️ Disclaimer

The voice data downloader (`genshin_voice_downloader.py`) is a **technical proof-of-concept** to demonstrate fetching and structuring data from publicly available web sources¹. It relies on third-party APIs and website structures that may change and **may be removed in future updates**.

Voice data from external sources **may be subject to copyright**. Users must comply with relevant licenses and terms. This project:
- ❌ Does **not endorse** copyright infringement.
- ✅ Supports **personal, educational, non-commercial** use under fair use.
- 📦 Ships **no pre-trained models** derived from the downloader.
- ⚠️ Generated audio should not misrepresent or impersonate without consent.

---
¹ Data sourced primarily from Genshin Impact Fandom Wiki (`genshin-impact.fandom.com`) and `genshin.jmp.blue` API. Thanks to these communities for public access.

## 📝 License

MIT License — hack, build, remix. Just don’t be evil.

---
