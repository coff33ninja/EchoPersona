# 🎙️ EchoPersona
**Proof of Concept: Custom TTS/STT with Character Voices**

## 🚀 Overview  
This project is a full-stack playground for Text-to-Speech (TTS) and Speech-to-Text (STT) enthusiasts. It empowers you to:

- 🎧 **Download Voice Data**: Automatically fetch and transcribe voice lines from characters (e.g., Genshin Impact).
- 🗂️ **Manage Datasets**: Organize and preprocess voice data per character.
- 🧠 **Train Custom TTS Models**: Build a personalized TTS model for each character using their voice samples.
- 🗣️ **Generate Speech**: Use trained models to synthesize dialogue in that character’s voice.
- 🛠️ **Standard Tools Built-In**: Includes classic TTS (pyttsx3, gTTS), STT (Whisper, Google), and voice cloning (XTTS).

---

## 🧩 Key Concepts

**Character**: Represents a unique voice persona. All associated audio, metadata, and models live in folders named after them.  
**Base Directories**:  
- `voice_datasets/` – Raw audio data per character  
- `trained_models/` – Output from model training

---

## 🌟 Features

### 🔁 Character-Specific Workflow
- Download, manage, and train voice models per character
- Genshin voice downloader automates grabbing and transcribing lines

### 🧰 Dataset Management (`voice_trainer_cli.py`)
- Record new samples
- Add/annotate existing WAV/MP3 files
- Validate dataset metadata
- View stats: duration, sample count, format consistency
- Augment audio (pitch, speed, noise)
- Trim silence and check audio quality

### 🎓 Model Training
Train VITS-based models with `voice_clone_train.py` via CLI.

### 💬 Generate TTS
Use trained models to create character-specific speech outputs.

### 🎙️ STT Support
Use Whisper (recommended) or other engines to transcribe any audio file.

### 🎤 Pre-Trained Voice Cloning
Use XTTSv2 and other tools to clone voices from reference samples.

---

## 🎵 Audio Playback
- Play audio files using `pydub` or `simpleaudio`.
- Supports WAV and MP3 formats.

---

## 🎛️ Audio Preprocessing
- Trim silence from audio files.
- Augment audio with background noise or pitch/speed adjustments.
- Validate audio quality for consistency.

---

## 🛠️ Installation

```bash
git clone <repository-url>
cd Ultimate-Open-Source-Multi-Voice-Radio
pip install -r requirements.txt
```

> ⚠️ **Note**:
- `openai-whisper` needs `ffmpeg` installed.
- Coqui TTS may have OS/CUDA-specific requirements.
- Whisper models download on first use (`~/.cache/whisper`).
- For Vosk STT, download models manually.

---

## 📁 Folder Structure

```
Your-Project-Directory/
├── voice_datasets/
│   ├── Amber/
│   │   ├── metadata.csv
│   │   ├── sample_Amber_....wav
│   │   └── VO_Amber_....wav
├── trained_models/
│   ├── Amber/
│   │   ├── best_model.pth
│   │   ├── config.json
│   │   └── ...
├── genshin_voice_downloader.py
├── voice_trainer_cli.py
├── voice_tools.py
├── voice_clone_train.py
├── test_trained_model.py
├── test_voice_tools.py
├── requirements.txt
├── background_noise.mp3
└── logo.png
```

---

## 🧪 Usage Workflow

### 1. 🔻 Fetch Initial Voice Data (Optional)

```bash
python genshin_voice_downloader.py --character "Amber" --output_dir voice_datasets
```

Creates `voice_datasets/Amber/` with WAVs + `metadata.csv`.

> You can skip this and directly create your own dataset folder manually.

---

### 2. 🧰 Manage Dataset

Use `--character <Name>` with all actions:

- **Record New Samples**  
  ```bash
  python voice_trainer_cli.py --character "Amber" --action record
  ```

- **Add Existing Audio**  
  ```bash
  python voice_trainer_cli.py --character "Amber" --action provide --file "/path/to/audio.wav"
  ```

- **Validate Metadata**  
  ```bash
  python voice_trainer_cli.py --character "Amber" --action validate
  ```

- **View Stats**  
  ```bash
  python voice_trainer_cli.py --character "Amber" --action stats
  ```

- **Augment File**  
  ```bash
  python voice_trainer_cli.py --character "Amber" --action augment --file "sample_Amber_001.wav"
  ```

- **Trim Silence**  
  ```bash
  python voice_trainer_cli.py --character "Amber" --action trim --file "VO_Amber_001.wav"
  ```

---

### 3. 🧠 Train the Model

```bash
python voice_trainer_cli.py --character "Amber" --action train
```

Output lands in `trained_models/Amber/`. Training may take hours, coffee recommended.

---

### 4. 🗣️ Use the Trained Voice

- **Quick Test via CLI**  
  ```bash
  python voice_trainer_cli.py --character "Amber" --action use --text "Hello, this is a test."
  ```

- **Manual Test Script**  
  ```bash
  python test_trained_model.py --character "Amber" --text "Testing the model directly." --output_file "output.wav"
  ```

---

### 5. 🔎 General Speech-to-Text (STT)

```bash
python voice_tools.py
# Choose a character to initialize (needed for structure)
# Select Option 11 to run STT on any file
```

STT works independently of character models.

---

### 6. 🧪 Full Interactive Menu

```bash
python voice_tools.py
```

This script gives you a guided, menu-driven experience—train, test, clone, transcribe—all in one.

---

## 🧪 Testing
- Run unit tests using `unittest` to validate functionality.
- Example:
  ```bash
  python -m unittest discover -s tests
  ```

---

## 🧾 Dependencies (see `requirements.txt`)

Key packages:
- `TTS` (Coqui TTS)
- `SpeechRecognition`
- `openai-whisper`
- `pydub`, `sounddevice`, `librosa`, `numpy`, `scipy`, `torch`
- `ffmpeg` (external)

---

## 🤝 Contributing

Pull requests are welcome! Feel free to open an issue or fork and experiment. Let's make talking computers cooler together.

---

## ⚠️ Disclaimer

The Genshin Impact voice downloader is included as a **technical proof of concept**.  
Voice data from external sources **may be copyrighted**.  
Use at your own risk. This project:
- ❌ Does **not endorse** copyright infringement.
- ✅ Supports **personal, educational, non-commercial** use under fair use.
- 📦 Ships **no pre-trained models** derived from such sources.
- ⚠️ Augmented audio should not be used to misrepresent or impersonate individuals.

## 📝 License

MIT License — hack it, build it, remix it. Just don't be evil.

---
