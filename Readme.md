Okay, I understand. You want to adjust the language in the `Readme.md` to be more cautious about the downloader component, frame it as a proof-of-concept, and acknowledge the data sources used.

Here is the revised `Readme.md` reflecting those changes:

```markdown
# 🎙️ EchoPersona
**Proof of Concept: Custom TTS/STT with Character Voices**

## 🚀 Overview
This project is a full-stack playground for Text-to-Speech (TTS) and Speech-to-Text (STT) enthusiasts. It empowers you to:

- 🎧 **Structure Voice Data**: Includes a **proof-of-concept** custom downloader (`genshin_voice_downloader.py`) to fetch and structure character voice data from publicly available web sources¹. This component may be removed in future versions.
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
- Manage, and train voice models per character.
- Includes the aforementioned proof-of-concept downloader to help structure initial data.

### 🧰 Dataset Management (`voice_trainer_cli.py`)
- Record new samples
- Add/annotate existing WAV/MP3 files
- Validate dataset metadata
- View stats: duration, sample count, format consistency
- Augment audio (pitch, speed, noise)
- Trim silence and check audio quality

### 🎓 Model Training
Train VITS-based models with `voice_clone_train.py` via the CLI or GUI.

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
git clone <repository-url>
cd ECHOPERSONA
pip install -r requirements.txt
```

> ⚠️ **Note**:
- `openai-whisper` needs `ffmpeg` installed and accessible in your system's PATH.
- Coqui TTS (`TTS` package) may have specific OS/CUDA requirements. Check their documentation.
- Whisper models download on first use to `~/.cache/whisper`.
- For Vosk STT, download models manually from the Vosk website.

---

## 📁 Folder Structure

```
Your-Project-Directory/
├── voice_datasets/
│   ├── Amber/                # Character dataset folder
│   │   ├── metadata.csv      # LJSpeech format: audio_file|text|normalized_text
│   │   ├── sample_Amber_....wav
│   │   └── VO_Amber_....wav
├── trained_models/
│   ├── Amber/                # Character model output folder
│   │   ├── training.log      # Main training log file
│   │   ├── best_model.pth    # Trained model weights
│   │   ├── config.json       # Model configuration
│   │   └── ...
├── logs/                       # Secondary logs (if using separate logger setup)
│   └── ...
├── enhanced_logger.py        # Optional separate logging script
├── genshin_voice_downloader.py # Proof-of-concept downloader
├── voice_trainer_cli.py
├── voice_trainer_gui.py      # GUI Application
├── voice_tools.py
├── voice_clone_train.py      # Core training script
├── test_trained_model.py
├── test_voice_tools.py
├── requirements.txt
├── background_noise.mp3      # Example noise file for augmentation
├── Readme.md                 # This file
└── logo.png
```

---

## 🧪 Usage Workflow

### 1. 🔻 Structure Initial Voice Data (Optional)

```bash
python genshin_voice_downloader.py --character "Amber" --output_dir voice_datasets
```
> **Note:** This script (`genshin_voice_downloader.py`) is a **proof-of-concept** intended to demonstrate structuring data from web sources¹ and may be removed in the future. Use responsibly.

Creates `voice_datasets/Amber/` with WAVs + `metadata.csv`. The script attempts to transcribe audio using Whisper.

> You can skip this and directly create your own dataset folder manually, ensuring `metadata.csv` follows the LJSpeech format: `filename.wav|Transcription text|normalized transcription text`.

---

### 2. 🧰 Manage Dataset

Use `--character <Name>` with all `voice_trainer_cli.py` actions:

- **Record New Samples**
  ```bash
  python voice_trainer_cli.py --character "Amber" --action record
  ```

- **Add Existing Audio** (Prompts for transcription)
  ```bash
  python voice_trainer_cli.py --character "Amber" --action provide --file "/path/to/audio.wav"
  ```

- **Validate Metadata** (Checks format & file existence)
  ```bash
  python voice_trainer_cli.py --character "Amber" --action validate
  ```

- **View Stats** (Total duration, file count, format warnings)
  ```bash
  python voice_trainer_cli.py --character "Amber" --action stats
  ```

- **Augment File** (Adds pitch/speed variation or noise)
  ```bash
  python voice_trainer_cli.py --character "Amber" --action augment --file "relative/path/within/dataset/sample_Amber_001.wav"
  ```

- **Trim Silence** (From start/end of file)
  ```bash
  python voice_trainer_cli.py --character "Amber" --action trim --file "relative/path/within/dataset/VO_Amber_001.wav"
  ```

---

### 3. 🧠 Train the Model

This uses the `voice_clone_train.py` script internally. You can run training via the CLI or GUI.

- **Via CLI:**
  ```bash
  # Basic command
  python voice_trainer_cli.py --character "Amber" --action train

  # With custom parameters (see Guidelines below)
  python voice_trainer_cli.py --character "Amber" --action train --epochs 500 --batch_size 16 --learning_rate 0.0001
  ```

- **Via GUI (`voice_trainer_gui.py`):**
  * Select the Character and the `train` action.
  * Adjust sliders for Epochs, Batch Size, and Learning Rate.
  * Click "Start Training".

Output (model files, config, logs) lands in `trained_models/Amber/`. Training can take a significant amount of time depending on dataset size, hardware, and epochs.

#### **Customizing Training Parameters:**

You can adjust several parameters to influence the training process, available via both CLI flags (for `voice_trainer_cli.py --action train`) and the GUI sliders:

* **`--epochs`** (Default: 1000 in `voice_clone_train.py`, 100 in CLI default): Number of passes through the entire dataset. More epochs generally lead to better quality but take longer and risk overfitting. Adjust based on dataset size and observed results in the logs/output directory.
* **`--batch_size`** (Default: 32 in `voice_clone_train.py`, 16 in CLI default): Number of audio samples processed in one go. **Crucial for memory usage.** If you get "CUDA out of memory" errors or the process crashes, **reduce this value** (e.g., 16, 8, 4). Larger batch sizes can sometimes speed up training if memory allows.
* **`--learning_rate`** (Default: 0.0002 in `voice_clone_train.py`, 0.001 in CLI default): Controls how much the model weights are adjusted during training. The default is often a good starting point. If training is unstable (loss fluctuates wildly or becomes `NaN` in logs), you might try slightly lowering it (e.g., 0.0001).

You can also edit `voice_clone_train.py` directly for more advanced options:

* **`num_loader_workers`**: Number of parallel processes for loading data. Increase if your CPU is bottlenecking data loading (monitor CPU usage). Decrease (even to 0) if you encounter strange multiprocessing errors during startup or data loading.
* **`use_phonemes`**: Whether to convert text to phonemes (recommended for better pronunciation). Requires a phonemizer for the specified `phoneme_language`.
* **`mixed_precision`**: Can speed up training and reduce memory on compatible GPUs, but defaulted to `False` in the current script. Set to `True` to try enabling it.

#### **Troubleshooting Training:**

* **Check Logs:** Always check `trained_models/<Character>/training.log` first for specific error messages.
* **Validate Data:** Use the `validate` action often.
* **Reduce Batch Size:** The most common fix for memory-related crashes.
* **Check Dependencies:** Ensure `requirements.txt` are met, `ffmpeg` is installed, and PyTorch/CUDA versions are compatible.
* **Try `num_loader_workers 0`:** Helps diagnose data loading issues.
* **Test on Subset:** Try training on just 5-10 files to ensure the basic pipeline works.

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

### 5. 🔄 Re-attempt Transcription (If using Downloader)

If the initial download/transcription process left files with `<transcription_failed>`, you can use the `genshin_voice_retranscriber.py` script (if available/updated) or manually provide transcriptions using the `provide` action. The downloader script also attempts to re-transcribe files marked "Validation Needed".

---

### 6. 🔎 General Speech-to-Text (STT)

Use the interactive menu in `voice_tools.py` (Option 11) or adapt the `SpeechToText` class for your own scripts. STT works independently of character models.

```bash
python voice_tools.py
# Choose a character to initialize (needed for structure)
# Select Option 11 to run STT on any file
```

---

### 7. 🧪 Full Interactive Menu

```bash
python voice_tools.py
```

This script gives you a guided, menu-driven experience—train, test, clone, transcribe—all in one.

---

## ⚠️ Notes and Best Practices

- **Integer Values**: Ensure that `epochs` and `batch_size` are integers when using the GUI or CLI to avoid compatibility issues.
- **Training Parameters**: Adjust parameters based on your dataset size, observed results (monitor logs/output quality), and hardware capabilities. Start with defaults and tune carefully.
- **Error Handling**: If you encounter errors, always check the logs in the relevant output directory (`trained_models/<Character>/`) for details.

---

## 🧪 Testing
- Run unit tests using `unittest` (if test files like `test_voice_tools.py` are provided and configured).
- Example:
  ```bash
  # If tests are in a 'tests' directory
  # python -m unittest discover -s tests

  # Or run specific test files
  python test_voice_tools.py
  ```

---

## 🧾 Dependencies (see `requirements.txt`)

Key packages:
- `TTS` (Coqui TTS)
- `SpeechRecognition`
- `openai-whisper`
- `pydub`, `sounddevice`, `librosa`, `numpy`, `scipy`, `torch`
- `ffmpeg` (external dependency, must be installed separately)

---

## 🤝 Contributing

Pull requests are welcome! Feel free to open an issue or fork and experiment.

---

## ⚠️ Disclaimer

The included voice data downloader (`genshin_voice_downloader.py`) is provided solely as a **technical proof-of-concept** to demonstrate fetching and structuring data from publicly available web sources¹. It relies on third-party APIs and website structures that may change. This component **may be removed in future updates**.

Voice data obtained from external sources **may be subject to copyright**. Users are responsible for ensuring their use complies with relevant licenses and terms of service. This project:
- ❌ Does **not endorse** activities that infringe on copyright.
- ✅ Supports **personal, educational, non-commercial** experimentation under fair use principles.
- 📦 Ships **no pre-trained models** derived directly from data gathered by the proof-of-concept downloader.
- ⚠️ Audio generated using any tool should not be used to misrepresent or impersonate individuals without explicit consent.

---
¹ The proof-of-concept downloader utilizes data primarily from the Genshin Impact Fandom Wiki (`genshin-impact.fandom.com`) and the `genshin.jmp.blue` API. We thank these communities for making information publicly accessible.

## 📝 License

MIT License — hack it, build it, remix it. Just don't be evil.

---
```
