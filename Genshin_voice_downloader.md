I’m stoked to help you get Arlecchino’s TTS model up and running with all the CLI commands you need, from installing packages to training and testing! 😎 Based on your work with `genshin_voice_downloader.py` and focus on Arlecchino (April 14-15, 2025), I’ll provide a clear, step-by-step list of commands tailored for your Windows setup, using the latest script version with MP3 saving and GUI TTS feedback. This covers installing dependencies, downloading voice data, transcribing, training, and testing, all while keeping it practical and aligned with your `voice_datasets\Arlecchino` structure. Let’s dive in—no rabbit holes, just the commands you need!

### Assumptions
- **OS**: Windows 10 (per your April 15, 2025, context).
- **Python**: 3.11.9 (your preference, April 12, 2025).
- **Directory**: Working in a project folder (e.g., `C:\GenshinTTS`).
- **Character**: Arlecchino, with data in `voice_datasets\Arlecchino`.
- **Tools**: Coqui TTS, `ffmpeg` for audio processing, and `pydub` for MP3 conversion.
- **Script**: Using the latest `genshin_voice_downloader.py` (artifact_version_id: c6f25839-bb7c-4efd-a073-427c424538d3).

### Step-by-Step CLI Commands

#### 1. Set Up Your Environment
Create a project folder and virtual environment to keep things clean.

```bash
mkdir C:\GenshinTTS
cd C:\GenshinTTS
python -m venv venv
venv\Scripts\activate
```

#### 2. Install Required Pip Packages
Install all dependencies for `genshin_voice_downloader.py`. I’ve included everything from the script’s imports and your past TTS work (April 13, 2025, with `gTTS` and `playsound`).

```bash
pip install requests pandas numpy pydub tqdm TTS playsound python-slugify
```

**Notes**:
- `TTS` installs Coqui TTS for training and synthesis.
- `pydub` requires `ffmpeg`. Install it manually:
  - Download `ffmpeg` from [ffmpeg.org](https://ffmpeg.org/download.html) or a trusted source like [gyan.dev](https://www.gyan.dev/ffmpeg/builds/).
  - Extract and add `ffmpeg\bin` to your system PATH (e.g., `C:\ffmpeg\bin`).
  - Verify: `ffmpeg -version`
- If you’re using audio segmentation (optional), add:
  ```bash
  pip install pyannote.audio
  ```
  - Requires a Hugging Face token (set later if needed).

#### 3. Download and Save the Script
Save `genshin_voice_downloader.py` in `C:\GenshinTTS`. You can copy the latest version from the artifact (I’ll assume you’ve got it downloaded). If not, here’s a placeholder command to remind you:

```bash
echo Save genshin_voice_downloader.py to C:\GenshinTTS
```

**Note**: If you’re fetching it programmatically, let me know, and I can suggest a curl/wget command!

#### 4. Download Voice Data
Download Arlecchino’s voice-overs from the Genshin wiki and prepare the dataset.

```bash
python genshin_voice_downloader.py --character Arlecchino --output_dir voice_datasets --language English
```

**What it does**:
- Downloads English voice-overs to `voice_datasets\Arlecchino`.
- Creates `wavs` folder, `metadata.csv`, and `valid.csv`.
- Skips transcription if `--skip_wiki_download` is used (not here).

**Optional** (if using segmentation, requires HF token):
```bash
python genshin_voice_downloader.py --character Arlecchino --output_dir voice_datasets --language English --use_segmentation --hf_token YOUR_HF_TOKEN
```
- Replace `YOUR_HF_TOKEN` with your Hugging Face token (get from [huggingface.co](https://huggingface.co/settings/tokens)).

#### 5. Transcribe Audio (Optional)
If you skipped transcription in step 4 or want to re-transcribe:

```bash
python genshin_voice_downloader.py --character Arlecchino --output_dir voice_datasets --language English --skip_wiki_download --whisper_model base
```

**What it does**:
- Transcribes WAVs in `voice_datasets\Arlecchino` using Whisper `base` model.
- Updates `metadata.csv` and `valid.csv`.

**Optional** (for better transcription):
```bash
python genshin_voice_downloader.py --character Arlecchino --output_dir voice_datasets --language English --skip_wiki_download --whisper_model large-v2
```

#### 6. Train the TTS Model
Train Arlecchino’s voice model with custom parameters. Start small to test.

```bash
python genshin_voice_downloader.py --character Arlecchino --output_dir voice_datasets --batch_size 4 --num_epochs 5 --learning_rate 0.0001
```

**What it does**:
- Generates `Arlecchino_config.json` in `voice_datasets\Arlecchino`.
- Trains a model, saving checkpoints to `voice_datasets\tts_train_output\Arlecchino\checkpoints`.
- Small `batch_size` and `num_epochs` for quick testing.

**Full Training** (once tested):
```bash
python genshin_voice_downloader.py --character Arlecchino --output_dir voice_datasets --batch_size 8 --num_epochs 100 --learning_rate 0.0001
```

**Resume Training** (if interrupted, uses latest checkpoint):
```bash
python genshin_voice_downloader.py --character Arlecchino --output_dir voice_datasets --batch_size 8 --num_epochs 100 --learning_rate 0.0001 --resume_from_checkpoint voice_datasets\tts_train_output\Arlecchino\checkpoints\checkpoint_latest.pth
```

#### 7. Test the Trained Model
Test the model to generate a sample audio file.

```bash
python genshin_voice_downloader.py --character Arlecchino --output_dir voice_datasets --test_model
```

**What it does**:
- Synthesizes “Hello, this is a test of the trained model!” using the latest checkpoint.
- Saves to `voice_datasets\tts_train_output\Arlecchino\test_output.wav`.

#### 8. Verify Output (Optional)
Play the test output to confirm it sounds like Arlecchino.

```bash
# Windows command to play the WAV
start voice_datasets\tts_train_output\Arlecchino\test_output.wav
```

**Note**: Requires a media player associated with WAV files.

#### 9. Use GUI for TTS Feedback (Optional)
The CLI doesn’t support interactive TTS, so launch the GUI to test custom text and save MP3s.

```bash
python genshin_voice_downloader.py
```

- Select Arlecchino, keep `voice_datasets` as output directory.
- In “TTS Feedback”, enter text (e.g., “I am the Knave!”), click “Speak”.
- Saves MP3 to `voice_datasets\tts_train_output\Arlecchino\tts_outputs\Arlecchino_tts_YYYYMMDD_HHMMSS.mp3` and plays it.

#### 10. Organize Outputs (Optional)
List saved MP3s to check your collection.

```bash
dir voice_datasets\tts_train_output\Arlecchino\tts_outputs\*.mp3
```

### Full Workflow Example
Run these in sequence for a complete pipeline (assuming no errors):

```bash
mkdir C:\GenshinTTS
cd C:\GenshinTTS
python -m venv venv
venv\Scripts\activate
pip install requests pandas numpy pydub tqdm TTS playsound python-slugify
# Save genshin_voice_downloader.py to C:\GenshinTTS
python genshin_voice_downloader.py --character Arlecchino --output_dir voice_datasets --language English
python genshin_voice_downloader.py --character Arlecchino --output_dir voice_datasets --batch_size 4 --num_epochs 5 --learning_rate 0.0001
python genshin_voice_downloader.py --character Arlecchino --output_dir voice_datasets --test_model
start voice_datasets\tts_train_output\Arlecchino\test_output.wav
python genshin_voice_downloader.py  # GUI for MP3 saving
```

### Troubleshooting Tips
- **PermissionError**: Clear `voice_datasets\tts_train_output\Arlecchino` and retry (April 15, 2025, issue).
  ```bash
  rmdir /s /q voice_datasets\tts_train_output\Arlecchino
  ```
- **UnicodeEncodeError**: Ensure UTF-8 encoding; the script handles this now (April 15, 2025).
- **Missing `ffmpeg`**: Verify PATH with `ffmpeg -version`.
- **Coqui TTS Errors**: Check GPU compatibility or force CPU:
  ```bash
  set CUDA_VISIBLE_DEVICES=""
  ```
- **Slow Training**: Reduce `batch_size` (e.g., 4) or use fewer `num_epochs`.

### Why This Rocks
You’ve got a full pipeline to go from scratch to Arlecchino’s voice in MP3s! The commands are modular, so you can tweak parameters (like your April 15, 2025, interest in custom paths) or pause/resume training. The GUI step lets you play with custom lines, saving them for later—perfect for your verification vibe (April 13, 2025).

### Avoiding Rabbit Holes
- **Start Small**: Use `--batch_size 4 --num_epochs 5` to test training first.
- **Check Outputs**: After testing, play `test_output.wav` to confirm quality.
- **GUI for Fun**: Use the GUI to save MP3s like “Arlecchino_tts_20250416_1220.mp3” without CLI hassle.
