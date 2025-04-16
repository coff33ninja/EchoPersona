Genshin Impact Voice Downloader & TTS Trainer
This project, genshin_voice_downloader.py, enables you to download voice-overs from the Genshin Impact wiki, transcribe them, train a text-to-speech (TTS) model using Coqui TTS, and generate custom MP3s for characters like Arlecchino. With both CLI and GUI support, it’s ideal for creating lines like “Would you sit by my side?” in Arlecchino’s voice. This README provides step-by-step instructions for your Windows setup, tailored to your existing project folder and 59 WAV files (30 successful, 11 failed, 18 silent as of April 16, 2025).
Features

Download: Fetches voice-overs from Genshin Impact wiki.
Transcribe: Uses Whisper (base or large-v2) to generate metadata.csv and valid.csv.
Train: Builds a TTS model with customizable batch_size, num_epochs, and learning_rate.
Test: Generates test audio to verify model quality.
Synthesize: Creates MP3s via GUI with custom text, saved as Arlecchino_tts_YYYYMMDD_HHMMSS.mp3.
Options: Supports segmentation (--use_segmentation), strict ASCII (--strict_ascii), and resuming training (--resume_from_checkpoint).

Prerequisites

OS: Windows 10/11
Python: 3.11.9
Project Directory: C:\Users\USER\Documents\GitHub\EchoPersona
Character: Arlecchino (default)
Data: ~59 WAVs in voice_datasets\Arlecchino\wavs (from April 16, 2025)
Dependencies: ffmpeg, Python packages (below)
Optional: Hugging Face token for audio segmentation

Setup Instructions
1. Navigate to Project
Open PowerShell and go to your project folder.
cd C:\Users\USER\Documents\GitHub\EchoPersona

2. Set Up Virtual Environment
Activate your existing virtual environment.
.venv\Scripts\Activate.ps1

If missing:
python -m venv .venv
.venv\Scripts\Activate.ps1

3. Install Dependencies
Install required Python packages.
pip install requests pandas numpy pydub tqdm TTS playsound python-slugify

Install ffmpeg:

Download from gyan.dev.
Extract to C:\ffmpeg.
Add to PATH:$env:Path += ";C:\ffmpeg\bin"


Verify:ffmpeg -version



For audio segmentation (optional):
pip install pyannote.audio


Requires a Hugging Face token (set in step 5).

4. Verify Script
Ensure genshin_voice_downloader.py is in C:\Users\USER\Documents\GitHub\EchoPersona and includes clean_metadata_file and validate_metadata_existence (fixes for NameError, April 16, 2025).
If outdated/missing:

Update with the latest script (artifact b6a459cb-46bc-41bf-b0e0-255c9dc8f3e4, version 1f6f83c3-4787-4443-9766-36639c9f2af4).
Save to C:\Users\USER\Documents\GitHub\EchoPersona\genshin_voice_downloader.py.

5. Check Voice Data
Verify your WAV files (~59 expected).
Get-ChildItem -Path voice_datasets\Arlecchino\wavs\*.wav | Measure-Object

If fewer than expected:

Download voice-overs:python genshin_voice_downloader.py --character Arlecchino --output_dir voice_datasets --language English


Saves WAVs to voice_datasets\Arlecchino\wavs.


With segmentation:python genshin_voice_downloader.py --character Arlecchino --output_dir voice_datasets --language English --use_segmentation --hf_token YOUR_HF_TOKEN


Replace YOUR_HF_TOKEN with your token from huggingface.co.


Strict ASCII (optional, for cleaner transcriptions):python genshin_voice_downloader.py --character Arlecchino --output_dir voice_datasets --language English --strict_ascii



Running the Pipeline
6. Transcribe Audio
Transcribe WAVs to generate metadata.csv and valid.csv, addressing failed (11) and silent (18) files.
python genshin_voice_downloader.py --character Arlecchino --output_dir voice_datasets --language English --skip_wiki_download --whisper_model base

For higher accuracy (recommended):
python genshin_voice_downloader.py --character Arlecchino --output_dir voice_datasets --language English --skip_wiki_download --whisper_model large-v2

With segmentation:
python genshin_voice_downloader.py --character Arlecchino --output_dir voice_datasets --language English --skip_wiki_download --whisper_model large-v2 --use_segmentation --hf_token YOUR_HF_TOKEN

With strict ASCII:
python genshin_voice_downloader.py --character Arlecchino --output_dir voice_datasets --language English --skip_wiki_download --whisper_model large-v2 --strict_ascii

Outputs:

voice_datasets\Arlecchino\wavs: ~30+ valid WAVs
voice_datasets\Arlecchino\silent_files: Failed/silent WAVs
voice_datasets\Arlecchino\metadata.csv: ~30 entries
voice_datasets\Arlecchino\valid.csv: ~6 entries

7. Verify Silent Files
Check files in silent_files to ensure they’re truly silent.
Get-ChildItem -Path voice_datasets\Arlecchino\silent_files\*.wav

If valid audio found:

Play to confirm:Start-Process voice_datasets\Arlecchino\silent_files\0.wav


Move back and re-transcribe:Move-Item -Path voice_datasets\Arlecchino\silent_files\*.wav -Destination voice_datasets\Arlecchino\wavs
Remove-Item -Path voice_datasets\Arlecchino\metadata.csv

Repeat step 6 with --whisper_model large-v2.

8. Validate Metadata
Ensure metadata.csv is clean and valid.
Get-Content voice_datasets\Arlecchino\metadata.csv

If empty or invalid:
Remove-Item -Path voice_datasets\Arlecchino\metadata.csv

Repeat step 6.
9. Train TTS Model
Test training with small parameters to verify setup.
python genshin_voice_downloader.py --character Arlecchino --output_dir voice_datasets --batch_size 4 --num_epochs 5 --learning_rate 0.0001

Full training:
python genshin_voice_downloader.py --character Arlecchino --output_dir voice_datasets --batch_size 8 --num_epochs 100 --learning_rate 0.0001

Resume training (if interrupted):
python genshin_voice_downloader.py --character Arlecchino --output_dir voice_datasets --batch_size 8 --num_epochs 100 --learning_rate 0.0001 --resume_from_checkpoint voice_datasets\tts_train_output\Arlecchino\checkpoints\checkpoint_latest.pth

Outputs:

voice_datasets\Arlecchino\Arlecchino_config.json
voice_datasets\tts_train_output\Arlecchino\checkpoints

10. Test Model
Generate a test audio to check model quality.
python genshin_voice_downloader.py --character Arlecchino --output_dir voice_datasets --test_model

Output:

voice_datasets\tts_train_output\Arlecchino\test_output.wav

11. Play Test Audio
Verify the model sounds like Arlecchino.
Start-Process voice_datasets\tts_train_output\Arlecchino\test_output.wav

12. Generate MP3s with GUI
Create custom MP3s with the GUI.
python genshin_voice_downloader.py


Select “Arlecchino” and voice_datasets.
In “TTS Feedback”, enter text (e.g., “Would you sit by my side?”).
Click “Speak”.
Saves to voice_datasets\tts_train_output\Arlecchino\tts_outputs\Arlecchino_tts_YYYYMMDD_HHMMSS.mp3.

13. Organize Outputs
List generated MP3s.
Get-ChildItem -Path voice_datasets\tts_train_output\Arlecchino\tts_outputs\*.mp3

Full Workflow
Run these commands for a complete pipeline:
cd C:\Users\USER\Documents\GitHub\EchoPersona
.venv\Scripts\Activate.ps1
pip install requests pandas numpy pydub tqdm TTS playsound python-slugify
Get-ChildItem -Path voice_datasets\Arlecchino\wavs\*.wav | Measure-Object
python genshin_voice_downloader.py --character Arlecchino --output_dir voice_datasets --language English --skip_wiki_download --whisper_model large-v2
Get-Content voice_datasets\Arlecchino\metadata.csv
python genshin_voice_downloader.py --character Arlecchino --output_dir voice_datasets --batch_size 4 --num_epochs 5 --learning_rate 0.0001
python genshin_voice_downloader.py --character Arlecchino --output_dir voice_datasets --test_model
Start-Process voice_datasets\tts_train_output\Arlecchino\test_output.wav
python genshin_voice_downloader.py  # GUI for MP3s

CLI Options
The script supports the following arguments:

--character: Character name (e.g., Arlecchino).
--output_dir: Output directory (default: voice_datasets).
--language: Language for voice-overs (default: English, options: English, Japanese, Chinese, Korean).
--whisper_model: Whisper model for transcription (default: base, options: base, large-v2).
--use_segmentation: Enable PyAnnote audio segmentation (requires --hf_token).
--hf_token: Hugging Face token for segmentation.
--strict_ascii: Force ASCII-only transcriptions.
--skip_wiki_download: Skip downloading from wiki, use existing WAVs.
--batch_size: Batch size for training (default: 16).
--num_epochs: Number of training epochs (default: 100).
--learning_rate: Learning rate for training (default: 0.0001).
--resume_from_checkpoint: Path to checkpoint to resume training.
--test_model: Test the trained model after training.

Troubleshooting

Transcription Failures (e.g., 11 failed, 18 silent, April 16, 2025):
Retry step 6 with --whisper_model large-v2.
Check silent_files (step 7).


PermissionError (April 15, 2025):Remove-Item -Recurse -Force voice_datasets\tts_train_output\Arlecchino


Metadata Errors (e.g., “not enough values”, April 16, 2025):Remove-Item -Path voice_datasets\Arlecchino\metadata.csv

Repeat step 6.
Missing voice_tools.py:
Ensure it’s in the project folder for SpeechToText.
If missing, transcription fails—reinstall or skip segmentation.


UnicodeEncodeError (April 15, 2025):
Script now uses UTF-8; verify file encodings.


Coqui TTS Errors:
Force CPU if GPU issues:$env:CUDA_VISIBLE_DEVICES=""




Slow Training:
Test with --batch_size 4 --num_epochs 5.


No Audio Output:
Check WAV/MP3 associations in Windows.
Verify playsound logs in GUI.



Notes

Data Safety:
Back up metadata.csv before re-transcribing:Copy-Item -Path voice_datasets\Arlecchino\metadata.csv -Destination voice_datasets\Arlecchino\metadata_backup.csv




Performance Tips:
Use --whisper_model large-v2 for better transcription accuracy.
Adjust batch_size (e.g., 4 for low memory, 8 for full training).


Verification:
Always check metadata.csv and silent_files to catch issues early.


GUI:
Ideal for generating MP3s with custom text; CLI handles downloading and training.



Support
If you encounter errors (e.g., low transcription counts or new crashes), check PowerShell logs or share them for assistance. This pipeline is optimized for your 59 WAVs and includes fixes for past issues (NameError, metadata errors). Get ready to hear Arlecchino’s voice in stunning MP3s! 🎙️
