import tkinter as tk
from tkinter import ttk, messagebox
import requests
import json
import re
import os
import subprocess
import argparse
import logging
import shutil
import csv
import pandas as pd
import numpy as np
from pydub import AudioSegment
from tqdm import tqdm
import unicodedata

# Ensure voice_tools.py is in the same directory or Python path
try:
    from voice_tools import SpeechToText
except ImportError:
    print("Warning: voice_tools.py not found. Transcription may fail if dependencies are missing.")
    logging.warning("Failed to import SpeechToText from voice_tools.py")
    SpeechToText = None

# Configure logging with UTF-8 encoding
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
    encoding="utf-8"
)

# --- Constants ---
BASE_DATA_DIR = "voice_datasets"
WIKI_API_URL = "https://genshin-impact.fandom.com/api.php"
JMP_API_URL_BASE = "https://genshin.jmp.blue"

# --- Helper Functions (Unchanged except for segment_audio_file) ---

def segment_audio_file(audio_path, output_dir, onset=0.6, offset=0.4, min_duration=2.0, min_duration_off=0.0, hf_token=""):
    """Segment audio using PyAnnote's segmentation model."""
    if not hf_token:
        logging.warning("No Hugging Face token provided. Skipping segmentation.")
        return []
    try:
        from pyannote.audio import Pipeline
        pipeline = Pipeline.from_pretrained("pyannote/segmentation", use_auth_token=hf_token)
        segments = pipeline(audio_path)
        audio = AudioSegment.from_file(audio_path)
        os.makedirs(output_dir, exist_ok=True)
        wav_files = []
        for i, segment in enumerate(segments.get_timeline()):
            start_ms = segment.start * 1000
            end_ms = segment.end * 1000
            if (end_ms - start_ms) / 1000 >= min_duration:
                segment_audio = audio[start_ms:end_ms]
                wav_path = os.path.join(output_dir, f"{i}.wav")
                segment_audio.export(wav_path, format="wav", parameters=["-ar", "22050", "-ac", "1"])
                wav_files.append(wav_path)
        return wav_files
    except ImportError:
        logging.error("PyAnnote not installed. Run 'pip install pyannote.audio'.")
        return []
    except Exception as e:
        logging.error(f"Error segmenting audio {audio_path}: {e}")
        return []

def clean_transcript(text, strict_ascii=False):
    """Clean transcription text, supporting Unicode or strict ASCII."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("|", " ").strip()
    if strict_ascii:
        text = text.encode("ascii", errors="ignore").decode("ascii")
    return text

def is_valid_for_phonemes(text):
    """Check if text contains characters likely to cause phoneme errors."""
    problematic = any(ord(c) > 127 and unicodedata.category(c).startswith("S") for c in text)
    return not problematic

def split_metadata(metadata_path, valid_ratio=0.2):
    """Split metadata into train and valid sets."""
    try:
        df = pd.read_csv(metadata_path, sep="|", encoding="utf-8")
        n_valid = int(len(df) * valid_ratio)
        indices = np.random.permutation(len(df))
        train_df = df.iloc[indices[n_valid:]]
        valid_df = df.iloc[indices[:n_valid]]
        train_path = metadata_path
        valid_path = os.path.join(os.path.dirname(metadata_path), "valid.csv")
        train_df.to_csv(train_path, sep="|", index=False, encoding="utf-8")
        valid_df.to_csv(valid_path, sep="|", index=False, encoding="utf-8")
        logging.info(f"Split metadata: {len(train_df)} train, {len(valid_df)} valid")
        return True
    except Exception as e:
        logging.error(f"Error splitting metadata {metadata_path}: {e}")
        return False

def get_category_files(category):
    """Fetches file titles from a Wiki category."""
    params = {
        "action": "query",
        "list": "categorymembers",
        "cmtype": "file",
        "cmtitle": category,
        "cmlimit": 500,
        "format": "json",
    }
    files = []
    cmcontinue = None
    while True:
        if cmcontinue:
            params["cmcontinue"] = cmcontinue
        try:
            response = requests.get(WIKI_API_URL, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()
        except requests.exceptions.RequestException as e:
            logging.error(f"Error fetching category files for {category}: {e}")
            break
        files.extend(
            [
                member["title"]
                for member in data["query"]["categorymembers"]
                if not re.search(r"Vo (JA|KO|ZH)", member["title"], re.IGNORECASE)
            ]
        )
        if "continue" in data and "cmcontinue" in data["continue"]:
            cmcontinue = data["continue"]["cmcontinue"]
        else:
            break
    logging.info(f"Found {len(files)} files in category '{category}'.")
    return files

def get_file_url(file_title):
    """Fetches the direct URL for a Wiki file."""
    params = {
        "action": "query",
        "titles": file_title,
        "prop": "imageinfo",
        "iiprop": "url",
        "format": "json",
    }
    try:
        response = requests.get(WIKI_API_URL, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        pages = data["query"]["pages"]
        page_id = list(pages.keys())[0]
        if "imageinfo" in pages[page_id]:
            return pages[page_id]["imageinfo"][0].get("url")
        return None
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching file URL for {file_title}: {e}")
        return None

def download_and_convert(file_url, output_dir, file_name, status_label=None):
    """Downloads OGG and converts to WAV."""
    safe_file_name = re.sub(r'[\\/*?:"<>|]', "_", file_name)
    ogg_file_name = safe_file_name if safe_file_name.lower().endswith(".ogg") else f"{safe_file_name}.ogg"
    wav_file_name = ogg_file_name.replace(".ogg", ".wav").replace(".OGG", ".wav")
    ogg_path = os.path.join(output_dir, ogg_file_name)
    wav_path = os.path.join(output_dir, wav_file_name)

    if os.path.exists(wav_path):
        logging.info(f"Skipping existing WAV: {wav_path}")
        if status_label:
            status_label.config(text=f"Skipped: {wav_file_name}")
        return wav_path

    try:
        if status_label:
            status_label.config(text=f"Downloading {ogg_file_name}...")
        response = requests.get(file_url, timeout=30)
        response.raise_for_status()
        with open(ogg_path, "wb") as f:
            f.write(response.content)
        subprocess.run(
            ["ffmpeg", "-y", "-i", ogg_path, "-ar", "22050", "-ac", "1", wav_path, "-loglevel", "error"],
            check=True, text=True
        )
        if status_label:
            status_label.config(text=f"Converted: {wav_file_name}")
        return wav_path
    except Exception as e:
        logging.error(f"Error processing {ogg_file_name}: {e}")
        if status_label:
            status_label.config(text=f"Error with {ogg_file_name}")
        return None
    finally:
        if os.path.exists(ogg_path):
            os.remove(ogg_path)

def fetch_character_list_from_api():
    """Fetches character names from jmp.blue API."""
    try:
        response = requests.get(f"{JMP_API_URL_BASE}/characters", timeout=10)
        response.raise_for_status()
        character_slugs = response.json()
        character_names = []
        for slug in character_slugs:
            try:
                char_response = requests.get(f"{JMP_API_URL_BASE}/characters/{slug}", timeout=5)
                char_response.raise_for_status()
                details = char_response.json()
                if "name" in details:
                    character_names.append(details["name"])
            except requests.exceptions.RequestException:
                pass
        return sorted(list(set(character_names)))
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching character list: {e}")
        return []

def transcribe_character_audio(character_output_dir, whisper_model="base", use_segmentation=False, hf_token="", strict_ascii=False, status_label=None):
    """Transcribes WAV files, with optional segmentation and Unicode support."""
    if SpeechToText is None:
        logging.error("Transcription unavailable: SpeechToText not imported.")
        if status_label:
            status_label.config(text="Transcription unavailable.")
        return

    metadata_path = os.path.join(character_output_dir, "metadata.csv")
    wavs_dir = os.path.join(character_output_dir, "wavs")
    os.makedirs(wavs_dir, exist_ok=True)

    existing_files = set()
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, "r", encoding="utf-8") as mf:
                lines = mf.readlines()
                for line in lines[1:]:
                    parts = line.strip().split("|")
                    if parts:
                        existing_files.add(parts[0])
        except UnicodeDecodeError:
            logging.error(f"Encoding error reading {metadata_path}. Ensure UTF-8 format.")
            return

    files_to_transcribe = []
    if use_segmentation and hf_token:
        for file in os.listdir(character_output_dir):
            if file.lower().endswith(".wav") and file not in os.listdir(wavs_dir):
                wav_path = os.path.join(character_output_dir, file)
                segmented_files = segment_audio_file(
                    wav_path, wavs_dir, onset=0.6, offset=0.4, min_duration=2.0, hf_token=hf_token
                )
                if segmented_files:
                    files_to_transcribe.extend([os.path.basename(f) for f in segmented_files])
                    os.remove(wav_path)
                else:
                    files_to_transcribe.append(file)  # Fallback to original
    else:
        files_to_transcribe = [
            file for file in os.listdir(character_output_dir)
            if file.lower().endswith(".wav") and file not in existing_files and not os.path.exists(os.path.join(wavs_dir, file))
        ]

    if not files_to_transcribe:
        logging.info("No new WAV files to transcribe.")
        if status_label:
            status_label.config(text="No new files to transcribe.")
        return

    logging.info(f"Transcribing {len(files_to_transcribe)} files with Whisper {whisper_model}...")
    transcribed_count = 0
    failed_count = 0
    file_mode = "a" if existing_files else "w"

    with open(metadata_path, file_mode, encoding="utf-8", newline="") as mf:
        if file_mode == "w":
            mf.write("audio_file|text|normalized_text|speaker_id\n")

        for file in tqdm(files_to_transcribe, desc="Transcribing"):
            wav_path = os.path.join(wavs_dir if file in os.listdir(wavs_dir) else character_output_dir, file)
            if status_label:
                status_label.config(text=f"Transcribing: {file}")
            try:
                stt = SpeechToText(
                    use_microphone=False,
                    audio_file=wav_path,
                    engine="whisper",
                    whisper_model_size=whisper_model,
                )
                audio_transcript = stt.process_audio(language="en")
                cleaned_transcript = clean_transcript(audio_transcript, strict_ascii=strict_ascii)
                if cleaned_transcript:
                    normalized = clean_transcript(cleaned_transcript, strict_ascii=True).lower().replace(".", "").replace(",", "")
                    if not is_valid_for_phonemes(normalized):
                        logging.warning(f"Potential phoneme issue in {file}: {normalized}")
                    metadata_entry = (
                        f"wavs/{file}|{cleaned_transcript}|{normalized}|speaker_1"
                        if file in os.listdir(wavs_dir)
                        else f"{file}|{cleaned_transcript}|{normalized}|speaker_1"
                    )
                    mf.write(metadata_entry + "\n")
                    transcribed_count += 1
                else:
                    metadata_entry = (
                        f"wavs/{file}|<transcription_failed>|<transcription_failed>|speaker_1"
                        if file in os.listdir(wavs_dir)
                        else f"{file}|<transcription_failed>|<transcription_failed>|speaker_1"
                    )
                    mf.write(metadata_entry + "\n")
                    failed_count += 1
            except Exception as e:
                logging.error(f"Transcription error for {file}: {e}")
                metadata_entry = (
                    f"wavs/{file}|<transcription_error>|<transcription_error>|speaker_1"
                    if file in os.listdir(wavs_dir)
                    else f"{file}|<transcription_error>|<transcription_error>|speaker_1"
                )
                mf.write(metadata_entry + "\n")
                failed_count += 1
            if status_label and "window" in globals() and window:
                window.update_idletasks()

    split_metadata(metadata_path, valid_ratio=0.2)

    final_status = f"Transcription complete. Successful: {transcribed_count}, Failed: {failed_count}."
    logging.info(final_status)
    if status_label:
        status_label.config(text=final_status)

def validate_metadata_existence(character_output_dir):
    """Check if metadata.csv exists."""
    metadata_path = os.path.join(character_output_dir, "metadata.csv")
    if not os.path.exists(metadata_path):
        logging.warning(f"Metadata file missing: {metadata_path}")
        return False
    return True

def validate_metadata_layout(metadata_path):
    """Validate metadata.csv format."""
    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        if len(lines) < 2:
            logging.error(f"Metadata file {metadata_path} is empty.")
            return False
        header = lines[0].strip()
        expected_headers = [
            "audio_file|text|normalized_text",
            "audio_file|text|normalized_text|speaker_id"
        ]
        if header not in expected_headers:
            logging.error(f"Invalid header in {metadata_path}: {header}")
            return False
        for i, line in enumerate(lines[1:], 2):
            if len(line.strip().split("|")) not in [3, 4]:
                logging.error(f"Invalid format at line {i} in {metadata_path}")
                return False
        return True
    except UnicodeDecodeError:
        logging.error(f"Encoding error in {metadata_path}. Ensure UTF-8 format.")
        return False
    except Exception as e:
        logging.error(f"Error validating {metadata_path}: {e}")
        return False

def process_character_voices(
    character, language, base_output_dir, download_wiki_audio=True, whisper_model="base",
    use_segmentation=False, hf_token="", strict_ascii=False, status_label=None
):
    """Downloads and processes voice lines."""
    safe_character_name = re.sub(r'[\\/*?:"<>|]', "_", character)
    character_folder = os.path.join(base_output_dir, safe_character_name)
    wavs_folder = os.path.join(character_folder, "wavs")
    os.makedirs(wavs_folder, exist_ok=True)

    if not download_wiki_audio:
        logging.info("Skipping Wiki download.")
        transcribe_character_audio(
            character_folder, whisper_model, use_segmentation, hf_token, strict_ascii, status_label
        )
        return character_folder

    categories = (
        [
            f"Category:{character} Voice-Overs",
            f"Category:English {character} Voice-Overs"
        ]
        if language == "English"
        else [f"Category:{language} {character} Voice-Overs"]
    )

    files_to_download = []
    for category in categories:
        files_to_download.extend(get_category_files(category))

    unique_files = sorted(list(set(files_to_download)))
    downloaded_count = 0
    failed_count = 0

    for i, file_title in enumerate(unique_files):
        file_name = re.match(r"File:(.*)", file_title).group(1).strip()
        file_url = get_file_url(file_title)
        if file_url:
            wav_file_path = download_and_convert(file_url, character_folder, file_name, status_label)
            if wav_file_path:
                downloaded_count += 1
            else:
                failed_count += 1
        else:
            failed_count += 1
        if status_label and "window" in globals() and window:
            window.update_idletasks()

    logging.info(f"Download complete. Downloaded: {downloaded_count}, Failed: {failed_count}.")
    transcribe_character_audio(
        character_folder, whisper_model, use_segmentation, hf_token, strict_ascii, status_label
    )

    # Move remaining WAV files to the 'wavs' folder
    for file in os.listdir(character_folder):
        if file.lower().endswith(".wav") and file not in os.listdir(wavs_folder):
            shutil.move(os.path.join(character_folder, file), os.path.join(wavs_folder, file))

    # Start TTS training after audio processing
    metadata_path = os.path.join(character_folder, "metadata.csv")
    if validate_metadata_existence(character_folder) and validate_metadata_layout(metadata_path):
        config_path = generate_character_config(
            character,
            character_folder,
            22050,  # Ensure this matches your audio sample rate
            selected_model="Fast Tacotron2"  # Default TTS model
        )
        if config_path:
            start_tts_training(config_path)
        else:
            logging.error("Failed to generate TTS config.")
    else:
        logging.error("Metadata validation failed. Skipping TTS training.")

    return character_folder

# --- New Functions for Config Generation and Training ---

AVAILABLE_MODELS = {
    "Fast Tacotron2": {
        "model_id": "tts_models/en/ljspeech/tacotron2-DDC",
        "use_pre_trained": True,
    },
    "High-Quality VITS": {
        "model_id": "tts_models/multilingual/multi-dataset/vits",
        "use_pre_trained": False,
    },
}

def generate_character_config(
    character,
    character_dir,
    sample_rate,
    selected_model="Fast Tacotron2",
    pre_trained_path=None,
):
    """Generates a Coqui TTS config.json for a specific character."""

    if selected_model not in AVAILABLE_MODELS:
        logging.error(f"Invalid model selected: {selected_model}")
        return None

    model_data = AVAILABLE_MODELS[selected_model]
    config = {
        "output_path": os.path.join(character_dir, "tts_output"),
        "datasets": [
            {
                "name": "ljspeech",
                "path": os.path.join(character_dir, "wavs"),
                "meta_file_train": os.path.join(character_dir, "train.csv"),
                "meta_file_val": os.path.join(character_dir, "valid.csv"),
            }
        ],
        "audio": {
            "sample_rate": sample_rate,
            "fft_size": 1024,
            "win_length": 1024,
            "hop_length": 256,
            "num_mels": 80,
            "mel_fmin": 0.0,
            "mel_fmax": 8000.0,
        },
        "model": model_data["model_id"],
        "batch_size": 16,
        "num_epochs": 100,
        "use_precomputed_alignments": False,
        "run_eval": True,
    }

    if model_data["use_pre_trained"] and pre_trained_path:
        config["restore_path"] = pre_trained_path

    config_path = os.path.join(character_dir, f"{character}_config.json")

    try:
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=4)
        logging.info(f"Config file generated: {config_path}")
        return config_path
    except Exception as e:
        logging.error(f"Error generating config for {character}: {e}")
        return None

def start_tts_training(config_path):
    """Starts Coqui TTS training using a generated config file."""
    try:
        subprocess.run(
            ["tts", "--config_path", config_path],
            check=True,
            text=True,
            capture_output=True
        )
        logging.info(f"Training started with config: {config_path}")
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"TTS Training failed with config: {config_path}: {e.stderr}")
        return False
    except FileNotFoundError:
        logging.error("Coqui TTS 'tts' command not found. Ensure it's installed and in your PATH.")
        return False
    except Exception as e:
        logging.error(f"Error starting TTS training: {e}")
        return False

# --- GUI ---

window = None

def main_gui():
    global window
    window = tk.Tk()
    window.title("Genshin Impact Voice Downloader & Transcriber")

    # Configuration Frame
    config_frame = ttk.LabelFrame(window, text="Configuration")
    config_frame.grid(row=0, column=0, padx=10, pady=10, sticky="ew")

    # Language Selection
    ttk.Label(config_frame, text="Language:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
    languages = ["English", "Japanese", "Chinese", "Korean"]
    language_var = tk.StringVar(value="English")
    ttk.Combobox(config_frame, textvariable=language_var, values=languages, state="readonly").grid(
        row=0, column=1, padx=5, pady=5, sticky="w"
    )

    # Character Selection
    ttk.Label(config_frame, text="Character:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
    character_names = fetch_character_list_from_api()
    character_var = tk.StringVar(value=character_names[0] if character_names else "No characters found")
    character_dropdown = ttk.Combobox(
        config_frame, textvariable=character_var, values=character_names, state="readonly", width=30
    )
    character_dropdown.grid(row=1, column=1, columnspan=2, padx=5, pady=5, sticky="ew")

    # Output Directory
    ttk.Label(config_frame, text="Output Dir:").grid(row=2, column=0, padx=5, pady=5, sticky="w")
    output_dir_var = tk.StringVar(value=BASE_DATA_DIR)
    ttk.Entry(config_frame, textvariable=output_dir_var, width=40).grid(
        row=2, column=1, columnspan=2, padx=5, pady=5, sticky="ew"
    )

    # Whisper Model Selection
    ttk.Label(config_frame, text="Whisper Model:").grid(row=3, column=0, padx=5, pady=5, sticky="w")
    whisper_models = ["base", "large-v2"]
    whisper_model_var = tk.StringVar(value="base")
    whisper_model_combo = ttk.Combobox(
        config_frame, textvariable=whisper_model_var, values=whisper_models, state="readonly"
    )
    whisper_model_combo.grid(row=3, column=1, padx=5, pady=5, sticky="w")

    # Segmentation Option
    use_segmentation_var = tk.BooleanVar(value=False)
    segmentation_check = ttk.Checkbutton(
        config_frame, text="Use Audio Segmentation", variable=use_segmentation_var
    )
    segmentation_check.grid(row=4, column=1, padx=5, pady=5, sticky="w")

    # Strict ASCII Option
    strict_ascii_var = tk.BooleanVar(value=False)
    ttk.Checkbutton(
        config_frame, text="Strict ASCII Transcriptions", variable=strict_ascii_var
    ).grid(row=5, column=1, padx=5, pady=5, sticky="w")

    # Hugging Face Token
    ttk.Label(config_frame, text="HF Token:").grid(row=6, column=0, padx=5, pady=5, sticky="w")
    hf_token_var = tk.StringVar()
    hf_token_entry = ttk.Entry(config_frame, textvariable=hf_token_var, width=40, show="*")
    hf_token_entry.grid(row=6, column=1, columnspan=2, padx=5, pady=5, sticky="ew")

    # Download Option
    download_wiki_audio_var = tk.BooleanVar(value=True)
    ttk.Checkbutton(
        config_frame, text="Download Wiki Audio", variable=download_wiki_audio_var
    ).grid(row=7, column=1, padx=5, pady=5, sticky="w")

    # TTS Model Selection
    ttk.Label(config_frame, text="TTS Model:").grid(row=8, column=0, padx=5, pady=5, sticky="w")
    tts_model_var = tk.StringVar(value="Fast Tacotron2")  # Default TTS model
    tts_model_combo = ttk.Combobox(
        config_frame, textvariable=tts_model_var, values=list(AVAILABLE_MODELS.keys()), state="readonly"
    )
    tts_model_combo.grid(row=8, column=1, padx=5, pady=5, sticky="w")

    # Control Frame
    control_frame = ttk.Frame(window)
    control_frame.grid(row=1, column=0, padx=10, pady=5, sticky="ew")

    status_label = ttk.Label(control_frame, text="Ready", relief=tk.SUNKEN, anchor="w", width=60)
    status_label.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5, pady=5)

    # Function to update HF token field state
    def update_hf_token_state(*args):
        if whisper_model_var.get() == "base" and not use_segmentation_var.get():
            hf_token_entry.config(state="disabled")
        else:
            hf_token_entry.config(state="normal")

    # Bind updates to Whisper model and segmentation checkbox
    whisper_model_var.trace_add("write", update_hf_token_state)
    use_segmentation_var.trace_add("write", update_hf_token_state)
    update_hf_token_state()  # Initial state

    def start_processing():
        character = character_var.get()
        language = language_var.get()
        base_output_dir = output_dir_var.get()
        whisper_model = whisper_model_var.get()
        use_segmentation = use_segmentation_var.get()
        strict_ascii = strict_ascii_var.get()
        hf_token = hf_token_var.get()
        download_wiki_audio = download_wiki_audio_var.get()
        selected_tts_model = tts_model_var.get()  # Get selected TTS model

        if not character or character == "No characters found":
            messagebox.showerror("Error", "Select a valid character.")
            return
        if not base_output_dir:
            messagebox.showerror("Error", "Enter an output directory.")
            return
        if use_segmentation and not hf_token:
            messagebox.showerror("Error", "Hugging Face token required for segmentation.")
            return

        download_button.config(state="disabled")
        status_label.config(text=f"Processing {character}...")
        window.update_idletasks()

        character_folder_path = process_character_voices(
            character, language, base_output_dir, download_wiki_audio,
            whisper_model, use_segmentation, hf_token, strict_ascii, status_label
        )

        if character_folder_path:
            if validate_metadata_existence(character_folder_path):
                metadata_path = os.path.join(character_folder_path, "metadata.csv")
                if validate_metadata_layout(metadata_path):
                    # Generate TTS config
                    config_path = generate_character_config(
                        character,
                        character_folder_path,
                        22050,  # Ensure this matches your audio sample rate
                        selected_model=selected_tts_model  # Use selected model
                    )
                    if config_path:
                        # Start TTS training
                        if start_tts_training(config_path):
                            status_label.config(text=f"Training started: {character}")
                        else:
                            status_label.config(text=f"Training failed: {character}")
                    else:
                        status_label.config(text=f"Config generation failed: {character}")
                else:
                    status_label.config(text="Invalid metadata layout.")
            else:
                status_label.config(text="Metadata missing.")
        else:
            status_label.config(text="Processing failed.")

        download_button.config(state="normal")

    download_button = ttk.Button(control_frame, text="Process Voices", command=start_processing)
    download_button.pack(side=tk.RIGHT, padx=5, pady=5)

    window.grid_columnconfigure(0, weight=1)
    config_frame.grid_columnconfigure(1, weight=1)
    window.mainloop()

# --- Main Execution ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and transcribe Genshin Impact voice data.")
    parser.add_argument("--character", type=str, help="Character name (e.g., Arlecchino).")
    parser.add_argument("--output_dir", type=str, default=BASE_DATA_DIR, help="Base output directory.")
    parser.add_argument("--language", type=str, default="English", choices=["English", "Japanese", "Chinese", "Korean"])
    parser.add_argument("--whisper_model", type=str, default="base", choices=["base", "large-v2"], help="Whisper model size.")
    parser.add_argument("--use_segmentation", action="store_true", help="Use PyAnnote segmentation.")
    parser.add_argument("--strict_ascii", action="store_true", help="Force ASCII-only transcriptions.")
    parser.add_argument("--hf_token", type=str, default="", help="Hugging Face token for segmentation.")
    parser.add_argument("--skip_wiki_download", action="store_true", help="Skip Wiki download.")
    args = parser.parse_args()

    if args.character is None:
        main_gui()
    else:
        logging.info(f"Processing {args.character} via CLI...")
        character_folder_path = process_character_voices(
            args.character, args.language, args.output_dir,
            not args.skip_wiki_download, args.whisper_model,
            args.use_segmentation, args.hf_token, args.strict_ascii
        )
        if character_folder_path:
            if validate_metadata_existence(character_folder_path):
                metadata_path = os.path.join(character_folder_path, "metadata.csv")
                validate_metadata_layout(metadata_path)
        logging.info(f"Finished processing {args.character}.")