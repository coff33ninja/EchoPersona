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
import datetime
import threading
import queue
from TTS.api import TTS
from TTS.utils.synthesizer import Synthesizer
import playsound

try:
    from voice_tools import SpeechToText
except ImportError:
    print("Warning: voice_tools.py not found. Transcription may fail.")
    logging.warning("Failed to import SpeechToText from voice_tools.py")
    SpeechToText = None

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

# --- Helper Functions ---

def segment_audio_file(audio_path, output_dir, onset=0.6, offset=0.4, min_duration=2.0, min_duration_off=0.0, hf_token=""):
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
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("|", " ").strip()
    if strict_ascii:
        text = text.encode("ascii", errors="ignore").decode("ascii")
    return text

def is_valid_for_phonemes(text):
    problematic = any(ord(c) > 127 and unicodedata.category(c).startswith("S") for c in text)
    return not problematic

def split_metadata(metadata_path, valid_ratio=0.2):
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

def download_and_convert(file_url, output_dir, file_name, status_queue=None):
    safe_file_name = re.sub(r'[\\/*?:"<>|]', "_", file_name)
    ogg_file_name = safe_file_name if safe_file_name.lower().endswith(".ogg") else f"{safe_file_name}.ogg"
    wav_file_name = ogg_file_name.replace(".ogg", ".wav").replace(".OGG", ".wav")
    ogg_path = os.path.join(output_dir, ogg_file_name)
    wav_path = os.path.join(output_dir, wav_file_name)

    if os.path.exists(wav_path):
        logging.info(f"Skipping existing WAV: {wav_path}")
        if status_queue:
            status_queue.put(f"Skipped: {wav_file_name}")
        return wav_path

    try:
        if status_queue:
            status_queue.put(f"Downloading {ogg_file_name}...")
        response = requests.get(file_url, timeout=30)
        response.raise_for_status()
        with open(ogg_path, "wb") as f:
            f.write(response.content)
        subprocess.run(
            ["ffmpeg", "-y", "-i", ogg_path, "-ar", "22050", "-ac", "1", wav_path, "-loglevel", "error"],
            check=True, text=True
        )
        if status_queue:
            status_queue.put(f"Converted: {wav_file_name}")
        return wav_path
    except Exception as e:
        logging.error(f"Error processing {ogg_file_name}: {e}")
        if status_queue:
            status_queue.put(f"Error with {ogg_file_name}")
        return None
    finally:
        if os.path.exists(ogg_path):
            os.remove(ogg_path)

def fetch_character_list_from_api():
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

def is_silent_audio(file_path, silence_threshold=-50.0):
    try:
        audio = AudioSegment.from_wav(file_path)
        return audio.max_dBFS < silence_threshold
    except Exception as e:
        logging.error(f"Error checking silence for {file_path}: {e}")
        return True

def clean_metadata_file(metadata_path):
    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        if not lines:
            logging.error(f"Metadata file {metadata_path} is empty.")
            return False

        cleaned_lines = []
        invalid_lines = []
        for i, line in enumerate(lines, 1):
            fields = line.strip().split("|")
            if i == 1:  # Header
                if fields != ["audio_file", "text", "normalized_text", "speaker_id"]:
                    logging.error(f"Invalid header in {metadata_path}: {line.strip()}")
                    return False
                cleaned_lines.append(line)
                continue
            if len(fields) != 4 or not fields[0] or not fields[1] or not fields[2]:
                invalid_lines.append((i, line.strip()))
                continue
            cleaned_lines.append(line)

        if invalid_lines:
            logging.warning(f"Removed {len(invalid_lines)} invalid metadata lines: {[f'Line {i}: {line}' for i, line in invalid_lines]}")

        with open(metadata_path, "w", encoding="utf-8", newline="") as f:
            f.writelines(cleaned_lines)
        logging.info(f"Cleaned {metadata_path}: {len(cleaned_lines)-1} valid entries.")
        return len(cleaned_lines) > 1
    except Exception as e:
        logging.error(f"Error cleaning {metadata_path}: {e}")
        return False

def transcribe_character_audio(
    character_output_dir,
    whisper_model="base",
    use_segmentation=False,
    hf_token="",
    strict_ascii=False,
    status_queue=None,
):
    if SpeechToText is None:
        logging.error("Transcription unavailable: SpeechToText not imported.")
        if status_queue:
            status_queue.put("Transcription unavailable.")
        return False

    metadata_path = os.path.join(character_output_dir, "metadata.csv")
    wavs_dir = os.path.join(character_output_dir, "wavs")
    silent_dir = os.path.join(character_output_dir, "silent_files")
    os.makedirs(wavs_dir, exist_ok=True)
    os.makedirs(silent_dir, exist_ok=True)

    existing_files = set()
    if os.path.exists(metadata_path) and validate_metadata_layout(metadata_path):
        try:
            with open(metadata_path, "r", encoding="utf-8") as mf:
                lines = mf.readlines()
                for line in lines[1:]:
                    parts = line.strip().split("|")
                    if len(parts) >= 1:
                        existing_files.add(parts[0])
        except UnicodeDecodeError:
            logging.error(f"Encoding error reading {metadata_path}. Ensure UTF-8 format.")
            return False
    else:
        logging.info(f"Invalid or missing metadata at {metadata_path}. Starting fresh.")
        existing_files = set()

    files_to_transcribe = []
    if use_segmentation and hf_token:
        for file in os.listdir(character_output_dir):
            if file.lower().endswith(".wav") and file not in os.listdir(wavs_dir):
                wav_path = os.path.join(character_output_dir, file)
                segmented_files = segment_audio_file(
                    wav_path,
                    wavs_dir,
                    onset=0.6,
                    offset=0.4,
                    min_duration=2.0,
                    hf_token=hf_token,
                )
                if segmented_files:
                    files_to_transcribe.extend([os.path.basename(f) for f in segmented_files])
                    os.remove(wav_path)
                else:
                    files_to_transcribe.append(file)
    else:
        for file in os.listdir(wavs_dir):
            if file.lower().endswith(".wav") and f"wavs/{file}" not in existing_files:
                files_to_transcribe.append(file)
        for file in os.listdir(character_output_dir):
            if file.lower().endswith(".wav") and file not in os.listdir(wavs_dir):
                files_to_transcribe.append(file)

    if not files_to_transcribe:
        logging.info("No new WAV files to transcribe.")
        if status_queue:
            status_queue.put("No new files to transcribe.")
        if os.path.exists(metadata_path):
            return clean_metadata_file(metadata_path)
        return False

    logging.info(f"Processing {len(files_to_transcribe)} WAV files with Whisper {whisper_model}...")
    transcribed_count = 0
    failed_count = 0
    silent_count = 0
    file_mode = "a" if existing_files else "w"

    with open(metadata_path, file_mode, encoding="utf-8", newline="") as mf:
        if file_mode == "w":
            mf.write("audio_file|text|normalized_text|speaker_id\n")

        for file in tqdm(files_to_transcribe, desc="Transcribing"):
            src_path = os.path.join(character_output_dir, file)
            wav_path = os.path.join(wavs_dir, file)
            if os.path.exists(src_path) and not os.path.exists(wav_path):
                shutil.move(src_path, wav_path)

            if status_queue:
                status_queue.put(f"Checking: {file}")

            if is_silent_audio(wav_path):
                logging.warning(f"Moving silent file {file} to silent_files")
                shutil.move(wav_path, os.path.join(silent_dir, file))
                silent_count += 1
                if status_queue:
                    status_queue.put(f"Moved silent file: {file}")
                continue

            for attempt in range(2):
                try:
                    if status_queue:
                        status_queue.put(f"Transcribing: {file} (Attempt {attempt+1})")
                    stt = SpeechToText(
                        use_microphone=False,
                        audio_file=wav_path,
                        engine="whisper",
                        whisper_model_size=whisper_model,
                    )
                    audio_transcript = stt.process_audio(language="en")
                    cleaned_transcript = clean_transcript(audio_transcript, strict_ascii=strict_ascii)
                    if not cleaned_transcript or not is_valid_for_phonemes(cleaned_transcript):
                        logging.warning(f"Invalid or empty transcription for {file}")
                        if attempt == 1:
                            logging.warning(f"Moving failed file {file} to silent_files")
                            shutil.move(wav_path, os.path.join(silent_dir, file))
                            failed_count += 1
                            if status_queue:
                                status_queue.put(f"Moved failed file: {file}")
                        continue
                    normalized = (
                        clean_transcript(cleaned_transcript, strict_ascii=True)
                        .lower()
                        .replace(".", "")
                        .replace(",", "")
                    )
                    if not normalized.strip():
                        logging.warning(f"Empty normalized transcription for {file}")
                        if attempt == 1:
                            logging.warning(f"Moving failed file {file} to silent_files")
                            shutil.move(wav_path, os.path.join(silent_dir, file))
                            failed_count += 1
                            if status_queue:
                                status_queue.put(f"Moved failed file: {file}")
                        continue
                    metadata_entry = (
                        f"{cleaned_transcript}|wavs/{file}|speaker_1"
                    )
                    if metadata_entry.count("|") != 3:
                        logging.warning(f"Invalid metadata entry for {file}: {metadata_entry}")
                        if attempt == 1:
                            failed_count += 1
                            if status_queue:
                                status_queue.put(f"Invalid entry for {file}")
                        continue
                    mf.write(metadata_entry + "\n")
                    transcribed_count += 1
                    if status_queue:
                        status_queue.put(f"Transcribed: {file}")
                    break
                except Exception as e:
                    logging.error(f"Transcription error for {file}: {e}")
                    if attempt == 1:
                        logging.warning(f"Moving errored file {file} to silent_files")
                        shutil.move(wav_path, os.path.join(silent_dir, file))
                        failed_count += 1
                        if status_queue:
                            status_queue.put(f"Error transcribing {file}: Moved to silent_files")
                    continue

    logging.info(
        f"Transcription complete. Successful: {transcribed_count}, Failed: {failed_count}, Silent: {silent_count}."
    )
    if status_queue:
        status_queue.put(
            f"Transcription complete. Successful: {transcribed_count}, Failed: {failed_count}, Silent: {silent_count}."
        )

    success = clean_metadata_file(metadata_path)
    if success:
        split_metadata(metadata_path, valid_ratio=0.2)
    return success

def validate_metadata_existence(character_dir):
    metadata_path = os.path.join(character_dir, "metadata.csv")
    if not os.path.exists(metadata_path):
        logging.error(f"Metadata file missing: {metadata_path}")
        return False
    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        if len(lines) < 2:
            logging.error(f"Metadata file {metadata_path} has no data entries.")
            return False
        return True
    except Exception as e:
        logging.error(f"Error checking metadata {metadata_path}: {e}")
        return False

def validate_metadata_for_training(metadata_path):
    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        if len(lines) < 2:
            logging.error(f"Metadata file {metadata_path} has no data entries.")
            return False
        invalid_lines = []
        for i, line in enumerate(lines[1:], 2):
            fields = line.strip().split("|")
            if len(fields) != 3 or not all(fields):
                invalid_lines.append((i, line.strip()))
        if invalid_lines:
            logging.error(f"Invalid metadata entries in {metadata_path}: {[f'Line {i}: {line}' for i, line in invalid_lines]}")
            return False
        return True
    except Exception as e:
        logging.error(f"Error validating metadata for training {metadata_path}: {e}")
        return False

def process_character_voices(
    character,
    language,
    base_output_dir,
    download_wiki_audio=True,
    whisper_model="base",
    use_segmentation=False,
    hf_token="",
    strict_ascii=False,
    status_queue=None,
    batch_size=16,
    num_epochs=100,
    learning_rate=0.0001,
    stop_event=None,
):
    safe_character_name = re.sub(r'[\\/*?:"<>|]', "_", character)
    character_folder = os.path.join(base_output_dir, safe_character_name)
    wavs_folder = os.path.join(character_folder, "wavs")
    os.makedirs(wavs_folder, exist_ok=True)

    if not download_wiki_audio:
        logging.info("Skipping Wiki download.")
        success = transcribe_character_audio(
            character_folder,
            whisper_model,
            use_segmentation,
            hf_token,
            strict_ascii,
            status_queue=status_queue,
        )
        if not success:
            logging.error(f"Transcription failed for {character}.")
            return None
        metadata_path = os.path.join(character_folder, "metadata.csv")
        if not generate_valid_csv(metadata_path):
            logging.error(f"Failed to generate valid.csv for {character}.")
            return None
        return character_folder

    categories = (
        [
            f"Category:{character} Voice-Overs",
            f"Category:English {character} Voice-Overs",
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
        if stop_event and stop_event.is_set():
            status_queue.put("Processing cancelled.")
            return None
        file_name = re.match(r"File:(.*)", file_title).group(1).strip()
        file_url = get_file_url(file_title)
        if file_url:
            wav_file_path = download_and_convert(
                file_url, character_folder, file_name, status_queue=status_queue
            )
            if wav_file_path:
                downloaded_count += 1
            else:
                failed_count += 1
        else:
            failed_count += 1

    logging.info(
        f"Download complete. Downloaded: {downloaded_count}, Failed: {failed_count}."
    )
    status_queue.put(
        f"Download complete. Downloaded: {downloaded_count}, Failed: {failed_count}."
    )

    for file in os.listdir(character_folder):
        if file.lower().endswith(".wav") and file not in os.listdir(wavs_folder):
            shutil.move(
                os.path.join(character_folder, file), os.path.join(wavs_folder, file)
            )

    success = transcribe_character_audio(
        character_folder,
        whisper_model,
        use_segmentation,
        hf_token,
        strict_ascii,
        status_queue=status_queue,
    )
    if not success:
        logging.error(f"Transcription failed for {character}.")
        return None

    metadata_path = os.path.join(character_folder, "metadata.csv")
    if not generate_valid_csv(metadata_path):
        logging.error(f"Failed to generate valid.csv for {character}.")
        return None
    return character_folder

def validate_metadata_layout(metadata_path):
    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        if len(lines) < 2:
            logging.error(f"Metadata file {metadata_path} is empty or has no data entries.")
            return False
        header = lines[0].strip()
        expected_header = "audio_file|text|normalized_text|speaker_id"
        if header != expected_header:
            logging.error(f"Invalid header in {metadata_path}: {header}")
            return False
        invalid_lines = []
        for i, line in enumerate(lines[1:], 2):
            fields = line.strip().split("|")
            if len(fields) != 4 or not all(fields[:3]):
                invalid_lines.append((i, line.strip()))
        if invalid_lines:
            logging.warning(
                f"Found {len(invalid_lines)} invalid lines in {metadata_path}: {[f'Line {i}: {line}' for i, line in invalid_lines]}"
            )
        return True
    except UnicodeDecodeError:
        logging.error(f"Encoding error in {metadata_path}. Ensure UTF-8 format.")
        return False
    except Exception as e:
        logging.error(f"Error validating {metadata_path}: {e}")
        return False

def validate_training_prerequisites(character_dir, config_path):
    metadata_path = os.path.join(character_dir, "metadata.csv")
    valid_metadata_path = os.path.join(character_dir, "valid.csv")
    wavs_dir = os.path.join(character_dir, "wavs")

    if not os.path.exists(config_path):
        logging.error(f"Configuration file missing: {config_path}")
        return False

    if not validate_metadata_existence(character_dir):
        return False
    if not os.path.exists(valid_metadata_path):
        logging.error(f"Validation metadata file missing: {valid_metadata_path}")
        return False

    if not validate_metadata_for_training(metadata_path):
        return False
    if not validate_metadata_for_training(valid_metadata_path):
        logging.error(f"Invalid metadata for training: {metadata_path}")
        return False

    if not os.path.exists(wavs_dir) or not os.listdir(wavs_dir):
        logging.error(f"No WAV files found in {wavs_dir}")
        return False

    try:
        df = pd.read_csv(metadata_path, sep="|", encoding="utf-8")
        for audio_file in df["audio_file"]:
            wav_path = os.path.join(character_dir, audio_file)
            if not os.path.exists(wav_path):
                logging.error(f"WAV file referenced in metadata does not exist: {wav_path}")
                return False
    except Exception as e:
        logging.error(f"Error validating WAV file references in metadata: {e}")
        return False

    return True

def update_character_config(
    character,
    base_output_dir,
    selected_model="Fast Tacotron2",
    batch_size=16,
    num_epochs=100,
    learning_rate=0.0001
):
    character_folder = os.path.join(base_output_dir, character)
    wavs_folder = os.path.join(character_folder, "wavs")
    metadata_path = os.path.join(character_folder, "metadata.csv")
    valid_metadata_path = os.path.join(character_folder, "valid.csv")
    config_path = os.path.join(character_folder, f"{character}_config.json")

    config = {
        "output_path": os.path.join(base_output_dir, "tts_train_output", character),
        "datasets": [
            {
                "name": character,
                "path": wavs_folder,
                "meta_file_train": metadata_path,
                "meta_file_val": valid_metadata_path
            }
        ],
        "audio": {
            "sample_rate": 22050,
            "fft_size": 1024,
            "win_length": 1024,
            "hop_length": 256,
            "num_mels": 80,
            "mel_fmin": 0.0,
            "mel_fmax": 8000.0
        },
        "model": AVAILABLE_MODELS[selected_model]["model_id"],
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "learning_rate": learning_rate,
        "lr_schedule": "noam",
        "use_precomputed_alignments": False,
        "run_eval": True,
        "characters": {
            "use_phonemes": False,
            "pad": "<PAD>",
            "eos": "<EOS>",
            "bos": "<BOS>",
            "blank": "<BLNK>",
            "characters": "abcdefghijklmnopqrstuvwxyz,.!? ",
            "punctuations": ",.!? "
        },
        "test_size": 0.2,
        "mixed_precision": False,
        "checkpointing": True,
        "checkpoint_interval": 10
    }

    try:
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=4)
        logging.info(f"Config file updated: {config_path}")
        return config_path
    except Exception as e:
        logging.error(f"Error updating config for {character}: {e}")
        return None

def generate_valid_csv(metadata_path, valid_ratio=0.2):
    try:
        df = pd.read_csv(metadata_path, sep="|", encoding="utf-8")
        n_valid = int(len(df) * valid_ratio)
        indices = np.random.permutation(len(df))
        valid_df = df.iloc[indices[:n_valid]]
        valid_path = os.path.join(os.path.dirname(metadata_path), "valid.csv")
        valid_df.to_csv(valid_path, sep="|", index=False, encoding="utf-8")
        logging.info(f"Valid CSV generated: {valid_path}")
        return valid_path
    except Exception as e:
        logging.error(f"Error generating valid.csv from {metadata_path}: {e}")
        return None

# --- Enhanced Training Functions ---

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
    batch_size=16,
    num_epochs=100,
    learning_rate=0.0001
):
    if selected_model not in AVAILABLE_MODELS:
        logging.error(f"Invalid model selected: {selected_model}")
        return None

    model_data = AVAILABLE_MODELS[selected_model]
    config = {
        "output_path": os.path.join(character_dir, "tts_output"),
        "datasets": [
            {
                "name": character,
                "path": os.path.join(character_dir, "wavs"),
                "meta_file_train": os.path.join(character_dir, "metadata.csv"),
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
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "learning_rate": learning_rate,
        "lr_schedule": "noam",
        "use_precomputed_alignments": False,
        "run_eval": True,
        "characters": {
            "use_phonemes": False,
            "pad": "<PAD>",
            "eos": "<EOS>",
            "bos": "<BOS>",
            "blank": "<BLNK>",
            "characters": "abcdefghijklmnopqrstuvwxyz,.!? ",
            "punctuations": ",.!? ",
        },
        "test_size": 0.2,
        "mixed_precision": False,
        "checkpointing": True,
        "checkpoint_interval": 10
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

def start_tts_training(config_path, resume_from_checkpoint=None, status_queue=None, stop_event=None):
    if not os.path.exists(config_path):
        logging.error(f"Configuration file not found: {config_path}")
        if status_queue:
            status_queue.put("Configuration file missing.")
        return False

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        metadata_path = config["datasets"][0]["meta_file_train"]
        valid_metadata_path = config["datasets"][0]["meta_file_val"]
        if not validate_metadata_for_training(metadata_path):
            logging.error(f"Invalid metadata for training: {metadata_path}")
            return False
        if not validate_metadata_for_training(valid_metadata_path):
            logging.error(f"Invalid validation metadata: {valid_metadata_path}")
            return False

        if not os.path.exists(config["output_path"]):
            os.makedirs(config["output_path"], exist_ok=True)
        else:
            backup_file(config["output_path"], "tts_output_backup")
            shutil.rmtree(config["output_path"])
            os.makedirs(config["output_path"], exist_ok=True)

        tts = TTS(model_name=config["model"].split("/")[-1] if config["model"].startswith("tts_models") else config["model"], progress_bar=True)
        character_dir = os.path.dirname(config_path)
        if not validate_training_prerequisites(character_dir, config_path):
            logging.error("Training prerequisites validation failed.")
            if status_queue:
                status_queue.put("Training prerequisites validation failed.")
            return False

        def training_callback(epoch, loss, metrics):
            if stop_event and stop_event.is_set():
                raise KeyboardInterrupt("Training cancelled.")
            log_message = f"Epoch {epoch}: Loss = {loss:.4f}, Metrics = {metrics}"
            logging.info(log_message)
            if status_queue:
                status_queue.put(log_message)

        training_args = {
            "config_path": config_path,
            "output_path": config["output_path"],
            "num_epochs": config["num_epochs"],
            "batch_size": config["batch_size"],
            "eval_split": config.get("test_size", 0.2),
            "run_eval": config["run_eval"],
            "progress_callback": training_callback,
            "restore_path": resume_from_checkpoint if resume_from_checkpoint else config.get("restore_path", None),
            "checkpointing": config.get("checkpointing", True),
            "checkpoint_interval": config.get("checkpoint_interval", 10)
        }

        logging.info(f"Starting TTS training with config: {config_path}")
        if resume_from_checkpoint:
            logging.info(f"Resuming from checkpoint: {resume_from_checkpoint}")
        if status_queue:
            status_queue.put(f"Starting TTS training{' (resuming)' if resume_from_checkpoint else ''}...")

        if os.path.exists(config["output_path"]):
            backup_file(config["output_path"], "tts_output_backup")

        tts.train(**training_args)

        logging.info(f"Training completed successfully: {config_path}")
        if status_queue:
            status_queue.put("Training completed successfully.")
        return True

    except KeyboardInterrupt:
        logging.info("Training interrupted by user.")
        if status_queue:
            status_queue.put("Training cancelled.")
        return False
    except ImportError as e:
        logging.error(f"Coqui TTS library not installed or incompatible: {e}")
        if status_queue:
            status_queue.put("Coqui TTS library not installed.")
        return False
    except Exception as e:
        logging.error(f"Error during TTS training: {e}")
        if status_queue:
            status_queue.put(f"Training failed: {str(e)}")
        return False

def test_trained_model(config_path, test_text="Hello, this is a test of the trained model!", output_wav="test_output.wav", status_queue=None):
    if not os.path.exists(config_path):
        logging.error(f"Configuration file not found: {config_path}")
        if status_queue:
            status_queue.put("Configuration file missing.")
        return False

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        output_path = config["output_path"]
        checkpoint_dir = os.path.join(output_path, "checkpoints")
        if not os.path.exists(checkpoint_dir):
            logging.error(f"No checkpoints found in {checkpoint_dir}")
            if status_queue:
                status_queue.put("No trained model checkpoints found.")
            return False

        latest_checkpoint = max(
            [os.path.join(checkpoint_dir, f) for f in os.listdir(checkpoint_dir) if f.startswith("checkpoint_")],
            key=os.path.getmtime,
            default=None
        )
        if not latest_checkpoint:
            logging.error("No valid checkpoint found.")
            if status_queue:
                status_queue.put("No valid checkpoint found.")
            return False

        tts = TTS(model_path=latest_checkpoint, config_path=config_path, progress_bar=False)
        output_wav_path = os.path.join(output_path, output_wav)
        if status_queue:
            status_queue.put(f"Synthesizing test audio: {output_wav}")

        tts.tts_to_file(text=test_text, file_path=output_wav_path)
        logging.info(f"Test audio generated: {output_wav_path}")
        if status_queue:
            status_queue.put(f"Test audio saved: {output_wav_path}")
        return output_wav_path

    except Exception as e:
        logging.error(f"Error testing trained model: {e}")
        if status_queue:
            status_queue.put(f"Test failed: {str(e)}")
        return False

def find_latest_checkpoint(output_path):
    checkpoint_dir = os.path.join(output_path, "checkpoints")
    if not os.path.exists(checkpoint_dir):
        return None
    checkpoints = [os.path.join(checkpoint_dir, f) for f in os.listdir(checkpoint_dir) if f.startswith("checkpoint_")]
    if not checkpoints:
        return None
    return max(checkpoints, key=os.path.getmtime)

def backup_file(file_path, suffix):
    if os.path.exists(file_path):
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        backup_path = f"{file_path}.{suffix}.{timestamp}"
        if os.path.isfile(file_path):
            shutil.copy(file_path, backup_path)
        else:
            shutil.copytree(file_path, backup_path, dirs_exist_ok=True)
        logging.info(f"Backup created: {backup_path}")

def has_trained_model(character, base_output_dir):
    output_path = os.path.join(base_output_dir, "tts_train_output", character)
    checkpoint_dir = os.path.join(output_path, "checkpoints")
    config_path = os.path.join(base_output_dir, character, f"{character}_config.json")
    if os.path.exists(checkpoint_dir) and os.path.exists(config_path):
        checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith("checkpoint_")]
        return len(checkpoints) > 0
    return False

def speak_tts(config_path, text, character, status_queue=None, stop_event=None):
    if not text.strip():
        if status_queue:
            status_queue.put("Please enter text to synthesize.")
        return False

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        output_path = config["output_path"]
        checkpoint_dir = os.path.join(output_path, "checkpoints")
        if not os.path.exists(checkpoint_dir):
            logging.error(f"No checkpoints found in {checkpoint_dir}")
            if status_queue:
                status_queue.put("No trained model checkpoints found.")
            return False

        latest_checkpoint = max(
            [os.path.join(checkpoint_dir, f) for f in os.listdir(checkpoint_dir) if f.startswith("checkpoint_")],
            key=os.path.getmtime,
            default=None
        )
        if not latest_checkpoint:
            logging.error("No valid checkpoint found.")
            if status_queue:
                status_queue.put("No valid checkpoint found.")
            return False

        if stop_event and stop_event.is_set():
            status_queue.put("TTS synthesis cancelled.")
            return False

        tts = TTS(model_path=latest_checkpoint, config_path=config_path, progress_bar=False)
        temp_wav = os.path.join(output_path, "temp_tts_output.wav")
        tts_output_dir = os.path.join(output_path, "tts_outputs")
        os.makedirs(tts_output_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        mp3_filename = f"{character}_tts_{timestamp}.mp3"
        mp3_path = os.path.join(tts_output_dir, mp3_filename)

        if status_queue:
            status_queue.put(f"Synthesizing audio: {mp3_filename}")

        tts.tts_to_file(text=text, file_path=temp_wav)
        if stop_event and stop_event.is_set():
            status_queue.put("TTS synthesis cancelled.")
            if os.path.exists(temp_wav):
                os.remove(temp_wav)
            return False

        audio = AudioSegment.from_wav(temp_wav)
        audio.export(mp3_path, format="mp3")
        os.remove(temp_wav)

        if status_queue:
            status_queue.put(f"Saved MP3: {mp3_filename}")

        if status_queue:
            status_queue.put("Playing audio...")
        playsound.playsound(mp3_path)
        logging.info(f"TTS audio saved and played: {mp3_path}")
        if status_queue:
            status_queue.put(f"Audio played: {mp3_filename}")
        return mp3_path

    except Exception as e:
        logging.error(f"Error in TTS synthesis: {e}")
        if status_queue:
            status_queue.put(f"TTS failed: {str(e)}")
        return False

# --- GUI ---

window = None
tts_frame = None

def main_gui():
    global window, tts_frame
    window = tk.Tk()
    window.title("Genshin Impact Voice Downloader & Transcriber")

    config_frame = ttk.LabelFrame(window, text="Configuration")
    config_frame.grid(row=0, column=0, padx=10, pady=10, sticky="ew")

    ttk.Label(config_frame, text="Language:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
    languages = ["English", "Japanese", "Chinese", "Korean"]
    language_var = tk.StringVar(value="English")
    ttk.Combobox(config_frame, textvariable=language_var, values=languages, state="readonly").grid(
        row=0, column=1, padx=5, pady=5, sticky="w"
    )

    ttk.Label(config_frame, text="Character:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
    character_names = fetch_character_list_from_api()
    character_var = tk.StringVar(value=character_names[0] if character_names else "No characters found")
    character_dropdown = ttk.Combobox(
        config_frame, textvariable=character_var, values=character_names, state="readonly", width=30
    )
    character_dropdown.grid(row=1, column=1, columnspan=2, padx=5, pady=5, sticky="ew")

    ttk.Label(config_frame, text="Output Dir:").grid(row=2, column=0, padx=5, pady=5, sticky="w")
    output_dir_var = tk.StringVar(value=BASE_DATA_DIR)
    ttk.Entry(config_frame, textvariable=output_dir_var, width=40).grid(
        row=2, column=1, columnspan=2, padx=5, pady=5, sticky="ew"
    )

    ttk.Label(config_frame, text="Whisper Model:").grid(row=3, column=0, padx=5, pady=5, sticky="w")
    whisper_models = ["base", "large-v2"]
    whisper_model_var = tk.StringVar(value="base")
    whisper_model_combo = ttk.Combobox(
        config_frame, textvariable=whisper_model_var, values=whisper_models, state="readonly"
    )
    whisper_model_combo.grid(row=3, column=1, padx=5, pady=5, sticky="w")

    ttk.Label(config_frame, text="Batch Size:").grid(row=4, column=0, padx=5, pady=5, sticky="w")
    batch_size_var = tk.StringVar(value="16")
    ttk.Entry(config_frame, textvariable=batch_size_var, width=10).grid(row=4, column=1, padx=5, pady=5, sticky="w")

    ttk.Label(config_frame, text="Num Epochs:").grid(row=5, column=0, padx=5, pady=5, sticky="w")
    num_epochs_var = tk.StringVar(value="100")
    ttk.Entry(config_frame, textvariable=num_epochs_var, width=10).grid(row=5, column=1, padx=5, pady=5, sticky="w")

    ttk.Label(config_frame, text="Learning Rate:").grid(row=6, column=0, padx=5, pady=5, sticky="w")
    learning_rate_var = tk.StringVar(value="0.0001")
    ttk.Entry(config_frame, textvariable=learning_rate_var, width=10).grid(row=6, column=1, padx=5, pady=5, sticky="w")

    use_segmentation_var = tk.BooleanVar(value=False)
    ttk.Checkbutton(
        config_frame, text="Use Audio Segmentation", variable=use_segmentation_var
    ).grid(row=7, column=1, padx=5, pady=5, sticky="w")

    strict_ascii_var = tk.BooleanVar(value=False)
    ttk.Checkbutton(
        config_frame, text="Strict ASCII Transcriptions", variable=strict_ascii_var
    ).grid(row=8, column=1, padx=5, pady=5, sticky="w")

    ttk.Label(config_frame, text="HF Token:").grid(row=9, column=0, padx=5, pady=5, sticky="w")
    hf_token_var = tk.StringVar()
    hf_token_entry = ttk.Entry(config_frame, textvariable=hf_token_var, width=40, show="*")
    hf_token_entry.grid(row=9, column=1, columnspan=2, padx=5, pady=5, sticky="ew")

    download_wiki_audio_var = tk.BooleanVar(value=True)
    ttk.Checkbutton(
        config_frame, text="Download Wiki Audio", variable=download_wiki_audio_var
    ).grid(row=10, column=1, padx=5, pady=5, sticky="w")

    ttk.Label(config_frame, text="TTS Model:").grid(row=11, column=0, padx=5, pady=5, sticky="w")
    tts_model_var = tk.StringVar(value="Fast Tacotron2")
    tts_model_combo = ttk.Combobox(
        config_frame, textvariable=tts_model_var, values=list(AVAILABLE_MODELS.keys()), state="readonly"
    )
    tts_model_combo.grid(row=11, column=1, padx=5, pady=5, sticky="w")

    tts_frame = ttk.LabelFrame(window, text="TTS Feedback")
    tts_frame.grid(row=1, column=0, padx=10, pady=10, sticky="ew")
    tts_frame.grid_remove()

    ttk.Label(tts_frame, text="TTS Text:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
    tts_text_var = tk.StringVar()
    tts_text_entry = ttk.Entry(tts_frame, textvariable=tts_text_var, width=40)
    tts_text_entry.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

    speak_button = ttk.Button(tts_frame, text="Speak", state="disabled")
    speak_button.grid(row=0, column=2, padx=5, pady=5)

    control_frame = ttk.Frame(window)
    control_frame.grid(row=2, column=0, padx=10, pady=5, sticky="ew")

    status_label = ttk.Label(control_frame, text="Ready", relief=tk.SUNKEN, anchor="w", width=60)
    status_label.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5, pady=5)

    status_queue = queue.Queue()
    stop_event = threading.Event()
    current_thread = [None]

    def update_hf_token_state(*args):
        if whisper_model_var.get() == "base" and not use_segmentation_var.get():
            hf_token_entry.config(state="disabled")
        else:
            hf_token_entry.config(state="normal")

    whisper_model_var.trace_add("write", update_hf_token_state)
    use_segmentation_var.trace_add("write", update_hf_token_state)
    update_hf_token_state()

    def update_tts_frame(*args):
        character = character_var.get()
        base_output_dir = output_dir_var.get()
        if character and character != "No characters found" and base_output_dir and has_trained_model(character, base_output_dir):
            tts_frame.grid()
            speak_button.config(state="normal")
        else:
            tts_frame.grid_remove()
            speak_button.config(state="disabled")

    character_var.trace_add("write", update_tts_frame)
    output_dir_var.trace_add("write", update_tts_frame)

    def update_status():
        try:
            while True:
                message = status_queue.get_nowait()
                status_label.config(text=message)
                window.update_idletasks()
                if message == "enable_buttons":
                    download_button.config(state="normal")
                    train_button.config(state="normal")
                    test_button.config(state="normal")
                    cancel_button.config(state="disabled")
                    speak_button.config(state="normal" if has_trained_model(character_var.get(), output_dir_var.get()) else "disabled")
        except queue.Empty:
            window.after(100, update_status)

    def cancel_operation():
        if current_thread[0] and current_thread[0].is_alive():
            stop_event.set()
            cancel_button.config(state="disabled")
            status_label.config(text="Cancelling... Please wait.")
            window.update_idletasks()

    def start_processing():
        if current_thread[0] and current_thread[0].is_alive():
            messagebox.showwarning("Warning", "Another operation is running. Please cancel or wait.")
            return

        character = character_var.get()
        language = language_var.get()
        base_output_dir = output_dir_var.get()
        whisper_model = whisper_model_var.get()
        use_segmentation = use_segmentation_var.get()
        strict_ascii = strict_ascii_var.get()
        hf_token = hf_token_var.get()
        download_wiki_audio = download_wiki_audio_var.get()
        selected_tts_model = tts_model_var.get()

        try:
            batch_size = int(batch_size_var.get())
            if batch_size <= 0:
                raise ValueError("Batch size must be positive.")
        except ValueError:
            messagebox.showerror("Error", "Invalid batch size.")
            return

        try:
            num_epochs = int(num_epochs_var.get())
            if num_epochs <= 0:
                raise ValueError("Number of epochs must be positive.")
        except ValueError:
            messagebox.showerror("Error", "Invalid number of epochs.")
            return

        try:
            learning_rate = float(learning_rate_var.get())
            if learning_rate <= 0:
                raise ValueError("Learning rate must be positive.")
        except ValueError:
            messagebox.showerror("Error", "Invalid learning rate.")
            return

        if not character or character == "No characters found":
            messagebox.showerror("Error", "Select a valid character.")
            return
        if not base_output_dir:
            messagebox.showerror("Error", "Enter an output directory.")
            return
        if use_segmentation and not hf_token:
            messagebox.showerror("Error", "Hugging Face token required for segmentation.")
            return

        stop_event.clear()
        download_button.config(state="disabled")
        train_button.config(state="disabled")
        test_button.config(state="disabled")
        cancel_button.config(state="normal")
        speak_button.config(state="disabled")
        status_label.config(text=f"Processing {character}...")

        def process_task():
            try:
                character_folder_path = process_character_voices(
                    character, language, base_output_dir, download_wiki_audio,
                    whisper_model, use_segmentation, hf_token, strict_ascii, status_queue,
                    batch_size, num_epochs, learning_rate, stop_event
                )
                if stop_event.is_set():
                    status_queue.put("Processing cancelled.")
                    return

                if character_folder_path:
                    if validate_metadata_existence(character_folder_path):
                        metadata_path = os.path.join(character_folder_path, "metadata.csv")
                        if validate_metadata_layout(metadata_path):
                            config_path = update_character_config(
                                character, base_output_dir, selected_model=selected_tts_model,
                                batch_size=batch_size, num_epochs=num_epochs, learning_rate=learning_rate
                            )
                            if config_path:
                                if start_tts_training(config_path, status_queue=status_queue, stop_event=stop_event):
                                    if not stop_event.is_set():
                                        test_trained_model(config_path, status_queue=status_queue)
                                else:
                                    status_queue.put(f"Training failed: {character}")
                            else:
                                status_queue.put(f"Config generation failed: {character}")
                        else:
                            status_queue.put("Invalid metadata layout.")
                    else:
                        status_queue.put("Metadata missing.")
                else:
                    status_queue.put("Processing failed.")
            except Exception as e:
                status_queue.put(f"Error: {str(e)}")
            finally:
                status_queue.put("enable_buttons")

        current_thread[0] = threading.Thread(target=process_task)
        current_thread[0].start()

    def start_training():
        if current_thread[0] and current_thread[0].is_alive():
            messagebox.showwarning("Warning", "Another operation is running. Please cancel or wait.")
            return

        character = character_var.get()
        base_output_dir = output_dir_var.get()
        selected_tts_model = tts_model_var.get()

        try:
            batch_size = int(batch_size_var.get())
            if batch_size <= 0:
                raise ValueError("Batch size must be positive.")
        except ValueError:
            messagebox.showerror("Error", "Invalid batch size.")
            return

        try:
            num_epochs = int(num_epochs_var.get())
            if num_epochs <= 0:
                raise ValueError("Number of epochs must be positive.")
        except ValueError:
            messagebox.showerror("Error", "Invalid number of epochs.")
            return

        try:
            learning_rate = float(learning_rate_var.get())
            if learning_rate <= 0:
                raise ValueError("Learning rate must be positive.")
        except ValueError:
            messagebox.showerror("Error", "Invalid learning rate.")
            return

        if not character or character == "No characters found":
            messagebox.showerror("Error", "Select a valid character.")
            return
        if not base_output_dir:
            messagebox.showerror("Error", "Enter an output directory.")
            return

        stop_event.clear()
        download_button.config(state="disabled")
        train_button.config(state="disabled")
        test_button.config(state="disabled")
        cancel_button.config(state="normal")
        speak_button.config(state="disabled")

        character_folder = os.path.join(base_output_dir, character)
        metadata_path = os.path.join(character_folder, "metadata.csv")
        valid_metadata_path = os.path.join(character_folder, "valid.csv")
        config_path = os.path.join(character_folder, f"{character}_config.json")

        if not os.path.exists(config_path):
            logging.warning(f"Config file not found: {config_path}. Recreating...")
            if os.path.exists(metadata_path):
                backup_file(metadata_path, "metadata_backup")
            if os.path.exists(valid_metadata_path):
                backup_file(valid_metadata_path, "valid_backup")
            generate_valid_csv(metadata_path)
            update_character_config(
                character, base_output_dir, selected_model=selected_tts_model,
                batch_size=batch_size, num_epochs=num_epochs, learning_rate=learning_rate
            )

        if not validate_training_prerequisites(character_folder, config_path):
            messagebox.showerror("Error", "Training prerequisites validation failed.")
            download_button.config(state="normal")
            train_button.config(state="normal")
            test_button.config(state="normal")
            cancel_button.config(state="disabled")
            speak_button.config(state="normal" if has_trained_model(character, base_output_dir) else "disabled")
            return

        current_whisper_model = whisper_model_var.get()
        if os.path.exists(config_path):
            backup_file(config_path, f"backup_{current_whisper_model}")

        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        checkpoint = find_latest_checkpoint(config["output_path"])

        def train_task():
            try:
                if start_tts_training(config_path, resume_from_checkpoint=checkpoint, status_queue=status_queue, stop_event=stop_event):
                    if not stop_event.is_set():
                        test_trained_model(config_path, status_queue=status_queue)
                else:
                    status_queue.put(f"Training failed for {character}.")
            except Exception as e:
                status_queue.put(f"Error: {str(e)}")
            finally:
                status_queue.put("enable_buttons")

        current_thread[0] = threading.Thread(target=train_task)
        current_thread[0].start()

    def test_model():
        if current_thread[0] and current_thread[0].is_alive():
            messagebox.showwarning("Warning", "Another operation is running. Please cancel or wait.")
            return

        character = character_var.get()
        base_output_dir = output_dir_var.get()
        if not character or character == "No characters found":
            messagebox.showerror("Error", "Select a valid character.")
            return
        if not base_output_dir:
            messagebox.showerror("Error", "Enter an output directory.")
            return

        character_folder = os.path.join(base_output_dir, character)
        config_path = os.path.join(character_folder, f"{character}_config.json")
        if not os.path.exists(config_path):
            messagebox.showerror("Error", f"Configuration file missing: {config_path}")
            return

        stop_event.clear()
        download_button.config(state="disabled")
        train_button.config(state="disabled")
        test_button.config(state="disabled")
        cancel_button.config(state="normal")
        speak_button.config(state="disabled")

        def test_task():
            try:
                test_trained_model(config_path, status_queue=status_queue)
            except Exception as e:
                status_queue.put(f"Error: {str(e)}")
            finally:
                status_queue.put("enable_buttons")

        current_thread[0] = threading.Thread(target=test_task)
        current_thread[0].start()

    def speak_model():
        if current_thread[0] and current_thread[0].is_alive():
            messagebox.showwarning("Warning", "Another operation is running. Please cancel or wait.")
            return

        character = character_var.get()
        base_output_dir = output_dir_var.get()
        text = tts_text_var.get()
        if not character or character == "No characters found":
            messagebox.showerror("Error", "Select a valid character.")
            return
        if not base_output_dir:
            messagebox.showerror("Error", "Enter an output directory.")
            return
        if not text.strip():
            messagebox.showerror("Error", "Enter text to synthesize.")
            return

        character_folder = os.path.join(base_output_dir, character)
        config_path = os.path.join(character_folder, f"{character}_config.json")
        if not os.path.exists(config_path):
            messagebox.showerror("Error", f"Configuration file missing: {config_path}")
            return

        stop_event.clear()
        download_button.config(state="disabled")
        train_button.config(state="disabled")
        test_button.config(state="disabled")
        cancel_button.config(state="normal")
        speak_button.config(state="disabled")

        def speak_task():
            try:
                speak_tts(config_path, text, character, status_queue=status_queue, stop_event=stop_event)
            except Exception as e:
                status_queue.put(f"Error: {str(e)}")
            finally:
                status_queue.put("enable_buttons")

        current_thread[0] = threading.Thread(target=speak_task)
        current_thread[0].start()

    download_button = ttk.Button(control_frame, text="Process Voices", command=start_processing)
    download_button.pack(side=tk.RIGHT, padx=5, pady=5)

    train_button = ttk.Button(control_frame, text="Start Training", command=start_training)
    train_button.pack(side=tk.RIGHT, padx=5, pady=5)

    test_button = ttk.Button(control_frame, text="Test Model", command=test_model)
    test_button.pack(side=tk.RIGHT, padx=5, pady=5)

    cancel_button = ttk.Button(control_frame, text="Cancel", command=cancel_operation, state="disabled")
    cancel_button.pack(side=tk.RIGHT, padx=5, pady=5)

    speak_button.configure(command=speak_model)

    window.after(100, update_status)
    window.grid_columnconfigure(0, weight=1)
    config_frame.grid_columnconfigure(1, weight=1)
    tts_frame.grid_columnconfigure(1, weight=1)
    window.mainloop()

# --- Main Execution ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download and transcribe Genshin Impact voice data."
    )
    parser.add_argument(
        "--character", type=str, help="Character name (e.g., Arlecchino)."
    )
    parser.add_argument(
        "--output_dir", type=str, default=BASE_DATA_DIR, help="Base output directory."
    )
    parser.add_argument(
        "--language",
        type=str,
        default="English",
        choices=["English", "Japanese", "Chinese", "Korean"],
    )
    parser.add_argument(
        "--whisper_model",
        type=str,
        default="base",
        choices=["base", "large-v2"],
        help="Whisper model size."
    )
    parser.add_argument(
        "--use_segmentation", action="store_true", help="Use PyAnnote segmentation."
    )
    parser.add_argument(
        "--strict_ascii", action="store_true", help="Force ASCII-only transcriptions."
    )
    parser.add_argument(
        "--hf_token", type=str, default="", help="Hugging Face token for segmentation."
    )
    parser.add_argument(
        "--skip_wiki_download", action="store_true", help="Skip Wiki download."
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Batch size for training."
    )
    parser.add_argument(
        "--num_epochs", type=int, default=100, help="Number of epochs for training."
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.0001,
        help="Learning rate for training."
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default="",
        help="Path to checkpoint to resume training."
    )
    parser.add_argument(
        "--test_model",
        action="store_true",
        help="Test the trained model after training."
    )
    args = parser.parse_args()

    if args.character is None:
        main_gui()
    else:
        logging.info(f"Processing {args.character} via CLI...")
        status_queue = queue.Queue()
        character_folder_path = process_character_voices(
            args.character,
            args.language,
            args.output_dir,
            not args.skip_wiki_download,
            args.whisper_model,
            args.use_segmentation,
            args.hf_token,
            args.strict_ascii,
            status_queue,
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
        )
        if not character_folder_path:
            logging.error(f"Processing failed for {args.character}. Exiting.")
            exit(1)
        if not validate_metadata_existence(character_folder_path):
            logging.error(
                f"No metadata found for {args.character}. Ensure WAV files are present and transcribed."
            )
            exit(1)
        if not validate_metadata_layout(
            os.path.join(character_folder_path, "metadata.csv")
        ):
            logging.error(
                f"Invalid metadata layout for {args.character}. Please re-transcribe."
            )
            exit(1)
        config_path = update_character_config(
            args.character,
            args.output_dir,
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
        )
        if not config_path:
            logging.error(f"Failed to generate config for {args.character}. Exiting.")
            exit(1)
        checkpoint = (
            args.resume_from_checkpoint
            if args.resume_from_checkpoint
            else find_latest_checkpoint(
                os.path.join(
                    args.output_dir, "tts_train_output", args.character
                )
            )
        )
        if start_tts_training(
            config_path,
            resume_from_checkpoint=checkpoint,
            status_queue=status_queue,
        ):
            logging.info(f"Training started for {args.character}.")
            if args.test_model:
                test_trained_model(config_path, status_queue=status_queue)
        else:
            logging.error(f"Training failed for {args.character}.")
            exit(1)
        logging.info(f"Finished processing {args.character}.")
