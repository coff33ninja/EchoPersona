import tkinter as tk
from tkinter import ttk, messagebox, filedialog
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
import threading
import queue
from gruut import sentences
import pygame
import platform

# --- SpeechToText Class ---
class SpeechToText:
    def __init__(
        self,
        use_microphone=False,
        audio_file=None,
        engine="whisper",
        whisper_model_size="base",
    ):
        try:
            import whisper
            self.model = whisper.load_model(whisper_model_size)
            self.audio_file = audio_file
            self.use_microphone = use_microphone
        except ImportError:
            logging.error(
                "OpenAI Whisper not installed. Install with 'pip install openai-whisper'."
            )
            raise

    def process_audio(self, language="en"):
        if not self.audio_file or not os.path.exists(self.audio_file):
            logging.error(
                f"No valid audio file provided for transcription: {self.audio_file}"
            )
            return ""
        try:
            result = self.model.transcribe(self.audio_file, language=language)
            return result["text"].strip()
        except Exception as e:
            logging.error(f"Whisper transcription failed for {self.audio_file}: {e}")
            return ""

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
    encoding="utf-8",
)

# --- Constants ---
DEFAULT_DATA_DIR = "voice_datasets"
WIKI_API_URL = "https://genshin-impact.fandom.com/api.php"
JMP_API_URL_BASE = "https://genshin.jmp.blue"

# --- Helper Functions ---

def get_phonemes(text, lang="en-us"):
    try:
        phonemes = []
        for sent in sentences(text, lang=lang):
            for word in sent:
                if word.phonemes:
                    phonemes.extend(word.phonemes)
        return " ".join(phonemes) if phonemes else ""
    except Exception as e:
        logging.error(f"Error generating phonemes for '{text}': {e}")
        return ""

def segment_audio_file(
    audio_path, output_dir, transcription="", onset=0.6, offset=0.4, min_duration=1.0, hf_token=""
):
    if not hf_token:
        logging.warning("No Hugging Face token provided for segmentation. Skipping.")
        return []
    try:
        from pyannote.audio import Pipeline
        pipeline = Pipeline.from_pretrained(
            "pyannote/segmentation", use_auth_token=hf_token
        )
        logging.info(f"Segmenting audio file: {audio_path}")
        segments = pipeline(audio_path)
        audio = AudioSegment.from_file(audio_path)
        os.makedirs(output_dir, exist_ok=True)
        wav_files = []

        # Use gruut to estimate word boundaries if transcription is provided
        word_boundaries = []
        if transcription:
            try:
                word_times = []
                total_duration = len(audio) / 1000.0  # Duration in seconds
                words = transcription.split()
                for i, word in enumerate(words):
                    # Approximate word timing (linear distribution)
                    start_time = (i / len(words)) * total_duration
                    end_time = ((i + 1) / len(words)) * total_duration
                    word_times.append((start_time, end_time))
                word_boundaries = word_times
            except Exception as e:
                logging.warning(f"Could not estimate word boundaries: {e}")

        for i, segment in enumerate(segments.get_timeline()):
            start_ms = segment.start * 1000
            end_ms = segment.end * 1000
            duration_sec = (end_ms - start_ms) / 1000
            if duration_sec >= min_duration:
                # Adjust segment to align with word boundaries if available
                if word_boundaries:
                    for word_start, word_end in word_boundaries:
                        if abs(segment.start - word_start) < 0.2 and abs(segment.end - word_end) < 0.2:
                            start_ms = word_start * 1000
                            end_ms = word_end * 1000
                            break
                segment_audio = audio[start_ms:end_ms]
                wav_path = os.path.join(output_dir, f"segment_{i}.wav")
                segment_audio.export(
                    wav_path, format="wav", parameters=["-ar", "22050", "-ac", "1"]
                )
                wav_files.append(wav_path)
        logging.info(
            f"Segmentation complete. Found {len(wav_files)} segments meeting criteria."
        )
        return wav_files
    except ImportError:
        logging.error(
            "PyAnnote.audio not installed. Install with 'pip install pyannote.audio'."
        )
        return []
    except Exception as e:
        logging.error(f"Error during audio segmentation for {audio_path}: {e}")
        return []

def clean_transcript(text, strict_ascii=False):
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("|", " ").strip()
    if strict_ascii:
        text = text.encode("ascii", errors="ignore").decode("ascii")
    return text

def get_category_files(category):
    params = {
        "action": "query",
        "list": "categorymembers",
        "cmtype": "file",
        "cmtitle": category,
        "cmlimit": "max",
        "format": "json",
    }
    files = []
    cmcontinue = None
    logging.info(f"Fetching files from Wiki category: {category}")
    while True:
        if cmcontinue:
            params["cmcontinue"] = cmcontinue
        try:
            response = requests.get(WIKI_API_URL, params=params, timeout=20)
            response.raise_for_status()
            data = response.json()
            new_files = [
                member["title"]
                for member in data.get("query", {}).get("categorymembers", [])
                if not re.search(
                    r"Vo (JA|KO|ZH)", member.get("title", ""), re.IGNORECASE
                )
            ]
            files.extend(new_files)
            if "continue" in data and "cmcontinue" in data["continue"]:
                cmcontinue = data["continue"]["cmcontinue"]
            else:
                break
        except requests.exceptions.RequestException as e:
            logging.error(f"Error fetching category files for {category}: {e}")
            break
        except json.JSONDecodeError as e:
            logging.error(f"Error decoding JSON response for {category}: {e}")
            break
    logging.info(f"Found {len(files)} potential files in category '{category}'.")
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
        response = requests.get(WIKI_API_URL, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
        pages = data.get("query", {}).get("pages", {})
        page_id = list(pages.keys())[0]
        if page_id != "-1" and "imageinfo" in pages[page_id]:
            if pages[page_id]["imageinfo"] and len(pages[page_id]["imageinfo"]) > 0:
                return pages[page_id]["imageinfo"][0].get("url")
        logging.warning(f"No URL found in imageinfo for {file_title}")
        return None
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching file URL for {file_title}: {e}")
        return None
    except (KeyError, IndexError, json.JSONDecodeError) as e:
        logging.error(f"Error parsing response or finding URL for {file_title}: {e}")
        return None

def download_and_convert(file_url, output_dir, file_name, status_queue=None):
    safe_file_name = re.sub(r'[\\/*?:"<>|]', "_", file_name)
    ogg_file_name = (
        safe_file_name
        if safe_file_name.lower().endswith(".ogg")
        else f"{safe_file_name}.ogg"
    )
    wav_file_name = os.path.splitext(ogg_file_name)[0] + ".wav"
    ogg_path = os.path.join(output_dir, "temp_" + ogg_file_name)
    wav_path = os.path.join(output_dir, wav_file_name)
    if os.path.exists(wav_path):
        logging.info(f"Skipping existing WAV: {wav_path}")
        if status_queue:
            status_queue.put(f"Skipped: {wav_file_name}")
        return wav_path
    try:
        if status_queue:
            status_queue.put(f"Downloading {ogg_file_name}...")
        response = requests.get(file_url, timeout=45)
        response.raise_for_status()
        os.makedirs(os.path.dirname(ogg_path), exist_ok=True)
        with open(ogg_path, "wb") as f:
            f.write(response.content)
        logging.info(f"Downloaded OGG: {ogg_path}")
        if status_queue:
            status_queue.put(f"Converting {ogg_file_name} to WAV...")
        process = subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                ogg_path,
                "-ar",
                "22050",
                "-ac",
                "1",
                wav_path,
                "-loglevel",
                "error",
            ],
            check=True,
            text=True,
            capture_output=True,
        )
        if process.stdout:
            logging.debug(f"FFmpeg stdout: {process.stdout}")
        if process.stderr:
            logging.debug(f"FFmpeg stderr: {process.stderr}")
        logging.info(f"Converted to WAV: {wav_path}")
        if status_queue:
            status_queue.put(f"Converted: {wav_file_name}")
        return wav_path
    except requests.exceptions.RequestException as e:
        logging.error(f"Download failed for {file_url}: {e}")
        if status_queue:
            status_queue.put(f"Download Error: {ogg_file_name}")
        return None
    except subprocess.CalledProcessError as e:
        logging.error(f"ffmpeg conversion failed for {ogg_path}: {e}")
        logging.error(f"FFmpeg stderr: {e.stderr}")
        if status_queue:
            status_queue.put(f"Conversion Error: {ogg_file_name}")
        return None
    except Exception as e:
        logging.error(f"Error processing {ogg_file_name} (URL: {file_url}): {e}")
        if status_queue:
            status_queue.put(f"Error with {ogg_file_name}")
        return None
    finally:
        if os.path.exists(ogg_path):
            try:
                os.remove(ogg_path)
            except OSError as e:
                logging.warning(f"Could not remove temporary OGG file {ogg_path}: {e}")

def fetch_character_list_from_api():
    try:
        response = requests.get(f"{JMP_API_URL_BASE}/characters", timeout=15)
        response.raise_for_status()
        character_slugs = response.json()
        character_names = []
        logging.info(f"Fetching details for {len(character_slugs)} characters...")
        for slug in character_slugs:
            char_response = requests.get(
                f"{JMP_API_URL_BASE}/characters/{slug}", timeout=5
            )
            char_response.raise_for_status()
            details = char_response.json()
            if "name" in details:
                character_names.append(details["name"])
        return sorted(list(set(character_names)))
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching character list from API: {e}")
        return []

def is_silent_audio(file_path, silence_threshold=-50.0, min_silence_duration=0.5):
    try:
        audio = AudioSegment.from_wav(file_path)
        # Split audio into chunks and check for sustained silence
        chunk_size_ms = 100  # Analyze in 100ms chunks
        silent_chunks = 0
        total_duration = len(audio) / 1000.0  # Duration in seconds
        for i in range(0, len(audio), chunk_size_ms):
            chunk = audio[i:i + chunk_size_ms]
            if chunk.max_dBFS < silence_threshold:
                silent_chunks += 1
        # Consider audio silent if a significant portion is below threshold
        silence_duration = (silent_chunks * chunk_size_ms) / 1000.0
        is_silent = silence_duration >= min_silence_duration and silence_duration > total_duration * 0.5
        if is_silent:
            logging.info(
                f"Audio file {file_path} detected as silent (max dBFS: {audio.max_dBFS:.2f} < {silence_threshold}, "
                f"silent duration: {silence_duration:.2f}s >= {min_silence_duration}s)."
            )
        return is_silent
    except FileNotFoundError:
        logging.error(f"File not found while checking silence: {file_path}")
        return True
    except Exception as e:
        logging.error(f"Error checking silence for {file_path}: {e}")
        return True

def clean_metadata_file(metadata_path):
    cleaned_lines = []
    invalid_lines_removed = 0
    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        if not lines:
            logging.error(f"Metadata file {metadata_path} is empty.")
            return False
        header_fields = lines[0].strip().split("|")
        if not all(col in header_fields for col in ["text", "audio_file"]):
            logging.error(
                f"Invalid header in {metadata_path}: Missing required columns 'text' or 'audio_file'. Found: {header_fields}"
            )
            return False
        cleaned_lines.append("|".join(header_fields) + "\n")
        for i, line in enumerate(lines[1:], 2):
            original_line = line.strip()
            if not original_line:
                invalid_lines_removed += 1
                continue
            fields = original_line.split("|")
            if len(fields) >= 2 and all(field.strip() for field in fields[:2]):  # text and audio_file required
                cleaned_lines.append("|".join(fields) + "\n")
            else:
                logging.warning(
                    f"Removing invalid metadata line {i}: '{original_line}' from {metadata_path}"
                )
                invalid_lines_removed += 1
        if invalid_lines_removed > 0:
            logging.warning(
                f"Removed {invalid_lines_removed} invalid lines from {metadata_path}."
            )
        if len(cleaned_lines) <= 1:
            logging.error(
                f"No valid data entries found in {metadata_path} after cleaning."
            )
            return False
        with open(metadata_path, "w", encoding="utf-8", newline="") as f:
            f.writelines(cleaned_lines)
        logging.info(
            f"Cleaned {metadata_path}: {len(cleaned_lines)-1} valid entries remain."
        )
        return True
    except FileNotFoundError:
        logging.error(f"Metadata file not found for cleaning: {metadata_path}")
        return False
    except Exception as e:
        logging.error(f"Error cleaning metadata file {metadata_path}: {e}")
        return False

def generate_valid_csv(metadata_path, valid_ratio=0.2):
    logging.info(
        f"Generating validation split from {metadata_path} (ratio: {valid_ratio})"
    )
    valid_path = os.path.join(os.path.dirname(metadata_path), "valid.csv")
    try:
        df = pd.read_csv(metadata_path, sep="|", encoding="utf-8", on_bad_lines="error")
        if len(df) < 2:
            logging.warning(
                f"Not enough data ({len(df)} samples) in {metadata_path} to create a validation split. Skipping."
            )
            if os.path.exists(valid_path):
                os.remove(valid_path)
            return None
        n_samples = len(df)
        n_valid = max(1, int(n_samples * valid_ratio))
        if n_valid >= n_samples:
            n_valid = n_samples - 1
        indices = np.random.permutation(n_samples)
        valid_df = df.iloc[indices[:n_valid]]
        train_df = df.iloc[indices[n_valid:]]
        backup_file(metadata_path, "metadata_presplit_backup")
        if os.path.exists(valid_path):
            backup_file(valid_path, "valid_presplit_backup")
        train_df.to_csv(
            metadata_path,
            sep="|",
            index=False,
            encoding="utf-8",
            quoting=csv.QUOTE_MINIMAL,
        )
        valid_df.to_csv(
            valid_path,
            sep="|",
            index=False,
            encoding="utf-8",
            quoting=csv.QUOTE_MINIMAL,
        )
        logging.info(
            f"Split metadata: {len(train_df)} train samples written to {metadata_path}, "
            f"{len(valid_df)} validation samples written to {valid_path}"
        )
        return valid_path
    except FileNotFoundError:
        logging.error(f"Metadata file not found for splitting: {metadata_path}")
        return None
    except pd.errors.EmptyDataError:
        logging.error(f"Metadata file {metadata_path} is empty, cannot split.")
        return None
    except Exception as e:
        logging.error(f"Error generating validation split from {metadata_path}: {e}")
        return None

def transcribe_character_audio(
    character_output_dir,
    whisper_model="base",
    use_segmentation=False,
    hf_token="",
    strict_ascii=False,
    min_silence_duration=0.5,
    status_queue=None,
):
    metadata_path = os.path.join(character_output_dir, "metadata.csv")
    wavs_dir = os.path.join(character_output_dir, "wavs")
    silent_dir = os.path.join(character_output_dir, "silent_files")
    temp_dir = os.path.join(character_output_dir, "temp_audio")
    os.makedirs(wavs_dir, exist_ok=True)
    os.makedirs(silent_dir, exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True)

    existing_transcribed_files = set()
    if os.path.exists(metadata_path) and validate_metadata_layout(metadata_path):
        try:
            df_existing = pd.read_csv(
                metadata_path, sep="|", encoding="utf-8", on_bad_lines="skip"
            )
            if all(col in df_existing.columns for col in ["text", "audio_file"]):
                existing_transcribed_files = set(
                    df_existing["audio_file"].apply(lambda x: os.path.basename(str(x)))
                )
                logging.info(
                    f"Loaded {len(existing_transcribed_files)} existing entries from {metadata_path}"
                )
            else:
                logging.warning(
                    f"Metadata file {metadata_path} is missing required columns. Treating as empty."
                )
        except pd.errors.EmptyDataError:
            logging.info(f"Metadata file {metadata_path} is empty. Starting fresh.")
        except Exception as e:
            logging.error(
                f"Error reading existing metadata {metadata_path}: {e}. Starting fresh."
            )
            backup_file(metadata_path, "metadata_read_error_backup")
            if os.path.exists(metadata_path):
                os.remove(metadata_path)

    files_to_process = []
    for item in os.listdir(character_output_dir):
        item_path = os.path.join(character_output_dir, item)
        if item.lower().endswith(".wav") and os.path.isfile(item_path):
            try:
                shutil.move(item_path, os.path.join(temp_dir, item))
                logging.info(f"Moved {item} to temp directory for processing.")
            except Exception as e:
                logging.warning(f"Could not move {item} to temp directory: {e}")

    for file in os.listdir(temp_dir):
        if file.lower().endswith(".wav"):
            temp_path = os.path.join(temp_dir, file)
            if file in existing_transcribed_files:
                logging.info(f"Skipping already transcribed file: {file}")
                try:
                    shutil.move(temp_path, os.path.join(wavs_dir, file))
                except Exception as e:
                    logging.warning(
                        f"Could not move skipped file {file} to wavs dir: {e}"
                    )
                continue
            if use_segmentation and hf_token:
                if status_queue:
                    status_queue.put(f"Transcribing for segmentation: {file}")
                # Pre-transcribe to guide segmentation
                try:
                    stt = SpeechToText(
                        use_microphone=False,
                        audio_file=temp_path,
                        engine="whisper",
                        whisper_model_size=whisper_model,
                    )
                    pre_transcript = stt.process_audio(language="en")
                    cleaned_pre_transcript = clean_transcript(
                        pre_transcript, strict_ascii=strict_ascii
                    )
                except Exception as e:
                    logging.warning(f"Pre-transcription failed for {file}: {e}")
                    cleaned_pre_transcript = ""
                if status_queue:
                    status_queue.put(f"Segmenting: {file}")
                segmented_files = segment_audio_file(
                    temp_path, wavs_dir, transcription=cleaned_pre_transcript, hf_token=hf_token, min_duration=1.0
                )
                if segmented_files:
                    files_to_process.extend(
                        [
                            os.path.basename(f)
                            for f in segmented_files
                            if os.path.basename(f) not in existing_transcribed_files
                        ]
                    )
                    try:
                        os.remove(temp_path)
                    except OSError as e:
                        logging.warning(
                            f"Could not remove original segmented file {temp_path}: {e}"
                        )
                else:
                    try:
                        shutil.move(temp_path, os.path.join(wavs_dir, file))
                        files_to_process.append(file)
                    except Exception as e:
                        logging.warning(
                            f"Could not move original file {file} to wavs dir after failed segmentation: {e}"
                        )
            else:
                try:
                    shutil.move(temp_path, os.path.join(wavs_dir, file))
                    files_to_process.append(file)
                except Exception as e:
                    logging.warning(f"Could not move file {file} to wavs dir: {e}")

    for file in os.listdir(wavs_dir):
        if (
            file.lower().endswith(".wav")
            and file not in existing_transcribed_files
            and file not in files_to_process
        ):
            files_to_process.append(file)
            logging.info(f"Found untranscribed file in wavs_dir: {file}")

    if os.path.exists(temp_dir) and not os.listdir(temp_dir):
        try:
            os.rmdir(temp_dir)
        except OSError as e:
            logging.warning(f"Could not remove empty temp directory {temp_dir}: {e}")

    if not files_to_process:
        logging.info("No new audio files found to transcribe.")
        if status_queue:
            status_queue.put("No new files to transcribe.")
        if os.path.exists(metadata_path):
            valid_csv_path = generate_valid_csv(metadata_path, valid_ratio=0.2)
            return valid_csv_path is not None
        else:
            try:
                with open(metadata_path, "w", encoding="utf-8", newline="") as mf:
                    writer = csv.writer(mf, delimiter="|", lineterminator="\n")
                    writer.writerow(["text", "audio_file", "phonemes"])
                logging.info(f"Created empty metadata file: {metadata_path}")
                return True
            except Exception as e:
                logging.error(
                    f"Failed to create empty metadata file {metadata_path}: {e}"
                )
                return False

    logging.info(
        f"Found {len(files_to_process)} new WAV files to transcribe using Whisper {whisper_model}..."
    )
    transcribed_count = 0
    failed_count = 0
    silent_count = 0
    file_mode = "a" if existing_transcribed_files else "w"

    try:
        with open(metadata_path, file_mode, encoding="utf-8", newline="") as mf:
            writer = csv.writer(
                mf, delimiter="|", lineterminator="\n", quoting=csv.QUOTE_MINIMAL
            )
            if file_mode == "w":
                writer.writerow(["text", "audio_file", "phonemes"])
            for file in tqdm(files_to_process, desc="Transcribing Audio"):
                wav_path = os.path.join(wavs_dir, file)
                if not os.path.exists(wav_path):
                    logging.warning(
                        f"File {file} listed for processing but not found in {wavs_dir}. Skipping."
                    )
                    failed_count += 1
                    continue
                if status_queue:
                    status_queue.put(f"Checking: {file}")
                if is_silent_audio(wav_path, min_silence_duration=min_silence_duration):
                    logging.warning(f"Moving silent file {file} to {silent_dir}")
                    try:
                        shutil.move(wav_path, os.path.join(silent_dir, file))
                        silent_count += 1
                        if status_queue:
                            status_queue.put(f"Moved silent file: {file}")
                    except Exception as move_err:
                        logging.error(f"Failed to move silent file {file}: {move_err}")
                        failed_count += 1
                    continue
                transcript = None
                for attempt in range(2):
                    try:
                        if status_queue:
                            status_queue.put(
                                f"Transcribing: {file} (Attempt {attempt+1})"
                            )
                        stt = SpeechToText(
                            use_microphone=False,
                            audio_file=wav_path,
                            engine="whisper",
                            whisper_model_size=whisper_model,
                        )
                        audio_transcript = stt.process_audio(language="en")
                        cleaned_transcript = clean_transcript(
                            audio_transcript, strict_ascii=strict_ascii
                        )
                        if cleaned_transcript:
                            transcript = cleaned_transcript
                            break
                        else:
                            logging.warning(
                                f"Empty transcription for {file} on attempt {attempt+1}. Transcript: '{audio_transcript}'"
                            )
                    except Exception as e:
                        logging.error(
                            f"Transcription error for {file} on attempt {attempt+1}: {e}"
                        )
                if transcript:
                    phonemes = get_phonemes(transcript)
                    relative_audio_path = f"wavs/{file}"
                    writer.writerow([transcript, relative_audio_path, phonemes])
                    transcribed_count += 1
                    if status_queue:
                        status_queue.put(f"Transcribed: {file}")
                else:
                    logging.warning(
                        f"Moving failed transcription file {file} to {silent_dir}"
                    )
                    try:
                        shutil.move(wav_path, os.path.join(silent_dir, file))
                        failed_count += 1
                        if status_queue:
                            status_queue.put(f"Moved failed file: {file}")
                    except Exception as move_err:
                        logging.error(
                            f"Failed to move failed transcription file {file}: {move_err}"
                        )
    except IOError as e:
        logging.error(f"Error writing to metadata file {metadata_path}: {e}")
        return False
    except Exception as e:
        logging.error(
            f"Unexpected error during transcription loop: {e}"
        )
        return False

    logging.info(
        f"Transcription complete. Successful: {transcribed_count}, Failed: {failed_count}, Silent/Moved: {silent_count}."
    )
    if status_queue:
        status_queue.put(
            f"Transcription complete. OK: {transcribed_count}, Fail: {failed_count}, Silent: {silent_count}."
        )

    if os.path.exists(metadata_path):
        final_clean_success = clean_metadata_file(metadata_path)
        if final_clean_success:
            valid_csv_path = generate_valid_csv(metadata_path, valid_ratio=0.2)
            return valid_csv_path is not None
        else:
            logging.error("Final metadata cleaning failed.")
            return False
    else:
        logging.error("Metadata file does not exist after transcription process.")
        return False

def validate_metadata_layout(metadata_path):
    if not os.path.exists(metadata_path):
        logging.warning(f"Metadata layout check: File not found {metadata_path}")
        return False
    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="|")
            try:
                header = next(reader)
                if not all(col in header for col in ["text", "audio_file"]):
                    logging.error(
                        f"Invalid header in {metadata_path}: Missing required columns 'text' or 'audio_file'. Found: {header}"
                    )
                    return False
            except StopIteration:
                logging.error(f"Metadata file {metadata_path} is empty.")
                return False
            for i, row in enumerate(reader):
                if i >= 5:
                    break
                if len(row) < 2:  # text and audio_file required
                    logging.error(
                        f"Incorrect column count ({len(row)}) found at line {i+2} in {metadata_path}. Expected at least 2."
                    )
                    return False
        return True
    except csv.Error as e:
        logging.error(
            f"CSV parsing error during layout validation of {metadata_path}: {e}"
        )
        return False
    except Exception as e:
        logging.error(f"Error validating layout of {metadata_path}: {e}")
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
    min_silence_duration=0.5,
    status_queue=None,
    stop_event=None,
):
    if stop_event and stop_event.is_set():
        logging.info("Processing cancelled before starting.")
        if status_queue:
            status_queue.put("Processing cancelled.")
        return None
    safe_character_name = re.sub(r'[\\/*?:"<>|]', "_", character)
    character_folder = os.path.join(base_output_dir, safe_character_name)
    wavs_folder = os.path.join(character_folder, "wavs")
    os.makedirs(wavs_folder, exist_ok=True)
    if download_wiki_audio:
        logging.info(f"--- Starting Wiki Download for {character} ---")
        if status_queue:
            status_queue.put(f"Downloading Wiki audio for {character}...")
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
            if stop_event and stop_event.is_set():
                logging.info("Download cancelled during category fetching.")
                if status_queue:
                    status_queue.put("Download cancelled.")
                return None
            files_to_download.extend(get_category_files(category))
        unique_files = sorted(list(set(files_to_download)))
        logging.info(
            f"Found {len(unique_files)} unique potential audio files from Wiki categories."
        )
        downloaded_count = 0
        failed_count = 0
        skipped_count = 0
        for i, file_title in enumerate(
            tqdm(unique_files, desc=f"Downloading {character} Audio")
        ):
            if stop_event and stop_event.is_set():
                logging.info("Download cancelled during file processing.")
                if status_queue:
                    status_queue.put("Download cancelled.")
                return None
            match = re.match(r"File:(.*)", file_title, re.IGNORECASE)
            if not match:
                logging.warning(f"Could not parse file name from title: {file_title}")
                failed_count += 1
                continue
            file_name = match.group(1).strip()
            file_url = get_file_url(file_title)
            if file_url:
                wav_file_path = download_and_convert(
                    file_url, wavs_folder, file_name, status_queue=status_queue
                )
                if wav_file_path:
                    is_skipped = False
                    if status_queue:
                        try:
                            q_list = list(status_queue.queue)[-5:]
                            if any(
                                f"Skipped: {os.path.basename(wav_file_path)}" in msg
                                for msg in q_list
                            ):
                                is_skipped = True
                        except Exception:
                            pass
                    if is_skipped:
                        skipped_count += 1
                    else:
                        downloaded_count += 1
                else:
                    failed_count += 1
            else:
                logging.warning(f"No URL found for {file_title}. Skipping.")
                failed_count += 1
        logging.info(
            f"Download phase complete. Newly Downloaded: {downloaded_count}, Skipped/Existing: {skipped_count}, Failed: {failed_count}."
        )
        if status_queue:
            status_queue.put(
                f"Download complete. New: {downloaded_count}, Skip: {skipped_count}, Fail: {failed_count}."
            )
    else:
        logging.info("Skipping Wiki download as requested.")
        if status_queue:
            status_queue.put("Skipping Wiki download.")
    if stop_event and stop_event.is_set():
        logging.info("Processing cancelled before transcription.")
        if status_queue:
            status_queue.put("Processing cancelled.")
        return None
    logging.info(f"--- Starting Transcription for {character} ---")
    if status_queue:
        status_queue.put(f"Starting transcription for {character}...")
    transcription_success = transcribe_character_audio(
        character_folder,
        whisper_model,
        use_segmentation,
        hf_token,
        strict_ascii,
        min_silence_duration,
        status_queue=status_queue,
    )
    if stop_event and stop_event.is_set():
        logging.info("Processing cancelled during/after transcription.")
        if status_queue:
            status_queue.put("Processing cancelled.")
        return None
    if not transcription_success:
        logging.error(
            f"Transcription process failed or produced no valid data for {character}."
        )
        if status_queue:
            status_queue.put(f"Transcription failed for {character}.")
        return None
    metadata_path = os.path.join(character_folder, "metadata.csv")
    valid_metadata_path = os.path.join(character_folder, "valid.csv")
    if not os.path.exists(metadata_path) or not os.path.exists(valid_metadata_path):
        logging.error(
            f"Metadata or validation file missing after transcription for {character}. Check logs."
        )
        if status_queue:
            status_queue.put("Metadata generation failed post-transcription.")
        return None
    logging.info(
        f"--- Successfully processed voices for {character}. ---"
    )
    if status_queue:
        status_queue.put(f"Processing complete for {character}.")
    return character_folder

def backup_file(path, suffix="backup"):
    if os.path.exists(path):
        timestamp = pd.Timestamp.now().strftime("%Y%m%d%H%M%S")
        if os.path.isfile(path):
            base, ext = os.path.splitext(path)
            backup_path = f"{base}.{suffix}.{timestamp}{ext}"
        elif os.path.isdir(path):
            backup_path = f"{path}.{suffix}.{timestamp}"
        else:
            logging.warning(
                f"Path exists but is neither file nor directory: {path}. Skipping backup."
            )
            return
        try:
            if os.path.isfile(path):
                shutil.copy2(path, backup_path)
                logging.info(f"Backup of file created: {backup_path}")
            elif os.path.isdir(path):
                shutil.copytree(path, backup_path, dirs_exist_ok=True)
                logging.info(f"Backup of directory created: {backup_path}")
        except Exception as e:
            logging.error(f"Failed to create backup for {path} -> {backup_path}: {e}")
    else:
        logging.warning(f"Cannot backup non-existent path: {path}")

# --- GUI ---

def load_csv_to_treeview(treeview, csv_path, base_dir, play_callback):
    # Clear existing items and columns
    treeview.delete(*treeview.get_children())
    for col in treeview["columns"]:
        treeview.heading(col, text="")
        treeview.column(col, width=0)
    treeview["columns"] = ()

    if not os.path.exists(csv_path):
        logging.info(f"CSV file not found for Treeview: {csv_path}")
        return

    try:
        df = pd.read_csv(csv_path, sep="|", encoding="utf-8", on_bad_lines="skip")
        if not all(col in df.columns for col in ["text", "audio_file"]):
            logging.warning(f"Invalid columns in {csv_path}. Expected at least 'text' and 'audio_file'.")
            return

        # Define columns: all CSV columns + "Play"
        columns = list(df.columns) + ["Play"]
        treeview["columns"] = columns

        # Configure column headings and widths
        for col in columns:
            treeview.heading(col, text=col if col != "Play" else "Play")
            if col == "Play":
                treeview.column(col, width=60, anchor=tk.CENTER, stretch=False)
            else:
                treeview.column(col, width=150, anchor=tk.W)

        # Sort by audio_file if present
        if "audio_file" in df.columns:
            df = df.sort_values(by="audio_file")

        # Insert rows
        for idx, row in df.iterrows():
            values = []
            audio_path = None
            for col in df.columns:
                value = str(row[col]) if pd.notna(row[col]) else ""
                values.append(value)
                if col == "audio_file":
                    audio_path = os.path.join(base_dir, value) if value else None
            # Append empty value for Play button
            values.append("Play")
            treeview.insert("", tk.END, iid=idx, values=values, tags=(audio_path,))
        logging.info(f"Loaded {len(df)} rows into Treeview from {csv_path}")
    except Exception as e:
        logging.error(f"Error loading {csv_path} into Treeview: {e}")

def create_table_frame(notebook, tab_name):
    frame = ttk.Frame(notebook)
    tree = ttk.Treeview(frame, show="headings", height=10)
    tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
    vscroll = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=tree.yview)
    vscroll.grid(row=0, column=1, sticky=(tk.N, tk.S))
    tree["yscrollcommand"] = vscroll.set
    hscroll = ttk.Scrollbar(frame, orient=tk.HORIZONTAL, command=tree.xview)
    hscroll.grid(row=1, column=0, sticky=(tk.W, tk.E))
    tree["xscrollcommand"] = hscroll.set
    frame.columnconfigure(0, weight=1)
    frame.rowconfigure(0, weight=1)
    return frame, tree

def create_json_frame(notebook, tab_name):
    frame = ttk.Frame(notebook)
    json_text = tk.Text(frame, height=10, width=50, wrap=tk.NONE)
    json_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
    vscroll = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=json_text.yview)
    vscroll.grid(row=0, column=1, sticky=(tk.N, tk.S))
    json_text["yscrollcommand"] = vscroll.set
    hscroll = ttk.Scrollbar(frame, orient=tk.HORIZONTAL, command=json_text.xview)
    hscroll.grid(row=1, column=0, sticky=(tk.W, tk.E))
    json_text["xscrollcommand"] = hscroll.set
    frame.columnconfigure(0, weight=1)
    frame.rowconfigure(0, weight=1)
    return frame, json_text

def main_gui():
    global window
    window = tk.Tk()
    window.title("Genshin Voice Downloader")
    window.geometry("600x500")

    # Initialize pygame mixer
    pygame.mixer.init()
    current_playing = [None]  # Track currently playing audio

    # Variables
    character_var = tk.StringVar()
    output_dir_var = tk.StringVar(value=DEFAULT_DATA_DIR)
    additional_csvs_var = tk.StringVar(value="")  # Comma-separated CSV filenames
    language_var = tk.StringVar(value="English")
    whisper_model_var = tk.StringVar(value="base")
    use_segmentation_var = tk.BooleanVar(value=False)
    hf_token_var = tk.StringVar()
    strict_ascii_var = tk.BooleanVar(value=False)
    min_silence_duration_var = tk.StringVar(value="0.5")
    status_queue = queue.Queue()
    stop_event = threading.Event()
    current_thread = [None]
    current_character_folder = [None]
    notebook = [None]  # Store notebook for dynamic updates
    tables = {}  # Store Treeview widgets for each CSV
    json_text_widget = [None]  # Store JSON Text widget

    # Frames
    main_frame = ttk.Frame(window, padding="10")
    main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
    window.columnconfigure(0, weight=1)
    window.rowconfigure(0, weight=1)

    # Input Frame
    input_frame = ttk.Frame(main_frame)
    input_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)

    # Character Selection
    ttk.Label(input_frame, text="Character:").grid(row=0, column=0, sticky=tk.W, pady=2)
    character_combobox = ttk.Combobox(input_frame, textvariable=character_var, width=30)
    character_combobox.grid(row=0, column=1, sticky=(tk.W, tk.E), pady=2)
    character_combobox["values"] = fetch_character_list_from_api()
    character_combobox["state"] = "readonly"

    # Output Directory
    ttk.Label(input_frame, text="Output Directory:").grid(row=1, column=0, sticky=tk.W, pady=2)
    output_dir_entry = ttk.Entry(input_frame, textvariable=output_dir_var, width=25)
    output_dir_entry.grid(row=1, column=1, sticky=(tk.W, tk.E), pady=2)
    ttk.Button(input_frame, text="Browse", command=lambda: browse_output_dir(output_dir_var)).grid(row=1, column=2, padx=5, pady=2)

    # Additional CSVs
    ttk.Label(input_frame, text="Additional CSVs (comma-separated):").grid(row=2, column=0, sticky=tk.W, pady=2)
    ttk.Entry(input_frame, textvariable=additional_csvs_var, width=30).grid(row=2, column=1, columnspan=2, sticky=(tk.W, tk.E), pady=2)

    # Language Selection
    ttk.Label(input_frame, text="Language:").grid(row=3, column=0, sticky=tk.W, pady=2)
    language_combobox = ttk.Combobox(
        input_frame,
        textvariable=language_var,
        values=["English", "Japanese", "Chinese", "Korean"],
        state="readonly",
        width=30,
    )
    language_combobox.grid(row=3, column=1, columnspan=2, sticky=(tk.W, tk.E), pady=2)

    # Whisper Model Selection
    ttk.Label(input_frame, text="Whisper Model:").grid(row=4, column=0, sticky=tk.W, pady=2)
    whisper_combobox = ttk.Combobox(
        input_frame,
        textvariable=whisper_model_var,
        values=["base", "small", "medium", "large-v2"],
        state="readonly",
        width=30,
    )
    whisper_combobox.grid(row=4, column=1, columnspan=2, sticky=(tk.W, tk.E), pady=2)

    # Segmentation and Token
    ttk.Checkbutton(
        input_frame,
        text="Use Audio Segmentation (requires HF token)",
        variable=use_segmentation_var,
    ).grid(row=5, column=0, columnspan=3, sticky=tk.W, pady=2)
    ttk.Label(input_frame, text="Hugging Face Token:").grid(row=6, column=0, sticky=tk.W, pady=2)
    ttk.Entry(input_frame, textvariable=hf_token_var, width=33).grid(
        row=6, column=1, columnspan=2, sticky=(tk.W, tk.E), pady=2
    )

    # Strict ASCII
    ttk.Checkbutton(
        input_frame,
        text="Strict ASCII Transcription",
        variable=strict_ascii_var,
    ).grid(row=7, column=0, columnspan=3, sticky=tk.W, pady=2)

    # Minimum Silence Duration
    ttk.Label(input_frame, text="Min Silence Duration (s):").grid(row=8, column=0, sticky=tk.W, pady=2)
    ttk.Entry(input_frame, textvariable=min_silence_duration_var, width=33).grid(
        row=8, column=1, columnspan=2, sticky=(tk.W, tk.E), pady=2
    )

    input_frame.columnconfigure(1, weight=1)

    # Tabbed Interface
    notebook_frame = ttk.Frame(main_frame)
    notebook_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=10)
    notebook[0] = ttk.Notebook(notebook_frame)
    notebook[0].grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
    notebook_frame.columnconfigure(0, weight=1)
    notebook_frame.rowconfigure(0, weight=1)

    # Status Tab
    status_frame = ttk.Frame(notebook[0])
    notebook[0].add(status_frame, text="Status")
    status_text = tk.Text(status_frame, height=10, width=50, wrap=tk.WORD)
    status_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
    status_scroll = ttk.Scrollbar(status_frame, orient=tk.VERTICAL, command=status_text.yview)
    status_scroll.grid(row=0, column=1, sticky=(tk.N, tk.S))
    status_text["yscrollcommand"] = status_scroll.set
    status_frame.columnconfigure(0, weight=1)
    status_frame.rowconfigure(0, weight=1)

    # Audio Playback Callback
    def play_audio(event):
        tree = event.widget
        item = tree.identify_row(event.y)
        if not item:
            return
        column = tree.identify_column(event.x)
        if tree.heading(column)["text"].lower() != "play":
            return
        audio_path = tree.item(item, "tags")[0] if tree.item(item, "tags") else None
        if not audio_path or not os.path.exists(audio_path):
            logging.error(f"Audio file not found: {audio_path}")
            status_text.insert(tk.END, f"Error: Audio file not found: {audio_path}\n")
            status_text.see(tk.END)
            return
        try:
            # Stop any currently playing audio
            if current_playing[0]:
                pygame.mixer.music.stop()
                current_playing[0] = None
            # Play new audio
            pygame.mixer.music.load(audio_path)
            pygame.mixer.music.play()
            current_playing[0] = audio_path
            status_text.insert(tk.END, f"Playing: {os.path.basename(audio_path)}\n")
            status_text.see(tk.END)
        except Exception as e:
            logging.error(f"Error playing audio {audio_path}: {e}")
            status_text.insert(tk.END, f"Error playing {audio_path}: {e}\n")
            status_text.see(tk.END)

    # Save JSON Callback
    def save_json():
        if not current_character_folder[0]:
            messagebox.showerror("No Character", "Please select a character first.", parent=window)
            return
        json_content = json_text_widget[0].get("1.0", tk.END).strip()
        if not json_content:
            messagebox.showerror("Empty JSON", "JSON content is empty.", parent=window)
            return
        try:
            # Validate JSON syntax
            json.loads(json_content)
        except json.JSONDecodeError as e:
            messagebox.showerror("Invalid JSON", f"Invalid JSON format: {e}", parent=window)
            return
        json_path = os.path.join(current_character_folder[0], "data.json")
        try:
            with open(json_path, "w", encoding="utf-8") as f:
                f.write(json_content)
            status_text.insert(tk.END, f"Saved JSON to {json_path}\n")
            status_text.see(tk.END)
            logging.info(f"Saved JSON to {json_path}")
        except Exception as e:
            status_text.insert(tk.END, f"Error saving JSON to {json_path}: {e}\n")
            status_text.see(tk.END)
            logging.error(f"Error saving JSON to {json_path}: {e}")
            messagebox.showerror("Save Error", f"Failed to save JSON: {e}", parent=window)

    # Initialize Default Tables and JSON Tab
    def update_notebook_tabs():
        # Remove existing tabs except Status
        for tab_id in notebook[0].tabs():
            if notebook[0].tab(tab_id, "text") != "Status":
                notebook[0].forget(tab_id)
        tables.clear()
        json_text_widget[0] = None
        # Add default tables
        csv_files = ["metadata.csv", "valid.csv"]
        tab_names = ["Transcriptions", "Validation"]
        # Add additional CSVs
        additional_csvs = [csv.strip() for csv in additional_csvs_var.get().split(",") if csv.strip()]
        for csv_file in additional_csvs:
            if csv_file.endswith(".csv") and csv_file not in csv_files:
                csv_files.append(csv_file)
                tab_names.append(os.path.splitext(csv_file)[0].capitalize())
        # Create CSV tabs
        for csv_file, tab_name in zip(csv_files, tab_names):
            frame, tree = create_table_frame(notebook[0], tab_name)
            notebook[0].add(frame, text=tab_name)
            tables[csv_file] = tree
            tree.bind("<ButtonRelease-1>", play_audio)
        # Create JSON tab
        json_frame, json_text = create_json_frame(notebook[0], "JSON Data")
        notebook[0].add(json_frame, text="JSON Data")
        json_text_widget[0] = json_text

    update_notebook_tabs()

    # Browse Output Directory
    def browse_output_dir(output_dir_var):
        directory = filedialog.askdirectory(initialdir=output_dir_var.get() or os.getcwd())
        if directory:
            output_dir_var.set(directory)
            update_character_tables()

    # Update Tables and JSON on Character Selection
    def update_character_tables(*args):
        character = character_var.get()
        output_dir = output_dir_var.get()
        if not character or not output_dir:
            for csv_file, tree in tables.items():
                tree.delete(*tree.get_children())
                tree["columns"] = ()
            if json_text_widget[0]:
                json_text_widget[0].delete("1.0", tk.END)
            current_character_folder[0] = None
            return
        safe_character_name = re.sub(r'[\\/*?:"<>|]', "_", character)
        character_folder = os.path.join(output_dir, safe_character_name)
        current_character_folder[0] = character_folder
        # Stop any playing audio
        if current_playing[0]:
            pygame.mixer.music.stop()
            current_playing[0] = None
            status_text.insert(tk.END, "Stopped audio playback due to character change.\n")
            status_text.see(tk.END)
        # Update CSV tables
        for csv_file, tree in tables.items():
            csv_path = os.path.join(character_folder, csv_file)
            load_csv_to_treeview(tree, csv_path, character_folder, play_audio)
        # Update JSON tab
        if json_text_widget[0]:
            json_text_widget[0].delete("1.0", tk.END)
            metadata_path = os.path.join(character_folder, "metadata.csv")
            if os.path.exists(metadata_path):
                try:
                    df = pd.read_csv(metadata_path, sep="|", encoding="utf-8", on_bad_lines="skip")
                    if all(col in df.columns for col in ["text", "audio_file"]):
                        # Convert DataFrame to JSON
                        json_data = df.to_dict(orient="records")
                        json_str = json.dumps(json_data, indent=2, ensure_ascii=False)
                        json_text_widget[0].insert("1.0", json_str)
                        status_text.insert(tk.END, f"Loaded JSON data from {metadata_path}\n")
                        status_text.see(tk.END)
                    else:
                        status_text.insert(tk.END, f"Invalid columns in {metadata_path} for JSON\n")
                        status_text.see(tk.END)
                except Exception as e:
                    status_text.insert(tk.END, f"Error loading JSON from {metadata_path}: {e}\n")
                    status_text.see(tk.END)
                    logging.error(f"Error loading JSON from {metadata_path}: {e}")

    character_var.trace("w", update_character_tables)
    additional_csvs_var.trace("w", lambda *args: update_notebook_tabs() or update_character_tables())

    # Update Status and Tables
    def update_status():
        while True:
            try:
                message = status_queue.get_nowait()
                status_text.insert(tk.END, message + "\n")
                status_text.see(tk.END)
                # Check for completion to update tables and JSON
                if "Processing complete for" in message and current_character_folder[0]:
                    for csv_file, tree in tables.items():
                        csv_path = os.path.join(current_character_folder[0], csv_file)
                        load_csv_to_treeview(tree, csv_path, current_character_folder[0], play_audio)
                    # Update JSON tab
                    if json_text_widget[0]:
                        json_text_widget[0].delete("1.0", tk.END)
                        metadata_path = os.path.join(current_character_folder[0], "metadata.csv")
                        if os.path.exists(metadata_path):
                            try:
                                df = pd.read_csv(metadata_path, sep="|", encoding="utf-8", on_bad_lines="skip")
                                if all(col in df.columns for col in ["text", "audio_file"]):
                                    json_data = df.to_dict(orient="records")
                                    json_str = json.dumps(json_data, indent=2, ensure_ascii=False)
                                    json_text_widget[0].insert("1.0", json_str)
                                    status_text.insert(tk.END, f"Reloaded JSON data from {metadata_path}\n")
                                    status_text.see(tk.END)
                                else:
                                    status_text.insert(tk.END, f"Invalid columns in {metadata_path} for JSON\n")
                                    status_text.see(tk.END)
                            except Exception as e:
                                status_text.insert(tk.END, f"Error reloading JSON from {metadata_path}: {e}\n")
                                status_text.see(tk.END)
            except queue.Empty:
                break
        window.after(100, update_status)

    # Refresh Tables and JSON on Tab Selection
    def on_tab_change(event):
        if current_character_folder[0]:
            selected_tab = notebook[0].select()
            tab_text = notebook[0].tab(selected_tab, "text")
            if tab_text == "JSON Data" and json_text_widget[0]:
                metadata_path = os.path.join(current_character_folder[0], "metadata.csv")
                json_text_widget[0].delete("1.0", tk.END)
                if os.path.exists(metadata_path):
                    try:
                        df = pd.read_csv(metadata_path, sep="|", encoding="utf-8", on_bad_lines="skip")
                        if all(col in df.columns for col in ["text", "audio_file"]):
                            json_data = df.to_dict(orient="records")
                            json_str = json.dumps(json_data, indent=2, ensure_ascii=False)
                            json_text_widget[0].insert("1.0", json_str)
                            status_text.insert(tk.END, f"Refreshed JSON data from {metadata_path}\n")
                            status_text.see(tk.END)
                        else:
                            status_text.insert(tk.END, f"Invalid columns in {metadata_path} for JSON\n")
                            status_text.see(tk.END)
                    except Exception as e:
                        status_text.insert(tk.END, f"Error refreshing JSON from {metadata_path}: {e}\n")
                        status_text.see(tk.END)
            else:
                for csv_file, tree in tables.items():
                    expected_tab_name = "Transcriptions" if csv_file == "metadata.csv" else \
                                       "Validation" if csv_file == "valid.csv" else \
                                       os.path.splitext(csv_file)[0].capitalize()
                    if tab_text == expected_tab_name:
                        csv_path = os.path.join(current_character_folder[0], csv_file)
                        load_csv_to_treeview(tree, csv_path, current_character_folder[0], play_audio)

    notebook[0].bind("<<NotebookTabChanged>>", on_tab_change)

    # Buttons
    button_frame = ttk.Frame(main_frame)
    button_frame.grid(row=2, column=0, columnspan=2, pady=10)

    def start_processing():
        if current_thread[0] and current_thread[0].is_alive():
            messagebox.showwarning("Process Running", "A process is already running!", parent=window)
            return
        character = character_var.get()
        output_dir = output_dir_var.get()
        if not character:
            messagebox.showerror("Input Error", "Please select a character.", parent=window)
            return
        if not output_dir:
            messagebox.showerror("Input Error", "Please specify an output directory.", parent=window)
            return
        try:
            min_silence_duration = float(min_silence_duration_var.get())
            if min_silence_duration <= 0:
                raise ValueError("Minimum silence duration must be positive.")
        except ValueError:
            messagebox.showerror(
                "Input Error",
                "Invalid Minimum Silence Duration. Must be a positive number.",
                parent=window,
            )
            return
        # Create output directory if it doesn't exist
        try:
            os.makedirs(output_dir, exist_ok=True)
        except OSError as e:
            messagebox.showerror("Directory Error", f"Cannot create output directory: {e}", parent=window)
            return
        stop_event.clear()
        status_text.delete(1.0, tk.END)
        # Clear tables and JSON
        for tree in tables.values():
            tree.delete(*tree.get_children())
            tree["columns"] = ()
        if json_text_widget[0]:
            json_text_widget[0].delete("1.0", tk.END)
        # Stop any playing audio
        if current_playing[0]:
            pygame.mixer.music.stop()
            current_playing[0] = None

        def process_thread():
            current_character_folder[0] = process_character_voices(
                character=character,
                language=language_var.get(),
                base_output_dir=output_dir,
                download_wiki_audio=True,
                whisper_model=whisper_model_var.get(),
                use_segmentation=use_segmentation_var.get(),
                hf_token=hf_token_var.get(),
                strict_ascii=strict_ascii_var.get(),
                min_silence_duration=min_silence_duration,
                status_queue=status_queue,
                stop_event=stop_event,
            )

        current_thread[0] = threading.Thread(target=process_thread, daemon=True)
        current_thread[0].start()

    def stop_processing():
        if current_thread[0] and current_thread[0].is_alive():
            stop_event.set()
            status_text.insert(tk.END, "Stopping process...\n")
            status_text.see(tk.END)
        # Stop any playing audio
        if current_playing[0]:
            pygame.mixer.music.stop()
            current_playing[0] = None
            status_text.insert(tk.END, "Stopped audio playback.\n")
            status_text.see(tk.END)

    ttk.Button(button_frame, text="Start Processing", command=start_processing).grid(
        row=0, column=0, padx=5
    )
    ttk.Button(button_frame, text="Stop", command=stop_processing).grid(
        row=0, column=1, padx=5
    )
    ttk.Button(button_frame, text="Save JSON", command=save_json).grid(
        row=0, column=2, padx=5
    )

    main_frame.columnconfigure(0, weight=1)
    main_frame.rowconfigure(1, weight=1)
    update_status()
    window.mainloop()

# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Genshin Impact Voice Downloader: Download and transcribe voice data with phonemes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    subparsers = parser.add_subparsers(dest='command', help='Action to perform (leave blank for GUI)')
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument(
        "--character", type=str, required=True, help="Character name (e.g., 'Arlecchino')."
    )
    parent_parser.add_argument(
        "--output_dir", type=str, default=DEFAULT_DATA_DIR, help="Base output directory for character data."
    )
    parser_process = subparsers.add_parser(
        'process',
        help='Download and transcribe voice data.',
        parents=[parent_parser],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser_process.add_argument(
        "--language",
        type=str,
        default="English",
        choices=["English", "Japanese", "Chinese", "Korean"],
        help="Voice language for Wiki category selection."
    )
    parser_process.add_argument(
        "--whisper_model",
        type=str,
        default="base",
        choices=["base", "small", "medium", "large-v2"],
        help="Whisper model size for transcription."
    )
    parser_process.add_argument(
        "--use_segmentation",
        action="store_true",
        help="Use PyAnnote segmentation (requires --hf_token and library install)."
    )
    parser_process.add_argument(
        "--strict_ascii",
        action="store_true",
        help="Force ASCII-only transcriptions (may lose data)."
    )
    parser_process.add_argument(
        "--hf_token",
        type=str,
        default="",
        help="Hugging Face token (required if --use_segmentation)."
    )
    parser_process.add_argument(
        "--skip_wiki_download",
        action="store_true",
        help="Skip downloading audio from the Wiki (only transcribe existing files)."
    )
    parser_process.add_argument(
        "--min_silence_duration",
        type=float,
        default=0.5,
        help="Minimum duration (seconds) of silence to classify audio as silent."
    )

    args = parser.parse_args()

    if args.command is None:
        logging.info("No command specified, launching GUI...")
        main_gui()
    else:
        logging.info(f"Executing command '{args.command}' for character '{args.character}' via CLI...")
        cli_status_queue = queue.Queue()
        if not os.path.isdir(args.output_dir):
            try:
                os.makedirs(args.output_dir, exist_ok=True)
                logging.info(f"Created base output directory: {args.output_dir}")
            except OSError as e:
                logging.error(f"Failed to create output directory {args.output_dir}: {e}")
                exit(1)
        process_character_voices(
            character=args.character,
            language=args.language,
            base_output_dir=args.output_dir,
            download_wiki_audio=not args.skip_wiki_download,
            whisper_model=args.whisper_model,
            use_segmentation=args.use_segmentation,
            hf_token=args.hf_token,
            strict_ascii=args.strict_ascii,
            min_silence_duration=args.min_silence_duration,
            status_queue=cli_status_queue,
        )
