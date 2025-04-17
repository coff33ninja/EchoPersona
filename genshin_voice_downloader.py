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
import playsound
from TTS.api import TTS

# --- Coqui TTS Imports ---
# Use try-except for robustness, especially for optional components
try:
    from TTS.api import TTS
    from TTS.config import load_config
    from TTS.utils.audio import AudioProcessor
    # Trainer related imports might vary slightly based on TTS version
    from trainer import Trainer # Changed import path
    # from TTS.utils.trainer_utils import get_trainer_args # If needed
except ImportError as e:
    print(f"Error importing Coqui TTS components: {e}")
    print("Please ensure Coqui TTS is installed correctly ('pip install TTS')")
    TTS = None
    Trainer = None
    # Add placeholders or exit if core components are missing


try:
    # Assuming voice_tools.py contains your SpeechToText class
    from voice_tools import SpeechToText
except ImportError:
    print("Warning: voice_tools.py not found. Transcription functions will be unavailable.")
    logging.warning("Failed to import SpeechToText from voice_tools.py")
    SpeechToText = None # Define as None if import fails


# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
    encoding="utf-8" # Ensure UTF-8 encoding for logs
)

# --- Constants ---
BASE_DATA_DIR = "voice_datasets" # Default directory for datasets
WIKI_API_URL = "https://genshin-impact.fandom.com/api.php" # Wiki API endpoint
JMP_API_URL_BASE = "https://genshin.jmp.blue" # API for character list 

# --- Available TTS Models ---
# Dictionary mapping user-friendly names to Coqui TTS model identifiers and properties
AVAILABLE_MODELS = {
    "Fast Tacotron2": {
        "model_id": "tts_models/en/ljspeech/tacotron2-DDC", # Coqui TTS model identifier
        "use_pre_trained": True, # Whether to use the pre-trained weights as a base
    },
    "High-Quality VITS": {
        "model_id": "tts_models/multilingual/multi-dataset/vits", # Coqui TTS VITS model
        "use_pre_trained": False, # VITS often trained from scratch or fine-tuned differently
    },
    # Add more models here if needed
}

# --- New Function for Automatic Model Download ---
def ensure_tts_model(model_id, status_queue=None):
    """
    Checks if a TTS model is available locally and downloads it if missing.

    Args:
        model_id (str): The Coqui TTS model identifier (e.g., 'tts_models/en/ljspeech/tacotron2-DDC').
        status_queue (queue.Queue, optional): Queue for GUI status updates.

    Returns:
        bool: True if the model is available or successfully downloaded, False otherwise.
    """
    if TTS is None:
        logging.error("Coqui TTS library is not available.")
        if status_queue:
            status_queue.put("Error: Coqui TTS library not installed.")
        return False

    try:
        # Initialize TTS without loading the model to check availability
        tts = TTS()
        available_models = list(tts.list_models())
        logging.debug(f"Available models: {available_models}")

        if model_id not in available_models:
            logging.error(f"Model {model_id} not supported by this Coqui TTS version.")
            if status_queue:
                status_queue.put(f"Error: Model {model_id} not supported.")
            return False

        # Attempt to initialize the model to trigger download if needed
        logging.info(f"Checking availability of model: {model_id}")
        if status_queue:
            status_queue.put(f"Checking model: {model_id}...")
        
        # Try initializing the model (this will download it if missing)
        tts = TTS(model_name=model_id, progress_bar=True)
        logging.info(f"Model {model_id} is available and ready.")
        if status_queue:
            status_queue.put(f"Model {model_id} ready.")
        return True

    except (TypeError, Exception) as e:
        logging.error(f"Failed to ensure model {model_id}: {e}")
        if status_queue:
            status_queue.put(f"Error downloading model {model_id}: {str(e)}")
        return False

# --- Helper Functions ---

def segment_audio_file(audio_path, output_dir, onset=0.6, offset=0.4, min_duration=2.0, hf_token=""):
    """
    Segments an audio file into smaller chunks based on voice activity detection using pyannote.audio.

    Args:
        audio_path (str): Path to the input audio file.
        output_dir (str): Directory to save the segmented WAV files.
        onset (float): Onset threshold for voice activity detection.
        offset (float): Offset threshold for voice activity detection.
        min_duration (float): Minimum duration (in seconds) for a segment to be kept.
        hf_token (str): Hugging Face authentication token (required for pyannote).

    Returns:
        list: A list of paths to the created segmented WAV files. Returns empty list on failure or if no token provided.
    """
    if not hf_token:
        logging.warning("No Hugging Face token provided for segmentation. Skipping.")
        return []
    try:
        from pyannote.audio import Pipeline # Import pyannote here to make it optional
        pipeline = Pipeline.from_pretrained("pyannote/segmentation", use_auth_token=hf_token)
        logging.info(f"Segmenting audio file: {audio_path}")
        segments = pipeline(audio_path) # Perform segmentation
        audio = AudioSegment.from_file(audio_path) # Load audio with pydub
        os.makedirs(output_dir, exist_ok=True) # Ensure output directory exists
        wav_files = []
        # Iterate through detected segments
        for i, segment in enumerate(segments.get_timeline()):
            start_ms = segment.start * 1000
            end_ms = segment.end * 1000
            duration_sec = (end_ms - start_ms) / 1000
            # Keep segments longer than the minimum duration
            if duration_sec >= min_duration:
                segment_audio = audio[start_ms:end_ms] # Extract segment
                wav_path = os.path.join(output_dir, f"segment_{i}.wav") # Use a descriptive name
                # Export segment as WAV with specific parameters (22050 Hz, mono)
                segment_audio.export(wav_path, format="wav", parameters=["-ar", "22050", "-ac", "1"])
                wav_files.append(wav_path)
        logging.info(f"Segmentation complete. Found {len(wav_files)} segments meeting criteria.")
        return wav_files
    except ImportError:
        logging.error("PyAnnote.audio not installed. Segmentation requires 'pip install pyannote.audio'.")
        return []
    except (TypeError, Exception) as e:
        # Catch specific exceptions if possible (e.g., AuthenticationError)
        logging.error(f"Error during audio segmentation for {audio_path}: {e}")
        return []

def clean_transcript(text, strict_ascii=False):
    """
    Cleans and normalizes a transcript text.

    Args:
        text (str): The input transcript text.
        strict_ascii (bool): If True, remove non-ASCII characters.

    Returns:
        str: The cleaned transcript text.
    """
    if not text:
        return ""
    # Normalize Unicode characters (NFKC recommended for compatibility)
    text = unicodedata.normalize("NFKC", text)
    # Replace potential problematic characters (like '|') and strip whitespace
    text = text.replace("|", " ").strip()
    # Optionally remove non-ASCII characters
    if strict_ascii:
        text = text.encode("ascii", errors="ignore").decode("ascii")
    return text

def is_valid_for_phonemes(text):
    """
    Checks if the text contains characters problematic for phonemizers (e.g., standalone symbols).

    Args:
        text (str): The input text.

    Returns:
        bool: True if the text is likely valid, False otherwise.
    """
    if not text: # Empty string is valid
        return True
    # Check for non-ASCII symbols (Unicode category starting with 'S')
    # This is a heuristic and might need refinement based on the phonemizer used.
    problematic = any(ord(c) > 127 and unicodedata.category(c).startswith("S") for c in text)
    return not problematic

def get_category_files(category):
    """
    Fetches file titles from a specific category on the Genshin Impact Fandom Wiki.

    Args:
        category (str): The category title (e.g., "Category:Arlecchino Voice-Overs").

    Returns:
        list: A list of file titles (e.g., "File:Vo arlecchino about us 01.ogg") found in the category.
              Filters out non-English voice lines based on common naming patterns.
    """
    params = {
        "action": "query",
        "list": "categorymembers",
        "cmtype": "file", # We only want files
        "cmtitle": category,
        "cmlimit": "max", # Get as many as possible per request
        "format": "json",
    }
    files = []
    cmcontinue = None # For handling pagination
    logging.info(f"Fetching files from Wiki category: {category}")
    while True:
        if cmcontinue:
            params["cmcontinue"] = cmcontinue
        try:
            # Make the API request
            response = requests.get(WIKI_API_URL, params=params, timeout=20) # Increased timeout
            response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
            data = response.json()

            # Extract file titles, filtering out non-English based on filename convention
            new_files = [
                member["title"]
                for member in data.get("query", {}).get("categorymembers", [])
                # Regex to exclude Japanese (JA), Korean (KO), Chinese (ZH) voice lines
                if not re.search(r"Vo (JA|KO|ZH)", member.get("title", ""), re.IGNORECASE)
            ]
            files.extend(new_files)

            # Check for continuation token
            if "continue" in data and "cmcontinue" in data["continue"]:
                cmcontinue = data["continue"]["cmcontinue"]
            else:
                break # No more pages
        except requests.exceptions.RequestException as e:
            logging.error(f"Error fetching category files for {category}: {e}")
            break # Stop fetching on error
        except json.JSONDecodeError as e:
            logging.error(f"Error decoding JSON response for {category}: {e}")
            break
    logging.info(f"Found {len(files)} potential files in category '{category}'.")
    return files

def get_file_url(file_title):
    """
    Gets the direct download URL for a file title from the Wiki.

    Args:
        file_title (str): The file title (e.g., "File:Vo arlecchino about us 01.ogg").

    Returns:
        str or None: The direct download URL if found, otherwise None.
    """
    params = {
        "action": "query",
        "titles": file_title,
        "prop": "imageinfo", # We need imageinfo property
        "iiprop": "url", # Specifically the URL
        "format": "json",
    }
    try:
        response = requests.get(WIKI_API_URL, params=params, timeout=15) # Increased timeout
        response.raise_for_status()
        data = response.json()
        # Navigate the JSON structure to find the URL
        pages = data.get("query", {}).get("pages", {})
        page_id = list(pages.keys())[0] # Get the first (and usually only) page ID
        if page_id != "-1" and "imageinfo" in pages[page_id]: # Check if page exists and has imageinfo
             # Ensure imageinfo is a list and has at least one element
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
    """
    Downloads an OGG file, converts it to WAV (22050 Hz, mono), and saves it.

    Args:
        file_url (str): The URL of the OGG file to download.
        output_dir (str): The directory to save the final WAV file.
        file_name (str): The base name for the file (will be sanitized).
        status_queue (queue.Queue, optional): Queue to send status updates for GUI. Defaults to None.

    Returns:
        str or None: The path to the created WAV file on success, otherwise None.
    """
    # Sanitize filename to remove characters invalid for file systems
    safe_file_name = re.sub(r'[\\/*?:"<>|]', "_", file_name)
    # Ensure correct extensions
    ogg_file_name = safe_file_name if safe_file_name.lower().endswith(".ogg") else f"{safe_file_name}.ogg"
    wav_file_name = os.path.splitext(ogg_file_name)[0] + ".wav" # More robust way to change extension

    ogg_path = os.path.join(output_dir, "temp_" + ogg_file_name) # Download OGG to a temporary location
    wav_path = os.path.join(output_dir, wav_file_name) # Final WAV path

    # Skip if WAV already exists
    if os.path.exists(wav_path):
        logging.info(f"Skipping existing WAV: {wav_path}")
        if status_queue:
            status_queue.put(f"Skipped: {wav_file_name}")
        return wav_path

    try:
        # --- Download ---
        if status_queue:
            status_queue.put(f"Downloading {ogg_file_name}...")
        response = requests.get(file_url, timeout=45) # Increased timeout for download
        response.raise_for_status()

        # Ensure the temporary download directory exists
        os.makedirs(os.path.dirname(ogg_path), exist_ok=True)
        with open(ogg_path, "wb") as f:
            f.write(response.content)
        logging.info(f"Downloaded OGG: {ogg_path}")

        # --- Convert ---
        if status_queue:
            status_queue.put(f"Converting {ogg_file_name} to WAV...")
        # Use ffmpeg for conversion. Ensure ffmpeg is in the system PATH.
        # -y: Overwrite output files without asking
        # -i: Input file
        # -ar: Audio sample rate (22050 Hz)
        # -ac: Audio channels (1 for mono)
        # -loglevel error: Suppress verbose ffmpeg output, show only errors
        process = subprocess.run(
            ["ffmpeg", "-y", "-i", ogg_path, "-ar", "22050", "-ac", "1", wav_path, "-loglevel", "error"],
            check=True, # Raise an exception if ffmpeg fails
            text=True, # Capture output as text (though loglevel error minimizes it)
            capture_output=True # Capture stdout/stderr
        )
        # Log ffmpeg output if needed, even if loglevel is error
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
        logging.error(f"FFmpeg stderr: {e.stderr}") # Log ffmpeg error output
        if status_queue:
            status_queue.put(f"Conversion Error: {ogg_file_name}")
        return None
    except (TypeError, Exception) as e:
        logging.error(f"Error processing {ogg_file_name} (URL: {file_url}): {e}")
        if status_queue:
            status_queue.put(f"Error with {ogg_file_name}")
        return None
    finally:
        # Clean up the temporary OGG file regardless of success or failure
        if os.path.exists(ogg_path):
            try:
                os.remove(ogg_path)
            except OSError as e:
                logging.warning(f"Could not remove temporary OGG file {ogg_path}: {e}")


def fetch_character_list_from_api():
    """
    Fetches a list of character names from the JMP Blue Genshin API.

    Returns:
        list: A sorted list of unique character names, or an empty list on error.
    """
    try:
        # Get the list of character slugs (identifiers)
        response = requests.get(f"{JMP_API_URL_BASE}/characters", timeout=15) # Increased timeout
        response.raise_for_status()
        character_slugs = response.json()

        character_names = []
        logging.info(f"Fetching details for {len(character_slugs)} characters...")
        # Fetch details for each character to get their name
        # Consider adding a progress bar if this takes long
        for slug in character_slugs:
            char_response = requests.get(f"{JMP_API_URL_BASE}/characters/{slug}", timeout=5)
            char_response.raise_for_status()
            details = char_response.json()
            if "name" in details:
                character_names.append(details["name"])
        return sorted(list(set(character_names)))
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching character list from API: {e}")
        return [] # Return empty list on error

def is_silent_audio(file_path, silence_threshold=-50.0, chunk_size=10):
    """
    Checks if a WAV file is likely silent or very quiet.

    Args:
        file_path (str): Path to the WAV file.
        silence_threshold (float): The dBFS threshold below which audio is considered silent. Defaults to -50.0.
        chunk_size (int): Size of chunks (in ms) to check for silence. Defaults to 10.

    Returns:
        bool: True if the audio's maximum dBFS is below the threshold, False otherwise.
              Returns True if an error occurs during processing (treats errors as silence).
    """
    try:
        audio = AudioSegment.from_wav(file_path)
        # Check if the loudest part of the audio is below the silence threshold
        is_silent = audio.max_dBFS < silence_threshold
        if is_silent:
             logging.info(f"Audio file {file_path} detected as silent (max dBFS: {audio.max_dBFS:.2f} < {silence_threshold}).")
        return is_silent
    except FileNotFoundError:
        logging.error(f"File not found while checking silence: {file_path}")
        return True # Treat as silent if file doesn't exist
    except (TypeError, Exception) as e:
        logging.error(f"Error checking silence for {file_path}: {e}")
        return True # Assume silent if error occurs to avoid processing potentially bad files


def clean_metadata_file(metadata_path):
    """
    Cleans a metadata CSV file, ensuring it has the correct header and format (text|audio_file|speaker_id).
    Removes lines with incorrect column counts or empty essential fields.

    Args:
        metadata_path (str): Path to the metadata CSV file.

    Returns:
        bool: True if the file was cleaned successfully and contains at least one valid data entry, False otherwise.
    """
    cleaned_lines = []
    invalid_lines_removed = 0
    expected_header = ["text", "audio_file", "speaker_id"]

    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        if not lines:
            logging.error(f"Metadata file {metadata_path} is empty.")
            return False

        # Process header
        header_fields = lines[0].strip().split("|")
        if header_fields != expected_header:
            logging.error(f"Invalid header in {metadata_path}: Found {header_fields}, expected {expected_header}")
            return False
        cleaned_lines.append("|".join(expected_header) + "\n") # Write the correct header

        # Process data lines
        for i, line in enumerate(lines[1:], 2): # Start from line 2
            original_line = line.strip()
            if not original_line: # Skip empty lines
                invalid_lines_removed += 1
                continue

            fields = original_line.split("|")
            # Check if exactly 3 fields exist and none are empty strings
            if len(fields) == 3 and all(field.strip() for field in fields):
                # Rejoin the first 3 fields to ensure no extra columns are kept
                cleaned_lines.append("|".join(fields[:3]) + "\n")
            else:
                logging.warning(f"Removing invalid metadata line {i}: '{original_line}' from {metadata_path}")
                invalid_lines_removed += 1

        if invalid_lines_removed > 0:
            logging.warning(f"Removed {invalid_lines_removed} invalid lines from {metadata_path}.")

        # Check if any data lines remain
        if len(cleaned_lines) <= 1:
            logging.error(f"No valid data entries found in {metadata_path} after cleaning.")
            # Optionally remove the file if it only contains the header
            # os.remove(metadata_path)
            return False

        # Overwrite the original file with cleaned content
        with open(metadata_path, "w", encoding="utf-8", newline="") as f:
            f.writelines(cleaned_lines)

        logging.info(f"Cleaned {metadata_path}: {len(cleaned_lines)-1} valid entries remain.")
        return True # Indicates success and data presence

    except FileNotFoundError:
        logging.error(f"Metadata file not found for cleaning: {metadata_path}")
        return False
    except (TypeError, Exception) as e:
        logging.error(f"Error cleaning metadata file {metadata_path}: {e}")
        return False


def transcribe_character_audio(
    character_output_dir,
    whisper_model="base",
    use_segmentation=False,
    hf_token="",
    strict_ascii=False,
    status_queue=None,
):
    """
    Transcribes WAV audio files for a character using Whisper via SpeechToText.
    Handles existing metadata, moves files, checks for silence, and cleans the final metadata.

    Args:
        character_output_dir (str): The main directory for the character's data.
        whisper_model (str): The Whisper model size to use (e.g., "base", "large-v2").
        use_segmentation (bool): Whether to segment long audio files first.
        hf_token (str): Hugging Face token (required for segmentation).
        strict_ascii (bool): Whether to enforce strict ASCII in transcripts.
        status_queue (queue.Queue, optional): Queue for GUI status updates. Defaults to None.

    Returns:
        bool: True if transcription process completes and produces valid metadata, False otherwise.
    """
    if SpeechToText is None:
        logging.error("Transcription unavailable: SpeechToText class not imported correctly.")
        if status_queue:
            status_queue.put("Transcription unavailable (SpeechToText missing).")
        return False

    metadata_path = os.path.join(character_output_dir, "metadata.csv")
    wavs_dir = os.path.join(character_output_dir, "wavs") # Standard subdirectory for WAVs
    silent_dir = os.path.join(character_output_dir, "silent_files") # Directory for silent/failed files
    temp_dir = os.path.join(character_output_dir, "temp_audio") # Temporary dir for segmentation source

    os.makedirs(wavs_dir, exist_ok=True)
    os.makedirs(silent_dir, exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True)

    existing_transcribed_files = set()
    # Load existing metadata if valid
    if os.path.exists(metadata_path) and validate_metadata_layout(metadata_path):
        try:
            # Use pandas for easier handling of CSV
            df_existing = pd.read_csv(metadata_path, sep="|", encoding="utf-8", on_bad_lines='skip')
            # Ensure required columns exist before proceeding
            if all(col in df_existing.columns for col in ["text", "audio_file", "speaker_id"]):
                 # Add the base filename from the 'audio_file' column (e.g., "vo_char_01.wav")
                 existing_transcribed_files = set(df_existing['audio_file'].apply(lambda x: os.path.basename(str(x))))
                 logging.info(f"Loaded {len(existing_transcribed_files)} existing entries from {metadata_path}")
            else:
                 logging.warning(f"Metadata file {metadata_path} is missing required columns. Treating as empty.")
                 existing_transcribed_files = set()

        except pd.errors.EmptyDataError:
             logging.info(f"Metadata file {metadata_path} is empty. Starting fresh.")
             existing_transcribed_files = set()
        except (TypeError, Exception) as e:
            logging.error(f"Error reading existing metadata {metadata_path}: {e}. Starting fresh.")
            existing_transcribed_files = set() # Reset on error
            # Optionally backup the problematic metadata file here
            backup_file(metadata_path, "metadata_read_error_backup")
            if os.path.exists(metadata_path):
                os.remove(metadata_path) # Remove bad file

    else:
        logging.info(f"Invalid or missing metadata at {metadata_path}. Starting fresh.")
        existing_transcribed_files = set()
        # Clean up potentially invalid metadata file if it exists
        if os.path.exists(metadata_path):
             backup_file(metadata_path, "metadata_invalid_layout_backup")
             os.remove(metadata_path)


    # --- File Discovery and Preparation ---
    files_to_process = []
    # 1. Move top-level WAVs to temp_dir for potential segmentation/processing
    for item in os.listdir(character_output_dir):
        item_path = os.path.join(character_output_dir, item)
        if item.lower().endswith(".wav") and os.path.isfile(item_path):
            try:
                shutil.move(item_path, os.path.join(temp_dir, item))
                logging.info(f"Moved {item} to temp directory for processing.")
            except (TypeError, Exception) as e:
                logging.warning(f"Could not move {item} to temp directory: {e}")

    # 2. Process files in temp_dir
    for file in os.listdir(temp_dir):
         if file.lower().endswith(".wav"):
             temp_path = os.path.join(temp_dir, file)
             # Skip if already transcribed
             if file in existing_transcribed_files:
                  logging.info(f"Skipping already transcribed file: {file}")
                  # Move directly to wavs_dir if skipped
                  try:
                      shutil.move(temp_path, os.path.join(wavs_dir, file))
                  except (TypeError, Exception) as e:
                      logging.warning(f"Could not move skipped file {file} to wavs dir: {e}")
                  continue

             if use_segmentation and hf_token:
                 if status_queue:
                    status_queue.put(f"Segmenting: {file}")
                 segmented_files = segment_audio_file(temp_path, wavs_dir, hf_token=hf_token)
                 if segmented_files:
                     # Add only the base filenames of new segments
                     files_to_process.extend([os.path.basename(f) for f in segmented_files if os.path.basename(f) not in existing_transcribed_files])
                     try:
                         os.remove(temp_path) # Remove original after successful segmentation
                     except OSError as e:
                          logging.warning(f"Could not remove original segmented file {temp_path}: {e}")
                 else:
                     # Segmentation failed or produced no valid segments, move original to wavs_dir
                     logging.warning(f"Segmentation failed for {file}. Moving original to wavs.")
                     try:
                         shutil.move(temp_path, os.path.join(wavs_dir, file))
                         files_to_process.append(file) # Add original file for transcription attempt
                     except (TypeError, Exception) as e:
                          logging.warning(f"Could not move original file {file} to wavs dir after failed segmentation: {e}")

             else:
                 # No segmentation, move directly to wavs_dir and add to list
                 try:
                     shutil.move(temp_path, os.path.join(wavs_dir, file))
                     files_to_process.append(file)
                 except (TypeError, Exception) as e:
                      logging.warning(f"Could not move file {file} to wavs dir: {e}")

    # 3. Check wavs_dir for any files missed (e.g., if script was interrupted)
    for file in os.listdir(wavs_dir):
         if file.lower().endswith(".wav") and file not in existing_transcribed_files and file not in files_to_process:
             files_to_process.append(file)
             logging.info(f"Found untranscribed file in wavs_dir: {file}")


    # Clean up empty temp_dir
    if os.path.exists(temp_dir) and not os.listdir(temp_dir):
        try:
            os.rmdir(temp_dir)
        except OSError as e:
            logging.warning(f"Could not remove empty temp directory {temp_dir}: {e}")


    if not files_to_process:
        logging.info("No new audio files found to transcribe.")
        if status_queue:
            status_queue.put("No new files to transcribe.")
        # Ensure metadata is clean even if no new files were added
        if os.path.exists(metadata_path):
            return clean_metadata_file(metadata_path)
        else:
            # Create empty metadata if it doesn't exist and no files processed
            try:
                with open(metadata_path, "w", encoding="utf-8", newline="") as mf:
                     writer = csv.writer(mf, delimiter='|', lineterminator='\n')
                     writer.writerow(["text", "audio_file", "speaker_id"])
                logging.info(f"Created empty metadata file: {metadata_path}")
                return True # Considered success (empty but valid state)
            except (TypeError, Exception) as e:
                 logging.error(f"Failed to create empty metadata file {metadata_path}: {e}")
                 return False


    logging.info(f"Found {len(files_to_process)} new WAV files to transcribe using Whisper {whisper_model}...")
    transcribed_count = 0
    failed_count = 0
    silent_count = 0
    # Open metadata in append mode if it exists and has data, otherwise write mode (with header)
    file_mode = "a" if existing_transcribed_files else "w"

    try:
        with open(metadata_path, file_mode, encoding="utf-8", newline="") as mf:
            writer = csv.writer(mf, delimiter='|', lineterminator='\n', quoting=csv.QUOTE_MINIMAL)
            if file_mode == "w":
                writer.writerow(["text", "audio_file", "speaker_id"]) # Write header only if creating new file

            # Use tqdm for progress bar
            for file in tqdm(files_to_process, desc="Transcribing Audio"):
                wav_path = os.path.join(wavs_dir, file) # Path to the file in the wavs directory

                if not os.path.exists(wav_path):
                    logging.warning(f"File {file} listed for processing but not found in {wavs_dir}. Skipping.")
                    failed_count += 1
                    continue

                if status_queue:
                    status_queue.put(f"Checking: {file}")

                # --- Silence Check ---
                if is_silent_audio(wav_path):
                    logging.warning(f"Moving silent file {file} to {silent_dir}")
                    try:
                        shutil.move(wav_path, os.path.join(silent_dir, file))
                        silent_count += 1
                        if status_queue:
                            status_queue.put(f"Moved silent file: {file}")
                    except Exception as move_err:
                        logging.error(f"Failed to move silent file {file}: {move_err}")
                        failed_count += 1 # Count as failure if move fails
                    continue # Skip transcription for silent files

                # --- Transcription Attempt ---
                transcript = None
                for attempt in range(2): # Retry transcription once on failure
                    try:
                        if status_queue:
                            status_queue.put(f"Transcribing: {file} (Attempt {attempt+1})")

                        # Initialize SpeechToText for the file
                        stt = SpeechToText(
                            use_microphone=False,
                            audio_file=wav_path,
                            engine="whisper",
                            whisper_model_size=whisper_model,
                        )
                        # Process audio to get transcript
                        audio_transcript = stt.process_audio(language="en") # Assuming English
                        cleaned_transcript = clean_transcript(audio_transcript, strict_ascii=strict_ascii)

                        # Validate transcript
                        if cleaned_transcript and is_valid_for_phonemes(cleaned_transcript):
                            transcript = cleaned_transcript
                            break # Success, exit retry loop
                        else:
                            logging.warning(f"Invalid or empty transcription for {file} on attempt {attempt+1}. Transcript: '{audio_transcript}' -> '{cleaned_transcript}'")
                            # Don't immediately move on first failure, allow retry

                    except (TypeError, Exception) as e:
                        logging.error(f"Transcription error for {file} on attempt {attempt+1}: {e}")
                        # Allow retry loop to continue

                # --- Handle Transcription Result ---
                if transcript:
                    # Write successful transcription to metadata
                    relative_audio_path = f"wavs/{file}" # Store relative path
                    writer.writerow([transcript, relative_audio_path, "speaker_1"]) # Assuming single speaker
                    transcribed_count += 1
                    if status_queue:
                        status_queue.put(f"Transcribed: {file}")
                else:
                    # Transcription failed after retries or yielded invalid result
                    logging.warning(f"Moving failed transcription file {file} to {silent_dir}")
                    try:
                        shutil.move(wav_path, os.path.join(silent_dir, file))
                        failed_count += 1
                        if status_queue:
                            status_queue.put(f"Moved failed file: {file}")
                    except Exception as move_err:
                        logging.error(f"Failed to move failed transcription file {file}: {move_err}")
                        # File remains in wavs dir but wasn't transcribed

    except IOError as e:
         logging.error(f"Error writing to metadata file {metadata_path}: {e}")
         return False # Cannot proceed if metadata cannot be written
    except (TypeError, Exception) as e:
         logging.exception(f"An unexpected error occurred during the transcription loop: {e}")
         return False # General failure


    logging.info(
        f"Transcription complete. Successful: {transcribed_count}, Failed: {failed_count}, Silent/Moved: {silent_count}."
    )
    if status_queue:
        status_queue.put(
            f"Transcription complete. OK: {transcribed_count}, Fail: {failed_count}, Silent: {silent_count}."
        )

    # Final cleanup and validation split
    if os.path.exists(metadata_path):
        final_clean_success = clean_metadata_file(metadata_path)
        if final_clean_success:
            valid_csv_path = generate_valid_csv(metadata_path, valid_ratio=0.1) # Use smaller validation set (e.g., 10%)
            return valid_csv_path is not None # Return True if valid.csv was generated
        else:
            logging.error("Final metadata cleaning failed.")
            return False
    else:
        logging.error("Metadata file does not exist after transcription process.")
        return False


def validate_metadata_existence(character_dir):
    """Checks if metadata.csv exists and has more than just a header line."""
    metadata_path = os.path.join(character_dir, "metadata.csv")
    if not os.path.exists(metadata_path):
        logging.error(f"Metadata file missing: {metadata_path}")
        return False
    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        # Check if file has at least 2 lines (header + one data line)
        if len(lines) < 2:
            logging.error(f"Metadata file {metadata_path} exists but has no data entries (only header or empty).")
            return False
        return True
    except (TypeError, Exception) as e:
        logging.error(f"Error checking metadata existence {metadata_path}: {e}")
        return False


def validate_metadata_for_training(metadata_path):
    """
    Validates if a metadata CSV file is suitable for training (correct format, columns, non-empty).

    Args:
        metadata_path (str): Path to the metadata file (e.g., metadata.csv or valid.csv).

    Returns:
        bool: True if the metadata is valid for training, False otherwise.
    """
    if not os.path.exists(metadata_path):
        logging.error(f"Metadata file not found for validation: {metadata_path}")
        return False

    try:
        line_count = 0
        invalid_lines = []
        expected_header = ["text", "audio_file", "speaker_id"]

        with open(metadata_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter='|')
            # Read header
            try:
                header = next(reader)
                line_count += 1
                if header != expected_header:
                    logging.error(f"Invalid header in {metadata_path}: Found {header}, expected {expected_header}")
                    return False
            except StopIteration:
                logging.error(f"Metadata file {metadata_path} is empty.")
                return False # Empty file

            # Read data lines
            for i, row in enumerate(reader, 2): # Start line count from 2
                line_count += 1
                # Check for exactly 3 columns and ensure none are empty/whitespace only
                if len(row) != 3 or not all(field.strip() for field in row):
                    invalid_lines.append((i, "|".join(row))) # Log the problematic row

        # Check if only header was present
        if line_count < 2:
            logging.error(f"Metadata file {metadata_path} has no data entries (only header).")
            return False

        if invalid_lines:
            # Log only the first few invalid lines to avoid spamming logs
            max_log = 5
            logging.error(f"Found {len(invalid_lines)} invalid metadata entries (wrong columns or empty fields) in {metadata_path}:")
            for i, (line_num, line_content) in enumerate(invalid_lines):
                 if i < max_log:
                     logging.error(f"  - Line {line_num}: '{line_content}'")
                 elif i == max_log:
                     logging.error("  - ... (further invalid lines suppressed)")
            return False # Metadata is invalid

        # If we reach here, the file exists, has a correct header, data lines, and no invalid lines found
        logging.info(f"Metadata validation successful for: {metadata_path} ({line_count-1} data entries)")
        return True

    except FileNotFoundError: # Should be caught by os.path.exists, but for safety
        logging.error(f"Metadata file not found during validation: {metadata_path}")
        return False
    except csv.Error as e:
         logging.error(f"CSV parsing error in {metadata_path}, line {reader.line_num}: {e}")
         return False
    except (TypeError, Exception) as e:
        logging.error(f"Unexpected error validating metadata {metadata_path}: {e}")
        return False


def process_character_voices(
    character,
    language, # Currently only used for category selection
    base_output_dir,
    download_wiki_audio=True,
    whisper_model="base",
    use_segmentation=False,
    hf_token="",
    strict_ascii=False,
    status_queue=None,
    stop_event=None, # Event to signal cancellation
    # Training params (batch_size, num_epochs, learning_rate) are removed as they belong to training step
):
    """
    Main function to process a character's voice data: download, (optionally) segment, and transcribe.

    Args:
        character (str): The character name.
        language (str): Language (used for selecting Wiki category).
        base_output_dir (str): The base directory where character folders will be created.
        download_wiki_audio (bool): Whether to download audio from the Wiki.
        whisper_model (str): Whisper model size for transcription.
        use_segmentation (bool): Whether to use audio segmentation.
        hf_token (str): Hugging Face token (needed for segmentation).
        strict_ascii (bool): Whether to enforce strict ASCII transcripts.
        status_queue (queue.Queue, optional): Queue for GUI status updates. Defaults to None.
        stop_event (threading.Event, optional): Event to signal cancellation. Defaults to None.

    Returns:
        str or None: The path to the character's processed data folder on success, None on failure or cancellation.
    """
    if stop_event and stop_event.is_set():
        logging.info("Processing cancelled before starting.")
        if status_queue:
            status_queue.put("Processing cancelled.")
        return None

    # Sanitize character name for folder creation
    safe_character_name = re.sub(r'[\\/*?:"<>|]', "_", character)
    character_folder = os.path.join(base_output_dir, safe_character_name)
    wavs_folder = os.path.join(character_folder, "wavs") # Define standard wavs subfolder
    os.makedirs(wavs_folder, exist_ok=True) # Ensure base and wavs folder exist

    # --- Download Step ---
    if download_wiki_audio:
        logging.info(f"--- Starting Wiki Download for {character} ---")
        if status_queue:
            status_queue.put(f"Downloading Wiki audio for {character}...")
        # Determine Wiki categories based on language (simple approach)
        # Might need refinement for more complex language/character name interactions
        categories = (
            [
                f"Category:{character} Voice-Overs", # General category
                f"Category:English {character} Voice-Overs", # Specific English category
            ]
            if language == "English"
            else [f"Category:{language} {character} Voice-Overs"] # Assumes format for other languages
        )

        files_to_download = []
        for category in categories:
            if stop_event and stop_event.is_set():
                logging.info("Download cancelled during category fetching.")
                if status_queue:
                    status_queue.put("Download cancelled.")
                return None
            files_to_download.extend(get_category_files(category))

        # Remove duplicates and sort
        unique_files = sorted(list(set(files_to_download)))
        logging.info(f"Found {len(unique_files)} unique potential audio files from Wiki categories.")

        downloaded_count = 0
        failed_count = 0
        skipped_count = 0

        # Download and convert each unique file
        for i, file_title in enumerate(tqdm(unique_files, desc=f"Downloading {character} Audio")):
            if stop_event and stop_event.is_set():
                logging.info("Download cancelled during file processing.")
                if status_queue:
                    status_queue.put("Download cancelled.")
                return None

            # Extract filename from title (e.g., "File:Vo xiao demo 01.ogg" -> "Vo xiao demo 01.ogg")
            match = re.match(r"File:(.*)", file_title, re.IGNORECASE)
            if not match:
                logging.warning(f"Could not parse file name from title: {file_title}")
                failed_count += 1
                continue
            file_name = match.group(1).strip()

            # Get the direct download URL
            file_url = get_file_url(file_title)
            if file_url:
                # Download and convert, saving directly to the wavs_folder
                wav_file_path = download_and_convert(
                    file_url, wavs_folder, file_name, status_queue=status_queue
                )
                if wav_file_path:
                    # Check if the file already existed (download_and_convert returns path even if skipped)
                    # A better check might be needed if status_queue is None
                    is_skipped = False
                    if status_queue:
                        try:
                            # Check recent messages in queue (heuristic)
                            q_list = list(status_queue.queue)[-5:] # Check last 5 messages
                            if any(f"Skipped: {os.path.basename(wav_file_path)}" in msg for msg in q_list):
                                is_skipped = True
                        except Exception: # Ignore queue errors
                            pass
                    if is_skipped:
                         skipped_count += 1
                    else:
                         downloaded_count += 1
                else:
                    failed_count += 1 # download_and_convert logs the error
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

    # --- Transcription Step ---
    if stop_event and stop_event.is_set():
        logging.info("Processing cancelled before transcription.")
        if status_queue:
            status_queue.put("Processing cancelled.")
        return None

    logging.info(f"--- Starting Transcription for {character} ---")
    if status_queue:
        status_queue.put(f"Starting transcription for {character}...")

    # Call the transcription function which handles metadata, silence check etc.
    transcription_success = transcribe_character_audio(
        character_folder, # Pass the main character folder
        whisper_model,
        use_segmentation,
        hf_token,
        strict_ascii,
        status_queue=status_queue,
        # Note: transcribe_character_audio now handles its own stop_event check internally if needed
    )

    if stop_event and stop_event.is_set(): # Check again after transcription attempt
        logging.info("Processing cancelled during/after transcription.")
        if status_queue:
            status_queue.put("Processing cancelled.")
        return None

    if not transcription_success:
        logging.error(f"Transcription process failed or produced no valid data for {character}.")
        if status_queue:
            status_queue.put(f"Transcription failed for {character}.")
        # Consider keeping downloaded files even if transcription fails? Or clean up?
        return None # Indicate failure

    # --- Final Validation ---
    # Transcription function now handles final cleaning and validation split
    metadata_path = os.path.join(character_folder, "metadata.csv")
    valid_metadata_path = os.path.join(character_folder, "valid.csv")

    if not os.path.exists(metadata_path) or not os.path.exists(valid_metadata_path):
        logging.error(f"Metadata or validation file missing after transcription for {character}. Check logs.")
        if status_queue:
            status_queue.put("Metadata generation failed post-transcription.")
        return None

    # Final check on the generated files
    if not validate_metadata_for_training(metadata_path) or not validate_metadata_for_training(valid_metadata_path):
         logging.error(f"Final validation failed for metadata/valid.csv for {character}.")
         if status_queue:
            status_queue.put("Final metadata validation failed.")
         return None


    logging.info(f"--- Successfully processed voices for {character}. Ready for training. ---")
    if status_queue:
        status_queue.put(f"Processing complete for {character}.")

    return character_folder # Return the path to the processed character folder


def validate_metadata_layout(metadata_path):
    """
    Validates the basic layout (header, column count) of a metadata file.
    Less strict than validate_metadata_for_training, used for initial checks.

    Args:
        metadata_path (str): Path to the metadata file.

    Returns:
        bool: True if the layout seems correct (header, 3 columns), False otherwise.
    """
    if not os.path.exists(metadata_path):
         logging.warning(f"Metadata layout check: File not found {metadata_path}")
         return False
    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter='|')
            # Check header
            try:
                header = next(reader)
                expected_header = ["text", "audio_file", "speaker_id"]
                if header != expected_header:
                    logging.error(f"Invalid header in {metadata_path}: Found {header}, expected {expected_header}")
                    return False
            except StopIteration:
                logging.error(f"Metadata file {metadata_path} is empty.")
                return False # Empty file

            # Check first few data lines for column count (optional optimization)
            for i, row in enumerate(reader):
                 if i >= 5:
                    break # Only check first 5 data lines for layout
                 if len(row) != 3:
                      logging.error(f"Incorrect column count ({len(row)}) found at line {i+2} in {metadata_path}. Expected 3.")
                      return False

        return True # Layout seems correct (header and column count)

    except FileNotFoundError: # Should be caught by os.path.exists
        logging.error(f"Metadata file not found during layout validation: {metadata_path}")
        return False
    except csv.Error as e:
         logging.error(f"CSV parsing error during layout validation of {metadata_path}: {e}")
         return False
    except (TypeError, Exception) as e:
        logging.error(f"Error validating layout of {metadata_path}: {e}")
        return False


def validate_training_prerequisites(character_dir, config_path):
    """
    Validates all necessary files and formats before starting TTS training.

    Args:
        character_dir (str): Path to the character's data directory.
        config_path (str): Path to the training configuration JSON file.

    Returns:
        bool: True if all prerequisites are met, False otherwise.
    """
    logging.info(f"--- Validating Training Prerequisites for {os.path.basename(character_dir)} ---")
    metadata_path = os.path.join(character_dir, "metadata.csv")
    valid_metadata_path = os.path.join(character_dir, "valid.csv")
    wavs_dir = os.path.join(character_dir, "wavs")

    checks = {
        "Config File": (os.path.exists, config_path),
        "WAVs Directory": (os.path.isdir, wavs_dir), # Check if it's a directory
        "Train Metadata": (validate_metadata_for_training, metadata_path),
        "Validation Metadata": (validate_metadata_for_training, valid_metadata_path),
    }

    all_passed = True
    for name, (check_func, path) in checks.items():
        try:
            if not check_func(path):
                # Specific error logged within the check_func (e.g., validate_metadata_for_training)
                logging.error(f"Prerequisite Check Failed: {name} validation failed for path: {path}")
                all_passed = False
            else:
                # Only log pass for existence checks, validation functions log their own success
                if check_func in [os.path.exists, os.path.isdir]:
                    logging.info(f"Prerequisite Check Passed: {name} exists at {path}")
        except (TypeError, Exception) as e:
            logging.error(f"Error during prerequisite check '{name}' for path {path}: {e}")
            all_passed = False

    if not all_passed:
        logging.error("One or more prerequisite checks failed.")
        return False

    # Check if WAVs directory is empty
    try:
        if not os.listdir(wavs_dir):
            logging.error(f"Prerequisite Check Failed: WAVs directory {wavs_dir} is empty.")
            return False
        else:
            logging.info("Prerequisite Check Passed: WAVs directory is not empty.")
    except FileNotFoundError:
        # This case should be caught by os.path.isdir check above, but added for safety
        logging.error(f"Prerequisite Check Failed: WAVs directory {wavs_dir} not found.")
        return False
    except (TypeError, Exception) as e:
        logging.error(f"Error checking contents of WAVs directory {wavs_dir}: {e}")
        return False

    # Check if WAV files listed in metadata exist
    logging.info("Validating WAV file references in metadata...")
    try:
        # Check train metadata
        df_train = pd.read_csv(metadata_path, sep="|", encoding="utf-8", usecols=["audio_file"], on_bad_lines='skip')
        missing_train_files = [
            fpath for fpath in df_train["audio_file"]
            if not os.path.exists(os.path.join(character_dir, str(fpath)))
        ]
        # Check validation metadata
        df_valid = pd.read_csv(valid_metadata_path, sep="|", encoding="utf-8", usecols=["audio_file"], on_bad_lines='skip')
        missing_valid_files = [
            fpath for fpath in df_valid["audio_file"]
            if not os.path.exists(os.path.join(character_dir, str(fpath)))
        ]

        missing_files = missing_train_files + missing_valid_files

        if missing_files:
            max_log = 10
            logging.error(f"Prerequisite Check Failed: {len(missing_files)} WAV files referenced in metadata do not exist:")
            for i, fpath in enumerate(missing_files):
                if i < max_log:
                    logging.error(f"  - Missing: {os.path.join(character_dir, str(fpath))}")
                elif i == max_log:
                    logging.error("  - ... (further missing files suppressed)")
            return False
        else:
            logging.info("Prerequisite Check Passed: All WAV files referenced in metadata exist.")

    except pd.errors.EmptyDataError:
        logging.error(f"Prerequisite Check Failed: Metadata file {metadata_path} or {valid_metadata_path} is empty.")
        return False
    except KeyError:
        logging.error(f"Prerequisite Check Failed: Metadata file {metadata_path} or {valid_metadata_path} missing 'audio_file' column.")
        return False
    except (TypeError, Exception) as e:
        logging.error(f"Error validating WAV file references in metadata: {e}")
        return False

    logging.info("--- All training prerequisites validated successfully. ---")
    return True


def update_character_config(
    character,
    base_output_dir,
    selected_model="Fast Tacotron2",
    batch_size=16,
    num_epochs=100,
    learning_rate=0.0001,
    status_queue=None,  # Added for status updates
):
    """
    Creates or updates the Coqui TTS JSON configuration file for a character.
    Ensures the selected model is available before generating the config.

    Args:
        character (str): The character name.
        base_output_dir (str): The base directory containing the character's folder.
        selected_model (str): The user-friendly name of the model from AVAILABLE_MODELS.
        batch_size (int): Training batch size.
        num_epochs (int): Number of training epochs.
        learning_rate (float): Training learning rate.
        status_queue (queue.Queue, optional): Queue for GUI status updates.

    Returns:
        str or None: The path to the generated/updated config file on success, None on failure.
    """
    safe_character_name = re.sub(r'[\\/*?:"<>|]', "_", character)
    character_folder = os.path.join(base_output_dir, safe_character_name)
    metadata_path_abs = os.path.abspath(os.path.join(character_folder, "metadata.csv"))
    valid_metadata_path_abs = os.path.abspath(
        os.path.join(character_folder, "valid.csv")
    )
    output_path_abs = os.path.abspath(
        os.path.join(base_output_dir, "tts_train_output", safe_character_name)
    )
    config_path = os.path.join(character_folder, f"{safe_character_name}_config.json")

    # Validate selected model
    if selected_model not in AVAILABLE_MODELS:
        logging.error(
            f"Invalid model selected: '{selected_model}'. Available: {list(AVAILABLE_MODELS.keys())}"
        )
        if status_queue:
            status_queue.put(f"Error: Invalid model {selected_model}.")
        return None
    model_data = AVAILABLE_MODELS[selected_model]
    model_id = model_data["model_id"]

    # Ensure the model is downloaded or available
    if model_data["use_pre_trained"]:
        if not ensure_tts_model(model_id, status_queue=status_queue):
            logging.error(f"Cannot proceed without model {model_id}.")
            return None

    # Define default audio parameters
    default_audio_config = {
        "sample_rate": 22050,
        "fft_size": 1024,
        "win_length": 1024,
        "hop_length": 256,
        "num_mels": 80,
        "mel_fmin": 0.0,
        "mel_fmax": 8000.0,
        "preemphasis": 0.97,
        "ref_level_db": 20,
        "min_level_db": -100,
        "signal_norm": True,
        "stats_path": None,
        "trim_db": 60,
    }

    # Define default character settings
    default_chars_config = {
        "use_phonemes": False,
        "phonemizer": None,
        "phoneme_language": None,
        "pad": "<PAD>",
        "eos": "<EOS>",
        "bos": "<BOS>",
        "blank": "<BLNK>",
        "characters": "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,!?'-() ",
    }

    # Construct config dictionary
    config = {
        "model": "tacotron2" if "tacotron2" in model_id.lower() else "vits",
        "run_name": f"{safe_character_name}_tts",
        "audio": default_audio_config,
        "characters": default_chars_config,
        "train_csv": metadata_path_abs,
        "eval_csv": valid_metadata_path_abs,
        "output_path": output_path_abs,
        "batch_size": batch_size,
        "epochs": num_epochs,
        "learning_rate": learning_rate,
        "use_pre_trained": model_data["use_pre_trained"],
        "pre_trained_model": model_id if model_data["use_pre_trained"] else None,
    }

    # Save config file
    try:
        os.makedirs(character_folder, exist_ok=True)
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=4)
        logging.info(f"TTS configuration file updated/created: {config_path}")
        if status_queue:
            status_queue.put(f"Config created: {os.path.basename(config_path)}")
        return config_path
    except (TypeError, Exception) as e:
        logging.error(f"Error saving config file {config_path}: {e}")
        if status_queue:
            status_queue.put(f"Error creating config: {str(e)}")
        return None


def generate_valid_csv(metadata_path, valid_ratio=0.1):
    """
    Generates valid.csv by splitting metadata.csv. Overwrites existing valid.csv.
    Modifies the original metadata.csv to contain only training samples.

    Args:
        metadata_path (str): Path to the main metadata.csv file.
        valid_ratio (float): Fraction of data to use for the validation set. Defaults to 0.1 (10%).

    Returns:
        str or None: Path to the created valid.csv on success, None on failure.
    """
    logging.info(f"Generating validation split from {metadata_path} (ratio: {valid_ratio})")
    valid_path = os.path.join(os.path.dirname(metadata_path), "valid.csv")

    try:
        # Read the full metadata using pandas
        df = pd.read_csv(metadata_path, sep="|", encoding="utf-8", on_bad_lines='error') # Error on bad lines

        if len(df) < 2: # Need at least 2 samples to split reasonably
            logging.warning(f"Not enough data ({len(df)} samples) in {metadata_path} to create a validation split. Skipping.")
            # If valid.csv exists, maybe remove it? Or leave it? For now, just skip generation.
            if os.path.exists(valid_path):
                os.remove(valid_path)
            return None # Indicate that validation file wasn't created/updated

        # Calculate number of validation samples, ensuring at least 1 train and 1 valid sample if possible
        n_samples = len(df)
        n_valid = max(1, int(n_samples * valid_ratio))
        if n_valid >= n_samples:
            n_valid = n_samples - 1 # Ensure at least one training sample remains

        # Shuffle indices and split DataFrame
        indices = np.random.permutation(n_samples)
        valid_df = df.iloc[indices[:n_valid]]
        train_df = df.iloc[indices[n_valid:]]

        # --- Overwrite Files ---
        # Backup original metadata before overwriting
        backup_file(metadata_path, "metadata_presplit_backup")
        # Backup existing validation file before overwriting
        if os.path.exists(valid_path):
             backup_file(valid_path, "valid_presplit_backup")

        # Overwrite train_df back to the original metadata file (now only training data)
        train_df.to_csv(metadata_path, sep="|", index=False, encoding="utf-8", quoting=csv.QUOTE_MINIMAL)
        # Write validation data to valid.csv
        valid_df.to_csv(valid_path, sep="|", index=False, encoding="utf-8", quoting=csv.QUOTE_MINIMAL)

        logging.info(f"Split metadata: {len(train_df)} train samples written to {metadata_path}, "
                     f"{len(valid_df)} validation samples written to {valid_path}")
        return valid_path # Return path to validation file

    except FileNotFoundError:
        logging.error(f"Metadata file not found for splitting: {metadata_path}")
        return None
    except pd.errors.EmptyDataError:
        logging.error(f"Metadata file {metadata_path} is empty, cannot split.")
        return None
    except (TypeError, Exception) as e:
        logging.error(f"Error generating validation split from {metadata_path}: {e}")
        # Attempt to restore backup if split failed? Might be complex.
        return None


def start_tts_training(config_path, resume_from_checkpoint=None, status_queue=None, stop_event=None):
    """
    Starts the TTS model training process using the specified config file.

    Args:
        config_path (str): Path to the training configuration JSON file.
        resume_from_checkpoint (str, optional): Path to a checkpoint to resume training.
        status_queue (queue.Queue, optional): Queue for GUI status updates.
        stop_event (threading.Event, optional): Event to signal cancellation.

    Returns:
        bool: True if training completes successfully, False otherwise.
    """
    if TTS is None:
        logging.error("Coqui TTS library not available for training.")
        if status_queue:
            status_queue.put("TTS Library Error.")
        return False

    if not os.path.exists(config_path):
        logging.error(f"Configuration file not found: {config_path}")
        if status_queue:
            status_queue.put("Config file missing.")
        return False

    try:
        # Load config
        logging.info(f"Loading configuration from: {config_path}")
        config = load_config(config_path)
        output_path = config.output_path

        # Ensure model is available if pre-trained
        if config.use_pre_trained:
            model_id = config.pre_trained_model
            if not ensure_tts_model(model_id, status_queue=status_queue):
                logging.error(f"Cannot proceed without model {model_id}.")
                return False

        # Initialize TTS model
        logging.info("Initializing TTS model and components...")
        if status_queue:
            status_queue.put("Initializing TTS model...")
        tts = TTS(model_path=resume_from_checkpoint, config_path=config_path, progress_bar=True)
        ap = AudioProcessor.init_from_config(config)

        # Initialize Trainer
        logging.info("Initializing Trainer...")
        trainer = Trainer(
            args=None,
            config=config,
            output_path=output_path,
            model=tts.model,
            train_loader=None,
            eval_loader=None,
            audio_processor=ap,
        )

        # Start Training
        logging.info(f"--- Starting TTS Training (Output: {output_path}) ---")
        if resume_from_checkpoint:
            logging.info(f"Resuming from checkpoint: {resume_from_checkpoint}")
        if status_queue:
            status_queue.put(f"Starting TTS training{' (resuming)' if resume_from_checkpoint else ''}...")
        trainer.fit(resume_path=resume_from_checkpoint if resume_from_checkpoint else None)

        if stop_event and stop_event.is_set():
            logging.warning(f"Training cancelled for {config_path}")
            if status_queue:
                status_queue.put("Training cancelled.")
            return False

        logging.info(f"--- Training completed successfully for: {config_path} ---")
        if status_queue:
            status_queue.put("Training completed successfully.")
        return True

    except KeyboardInterrupt:
        logging.info("Training interrupted by user.")
        if status_queue:
            status_queue.put("Training interrupted.")
        return False
    except (TypeError, Exception) as e:
        logging.error(f"Error during TTS training: {e}")
        if status_queue:
            status_queue.put(f"Training failed: {str(e)}")
        return False
    finally:
        # Cleanup: Close any resources if needed (e.g., TensorBoard, GPU sessions)
        if hasattr(tts, "close"):
            tts.close()
        if hasattr(ap, "close"):
            ap.close()


def test_trained_model(config_path, test_text="Hello, this is a test of the trained model!", output_wav="test_output.wav", status_queue=None):
    """
    Tests the latest trained TTS model by synthesizing sample text.

    Args:
        config_path (str): Path to the configuration JSON file used for training.
        test_text (str): The text to synthesize.
        output_wav (str): The filename for the output WAV file (saved in model's output dir).
        status_queue (queue.Queue, optional): Queue for GUI status updates. Defaults to None.

    Returns:
        str or False: The path to the generated WAV file on success, False on failure.
    """
    if TTS is None:
         logging.error("Coqui TTS not available for testing.")
         if status_queue:
            status_queue.put("TTS Library Error.")
         return False

    if not os.path.exists(config_path):
        logging.error(f"Configuration file not found for testing: {config_path}")
        if status_queue:
            status_queue.put("Config file missing for test.")
        return False

    try:
        # Load config to find the output path
        with open(config_path, "r", encoding="utf-8") as f:
            config_dict = json.load(f) # Load as dict to get output_path easily
        output_path = config_dict.get("output_path")
        if not output_path:
             logging.error("Output path not found in config file.")
             if status_queue:
                status_queue.put("Output path missing in config.")
             return False
        if not os.path.isdir(output_path):
             logging.error(f"Model output directory not found: {output_path}")
             if status_queue:
                status_queue.put("Model output directory missing.")
             return False


        # Find the latest checkpoint within the output directory
        latest_checkpoint = find_latest_checkpoint(output_path)

        if not latest_checkpoint:
            logging.error(f"No valid checkpoint (.pth file) found in {output_path}")
            if status_queue:
                status_queue.put("No trained model checkpoint found.")
            return False

        logging.info(f"Loading model for testing from checkpoint: {latest_checkpoint}")
        if status_queue:
            status_queue.put("Loading trained model for test...")

        # Initialize TTS with the specific checkpoint and config
        # Ensure config_path points to the *original* config used for training that checkpoint
        tts_tester = TTS(model_path=latest_checkpoint, config_path=config_path, progress_bar=False)

        # Define output path within the model's output directory
        output_wav_path = os.path.join(output_path, output_wav) # Save test output in the model dir

        logging.info(f"Synthesizing test audio to: {output_wav_path}")
        if status_queue:
            status_queue.put("Synthesizing test audio...")

        # Synthesize audio using the loaded model
        tts_tester.tts_to_file(text=test_text, file_path=output_wav_path)

        logging.info(f"Test audio generated successfully: {output_wav_path}")
        if status_queue:
            status_queue.put(f"Test audio saved: {output_wav}") # Show relative name
        return output_wav_path # Return the full path to the generated audio

    except FileNotFoundError as e:
        logging.error(f"File not found during model test: {e}")
        if status_queue:
            status_queue.put(f"Test failed: File not found {e}")
        return False
    except ImportError as e:
         logging.error(f"Coqui TTS library import or component error during testing: {e}")
         if status_queue:
            status_queue.put("TTS Library Error during test.")
         return False
    except (TypeError, Exception) as e:
        logging.exception(f"Error testing trained model from {config_path}: {e}") # Log traceback
        if status_queue:
            status_queue.put(f"Test failed: {str(e)}")
        return False


def find_latest_checkpoint(output_path):
    """
    Finds the latest Coqui TTS checkpoint file (model.pth or checkpoint_*.pth) in the output directory.

    Args:
        output_path (str): The directory where training outputs (checkpoints) are saved.

    Returns:
        str or None: The path to the latest checkpoint file, or None if none are found.
    """
    checkpoint_dir = output_path # Coqui often saves checkpoints directly in output_path or a subfolder
    if not os.path.exists(checkpoint_dir) or not os.path.isdir(checkpoint_dir):
        logging.warning(f"Checkpoint directory not found or not a directory: {checkpoint_dir}")
        return None

    # Look for standard Coqui TTS checkpoint files
    # Prioritize 'best_model.pth' if it exists, otherwise find latest 'checkpoint_*.pth' or 'model_*.pth'
    checkpoints = []
    best_model_path = os.path.join(checkpoint_dir, "best_model.pth") # Common name for best model

    if os.path.exists(best_model_path):
         logging.info(f"Found best model checkpoint: {best_model_path}")
         return best_model_path # Prefer best model if available

    # If no best_model.pth, look for other checkpoint files
    try:
        all_files = os.listdir(checkpoint_dir)
        # Regex to match common checkpoint naming patterns
        checkpoint_pattern = re.compile(r"^(checkpoint|model)_(\d+)\.pth$")
        for f in all_files:
            if f.endswith(".pth") and checkpoint_pattern.match(f) :
                 checkpoints.append(os.path.join(checkpoint_dir, f))

    except FileNotFoundError:
         logging.error(f"Checkpoint directory {checkpoint_dir} not found when listing files.")
         return None
    except (TypeError, Exception) as e:
         logging.error(f"Error listing files in checkpoint directory {checkpoint_dir}: {e}")
         return None


    if not checkpoints:
        logging.warning(f"No checkpoint files (.pth matching pattern) found in {checkpoint_dir}")
        # Fallback: Check for any .pth file as a last resort
        any_pth = [os.path.join(checkpoint_dir, f) for f in all_files if f.endswith(".pth")]
        if any_pth:
             latest_any_pth = max(any_pth, key=os.path.getmtime)
             logging.warning(f"Found a generic .pth file, using latest: {latest_any_pth}")
             return latest_any_pth
        return None # Truly no checkpoints found

    # Find the most recently modified checkpoint file among numbered ones
    try:
        latest_checkpoint = max(checkpoints, key=os.path.getmtime)
        logging.info(f"Found latest numbered checkpoint: {latest_checkpoint}")
        return latest_checkpoint
    except (TypeError, Exception) as e:
         logging.error(f"Error finding latest checkpoint by modification time: {e}")
         return None


def backup_file(path, suffix="backup"):
    """
    Creates a timestamped backup of a file or directory.

    Args:
        path (str): The path to the file or directory to back up.
        suffix (str): A suffix to add before the timestamp in the backup name. Defaults to "backup".
    """
    if os.path.exists(path):
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        # Insert suffix before the extension for files
        if os.path.isfile(path):
             base, ext = os.path.splitext(path)
             backup_path = f"{base}.{suffix}.{timestamp}{ext}"
        elif os.path.isdir(path):
             backup_path = f"{path}.{suffix}.{timestamp}"
        else:
             logging.warning(f"Path exists but is neither file nor directory: {path}. Skipping backup.")
             return

        try:
            if os.path.isfile(path):
                shutil.copy2(path, backup_path) # copy2 preserves metadata
                logging.info(f"Backup of file created: {backup_path}")
            elif os.path.isdir(path):
                shutil.copytree(path, backup_path, dirs_exist_ok=True) # Copy directory tree
                logging.info(f"Backup of directory created: {backup_path}")
        except (TypeError, Exception) as e:
            logging.error(f"Failed to create backup for {path} -> {backup_path}: {e}")
    else:
         logging.warning(f"Cannot backup non-existent path: {path}")


def has_trained_model(character, base_output_dir):
    """
    Checks if a trained model (config and at least one checkpoint) exists for the character.

    Args:
        character (str): The character name.
        base_output_dir (str): The base directory containing character folders.

    Returns:
        bool: True if a config and a checkpoint are found, False otherwise.
    """
    safe_character_name = re.sub(r'[\\/*?:"<>|]', "_", character)
    character_folder = os.path.join(base_output_dir, safe_character_name)
    config_path = os.path.join(character_folder, f"{safe_character_name}_config.json")

    # 1. Check if config file exists
    if not os.path.exists(config_path):
        # logging.debug(f"Trained model check: Config not found at {config_path}")
        return False

    # 2. Check if output path exists and contains a checkpoint
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
            output_path = config.get("output_path") # Get output path defined in config
            if not output_path:
                 # logging.debug("Trained model check: Output path missing in config.")
                 return False
            if not os.path.isdir(output_path):
                 # logging.debug(f"Trained model check: Output directory not found: {output_path}")
                 return False
            # Use the helper function to find the latest checkpoint
            latest_checkpoint = find_latest_checkpoint(output_path)
            if latest_checkpoint is None:
                 # logging.debug(f"Trained model check: No checkpoint found in {output_path}")
                 return False
            # logging.debug(f"Trained model check: Found config and checkpoint {latest_checkpoint}")
            return True # Config exists and a checkpoint was found
    except FileNotFoundError: # Config file might disappear between os.path.exists and open
         # logging.debug(f"Trained model check: Config file disappeared before reading: {config_path}")
         return False
    except (json.JSONDecodeError, KeyError) as e:
        logging.warning(f"Error reading config file {config_path} during trained model check: {e}")
        return False
    except (TypeError, Exception) as e:
         logging.error(f"Unexpected error during trained model check for {character}: {e}")
         return False


def speak_tts(config_path, text, character, status_queue=None, stop_event=None):
    """
    Synthesizes speech using the latest trained model for the character and plays it.

    Args:
        config_path (str): Path to the character's training configuration JSON file.
        text (str): The text to synthesize.
        character (str): The character name (used for naming output file).
        status_queue (queue.Queue, optional): Queue for GUI status updates. Defaults to None.
        stop_event (threading.Event, optional): Event to signal cancellation. Defaults to None.

    Returns:
        str or False: The path to the generated MP3 file on success, False on failure or cancellation.
    """
    if TTS is None:
         logging.error("Coqui TTS not available for synthesis.")
         if status_queue:
            status_queue.put("TTS Library Error.")
         return False

    if not text.strip():
        msg = "Please enter text to synthesize."
        logging.warning(msg)
        if status_queue:
            status_queue.put(msg)
        return False

    if not os.path.exists(config_path):
        logging.error(f"Configuration file not found for synthesis: {config_path}")
        if status_queue:
            status_queue.put("Configuration file missing.")
        return False

    if stop_event and stop_event.is_set():
        logging.info("TTS synthesis cancelled before starting.")
        if status_queue:
            status_queue.put("TTS synthesis cancelled.")
        return False

    try:
        # Load config to find output path
        with open(config_path, "r", encoding="utf-8") as f:
            config_dict = json.load(f)
        output_path = config_dict.get("output_path")
        if not output_path or not os.path.isdir(output_path):
             logging.error(f"Model output directory not found or invalid based on config: {output_path}")
             if status_queue:
                status_queue.put("Model output directory missing.")
             return False

        # Find the latest checkpoint
        latest_checkpoint = find_latest_checkpoint(output_path)
        if not latest_checkpoint:
            logging.error(f"No valid checkpoint found in {output_path} for synthesis.")
            if status_queue:
                status_queue.put("No trained model checkpoint found.")
            return False

        logging.info(f"Loading model for TTS synthesis: {latest_checkpoint}")
        if status_queue:
            status_queue.put("Loading model for synthesis...")
        tts_speaker = TTS(model_path=latest_checkpoint, config_path=config_path, progress_bar=False)

        # Define output paths for synthesized audio
        tts_output_dir = os.path.join(output_path, "tts_outputs") # Subdir for generated audio
        os.makedirs(tts_output_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_char_name = re.sub(r'[\\/*?:"<>|]', "_", character)
        base_filename = f"{safe_char_name}_tts_{timestamp}"
        # Use temporary WAV file for initial synthesis
        temp_wav = os.path.join(tts_output_dir, f"{base_filename}_temp.wav")
        mp3_path = os.path.join(tts_output_dir, f"{base_filename}.mp3") # Final MP3 path

        logging.info(f"Synthesizing audio to {mp3_path}")
        if status_queue:
            status_queue.put(f"Synthesizing: {base_filename}.mp3")

        # Synthesize to temporary WAV file
        tts_speaker.tts_to_file(text=text, file_path=temp_wav)

        # Check for cancellation after synthesis, before playback
        if stop_event and stop_event.is_set():
            logging.info("TTS synthesis cancelled after generation, before playback/conversion.")
            if status_queue:
                status_queue.put("TTS synthesis cancelled.")
            if os.path.exists(temp_wav):
                os.remove(temp_wav) # Clean up temp file
            return False # Indicate cancellation

        # --- Convert WAV to MP3 ---
        try:
            logging.info(f"Converting {temp_wav} to MP3...")
            audio = AudioSegment.from_wav(temp_wav)
            audio.export(mp3_path, format="mp3") # Export as MP3
            logging.info(f"Converted to MP3: {mp3_path}")
            if status_queue:
                status_queue.put(f"Saved MP3: {base_filename}.mp3")
        except Exception as convert_err:
            logging.error(f"Failed to convert {temp_wav} to MP3: {convert_err}")
            if status_queue:
                status_queue.put("MP3 conversion failed.")
            # Keep the WAV file in case of conversion error? Or return failure?
            # For now, return False as the desired output wasn't fully created.
            if os.path.exists(temp_wav): # Keep temp wav if conversion fails
                 logging.info(f"Temporary WAV file kept at: {temp_wav}")
            return False # Indicate failure

        finally:
            # Clean up temporary WAV file if MP3 conversion was successful
            if os.path.exists(mp3_path) and os.path.exists(temp_wav):
                try:
                    os.remove(temp_wav)
                except OSError as e:
                     logging.warning(f"Could not remove temporary WAV file {temp_wav}: {e}")


        # --- Play the MP3 sound ---
        try:
            logging.info(f"Playing audio: {mp3_path}")
            if status_queue:
                status_queue.put("Playing audio...")
            playsound.playsound(mp3_path) # This blocks until playback finishes
            logging.info(f"Finished playing: {mp3_path}")
            if status_queue:
                status_queue.put(f"Audio played: {base_filename}.mp3")
            return mp3_path # Return path on successful synthesis and playback

        except Exception as play_err:
            # Catch potential playsound errors (e.g., device issues, file access)
            logging.error(f"Error playing sound {mp3_path}: {play_err}")
            if status_queue:
                status_queue.put(f"Playback failed: {play_err}")
            return False # Indicate failure even if MP3 was created

    except FileNotFoundError as e:
        logging.error(f"File not found during TTS synthesis: {e}")
        if status_queue:
            status_queue.put(f"TTS failed: File not found {e}")
        return False
    except ImportError as e:
         logging.error(f"Coqui TTS library import or component error during synthesis: {e}")
         if status_queue:
            status_queue.put("TTS Library Error during synthesis.")
         return False
    except (TypeError, Exception) as e:
        logging.exception(f"An unexpected error occurred in TTS synthesis or playback for {character}: {e}")
        if status_queue:
            status_queue.put(f"TTS failed: {str(e)}")
        return False
    finally:
        # Ensure temporary WAV is cleaned up if it still exists somehow
        # (Should be handled by conversion block, but as a safeguard)
        temp_wav_final_check = os.path.join(tts_output_dir if 'tts_output_dir' in locals() else output_path, f"{base_filename if 'base_filename' in locals() else 'unknown'}_temp.wav")
        if os.path.exists(temp_wav_final_check):
             try:
                 os.remove(temp_wav_final_check)
                 logging.info(f"Cleaned up leftover temporary WAV: {temp_wav_final_check}")
             except OSError as e:
                  logging.warning(f"Could not remove final check temp WAV {temp_wav_final_check}: {e}")


# --- GUI ---

# Global variables for GUI elements (consider encapsulating in a class later)
window = None
tts_frame = None
status_queue = queue.Queue() # Queue for thread communication
stop_event = threading.Event() # Event to signal cancellation
current_thread = [None] # List to hold the currently running background thread

# Helper function for GUI parameter validation
def validate_gui_parameters(batch_size_var, num_epochs_var, learning_rate_var):
    """Validates numeric parameters from GUI StringVars. Returns dict or None."""
    params = {}
    try:
        params["batch_size"] = int(batch_size_var.get())
        if params["batch_size"] <= 0:
            raise ValueError("Batch size must be positive.")
    except ValueError:
        messagebox.showerror("Validation Error", "Invalid Batch Size. Must be a positive integer.", parent=window)
        return None
    try:
        params["num_epochs"] = int(num_epochs_var.get())
        if params["num_epochs"] <= 0:
            raise ValueError("Number of epochs must be positive.")
    except ValueError:
        messagebox.showerror("Validation Error", "Invalid Number of Epochs. Must be a positive integer.", parent=window)
        return None
    try:
        params["learning_rate"] = float(learning_rate_var.get())
        if params["learning_rate"] <= 0:
            raise ValueError("Learning rate must be positive.")
    except ValueError:
        messagebox.showerror("Validation Error", "Invalid Learning Rate. Must be a positive number.", parent=window)
        return None
    logging.info(f"GUI parameters validated: {params}")
    return params

# --- GUI Main Function ---
def main_gui():
    """Sets up and runs the main Tkinter GUI."""
    global window, tts_frame, status_queue, stop_event, current_thread

    window = tk.Tk()
    window.title("Genshin Impact Voice Toolkit")
    window.geometry("650x650") # Set a default size

    # --- Main Frame ---
    main_frame = ttk.Frame(window, padding="10")
    main_frame.pack(fill=tk.BOTH, expand=True)

    # Make the main column expandable
    main_frame.columnconfigure(0, weight=1)

    # --- Configuration Frame ---
    config_frame = ttk.LabelFrame(main_frame, text="Configuration", padding="10")
    config_frame.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
    config_frame.columnconfigure(1, weight=1) # Allow entry fields to expand

    # Language Selection
    ttk.Label(config_frame, text="Language:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
    languages = ["English", "Japanese", "Chinese", "Korean"] # Add more if supported by Wiki categories/TTS
    language_var = tk.StringVar(value="English")
    ttk.Combobox(config_frame, textvariable=language_var, values=languages, state="readonly", width=15).grid(
        row=0, column=1, padx=5, pady=5, sticky="w" # Align left
    )

    # Character Selection
    ttk.Label(config_frame, text="Character:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
    character_names = fetch_character_list_from_api() # Fetch names on startup
    character_var = tk.StringVar(value=character_names[0] if character_names else "No characters found")
    character_dropdown = ttk.Combobox(
        config_frame, textvariable=character_var, values=character_names,
        state="readonly" if character_names else "disabled", width=35 # Wider dropdown
    )
    character_dropdown.grid(row=1, column=1, columnspan=2, padx=5, pady=5, sticky="ew") # Span columns and expand

    # Output Directory
    ttk.Label(config_frame, text="Output Dir:").grid(row=2, column=0, padx=5, pady=5, sticky="w")
    output_dir_var = tk.StringVar(value=os.path.abspath(BASE_DATA_DIR)) # Default to absolute path
    output_dir_entry = ttk.Entry(config_frame, textvariable=output_dir_var, width=40)
    output_dir_entry.grid(row=2, column=1, padx=5, pady=5, sticky="ew")
    # Add Browse button (optional)
    # browse_button = ttk.Button(config_frame, text="Browse...", command=lambda: ...)
    # browse_button.grid(row=2, column=2, padx=5, pady=5)

    # --- Processing Options Frame ---
    processing_frame = ttk.LabelFrame(main_frame, text="Processing Options", padding="10")
    processing_frame.grid(row=1, column=0, padx=5, pady=5, sticky="nsew")
    processing_frame.columnconfigure(1, weight=1)

    # Whisper Model Selection
    ttk.Label(processing_frame, text="Whisper Model:").grid(row=0, column=0, padx=5, pady=2, sticky="w")
    whisper_models = ["base", "small", "medium", "large-v2"] # Available Whisper model sizes
    whisper_model_var = tk.StringVar(value="base")
    whisper_model_combo = ttk.Combobox(
        processing_frame, textvariable=whisper_model_var, values=whisper_models, state="readonly", width=15
    )
    whisper_model_combo.grid(row=0, column=1, padx=5, pady=2, sticky="w") # Align left

    # Segmentation Checkbox and HF Token Entry
    use_segmentation_var = tk.BooleanVar(value=False)
    seg_check = ttk.Checkbutton(
        processing_frame, text="Use Audio Segmentation (Requires PyAnnote & HF Token)", variable=use_segmentation_var
    )
    seg_check.grid(row=1, column=0, columnspan=3, padx=5, pady=2, sticky="w") # Span columns

    ttk.Label(processing_frame, text="HF Token:").grid(row=2, column=0, padx=5, pady=2, sticky="w")
    hf_token_var = tk.StringVar()
    hf_token_entry = ttk.Entry(processing_frame, textvariable=hf_token_var, width=40, show="*", state="disabled") # Start disabled
    hf_token_entry.grid(row=2, column=1, columnspan=2, padx=5, pady=2, sticky="ew") # Expand entry

    # Strict ASCII Checkbox
    strict_ascii_var = tk.BooleanVar(value=False)
    ttk.Checkbutton(
        processing_frame, text="Strict ASCII Transcriptions (Removes non-ASCII chars)", variable=strict_ascii_var
    ).grid(row=3, column=0, columnspan=3, padx=5, pady=2, sticky="w")

    # Download Wiki Audio Checkbox
    download_wiki_audio_var = tk.BooleanVar(value=True)
    ttk.Checkbutton(
        processing_frame, text="Download/Update Audio from Wiki", variable=download_wiki_audio_var
    ).grid(row=4, column=0, columnspan=3, padx=5, pady=2, sticky="w")


    # --- Training Options Frame ---
    training_frame = ttk.LabelFrame(main_frame, text="Training Options", padding="10")
    training_frame.grid(row=2, column=0, padx=5, pady=5, sticky="nsew")
    training_frame.columnconfigure(1, weight=1) # Allow combobox to expand slightly if needed

    # TTS Model Selection
    ttk.Label(training_frame, text="TTS Model:").grid(row=0, column=0, padx=5, pady=2, sticky="w")
    tts_model_var = tk.StringVar(value=list(AVAILABLE_MODELS.keys())[0]) # Default to first available model
    tts_model_combo = ttk.Combobox(
        training_frame, textvariable=tts_model_var, values=list(AVAILABLE_MODELS.keys()), state="readonly", width=25
    )
    tts_model_combo.grid(row=0, column=1, padx=5, pady=2, sticky="w") # Align left

    # Training Hyperparameters (Batch Size, Epochs, Learning Rate)
    param_frame = ttk.Frame(training_frame) # Sub-frame for parameters
    param_frame.grid(row=1, column=0, columnspan=3, pady=5, sticky="w")

    ttk.Label(param_frame, text="Batch Size:").pack(side=tk.LEFT, padx=(5,2))
    batch_size_var = tk.StringVar(value="16")
    ttk.Entry(param_frame, textvariable=batch_size_var, width=6).pack(side=tk.LEFT, padx=(0,10))

    ttk.Label(param_frame, text="Epochs:").pack(side=tk.LEFT, padx=(5,2))
    num_epochs_var = tk.StringVar(value="100") # Adjust default based on typical needs
    ttk.Entry(param_frame, textvariable=num_epochs_var, width=6).pack(side=tk.LEFT, padx=(0,10))

    ttk.Label(param_frame, text="Learning Rate:").pack(side=tk.LEFT, padx=(5,2))
    learning_rate_var = tk.StringVar(value="0.0001")
    ttk.Entry(param_frame, textvariable=learning_rate_var, width=8).pack(side=tk.LEFT, padx=(0,5))


    # --- TTS Playback Frame (Initially Hidden) ---
    tts_frame = ttk.LabelFrame(main_frame, text="TTS Playback", padding="10")
    # Grid config deferred until update_tts_frame_visibility

    ttk.Label(tts_frame, text="Text to Speak:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
    tts_text_var = tk.StringVar()
    tts_text_entry = ttk.Entry(tts_frame, textvariable=tts_text_var, width=45)
    tts_text_entry.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

    speak_button = ttk.Button(tts_frame, text="Speak", state="disabled") # Command assigned later
    speak_button.grid(row=0, column=2, padx=5, pady=5)
    tts_frame.columnconfigure(1, weight=1) # Allow text entry to expand


    # --- Control Frame (Status and Buttons) ---
    control_frame = ttk.Frame(main_frame, padding="5")
    control_frame.grid(row=4, column=0, padx=5, pady=5, sticky="nsew")
    control_frame.columnconfigure(0, weight=1) # Allow status label to expand

    status_label = ttk.Label(control_frame, text="Ready.", relief=tk.SUNKEN, anchor="w", padding=(5, 2))
    status_label.grid(row=0, column=0, padx=(0, 5), pady=5, sticky="ew") # Status label on the left

    button_frame = ttk.Frame(control_frame) # Frame for buttons on the right
    button_frame.grid(row=0, column=1, sticky="e")

    # --- Buttons ---
    process_button = ttk.Button(button_frame, text="Process Voices") # Command assigned later
    process_button.pack(side=tk.LEFT, padx=2)

    train_button = ttk.Button(button_frame, text="Start Training") # Command assigned later
    train_button.pack(side=tk.LEFT, padx=2)

    test_button = ttk.Button(button_frame, text="Test Model") # Command assigned later
    test_button.pack(side=tk.LEFT, padx=2)

    cancel_button = ttk.Button(button_frame, text="Cancel", state="disabled") # Command assigned later
    cancel_button.pack(side=tk.LEFT, padx=2)


    # --- GUI Logic Functions ---

    def update_hf_token_state(*args):
        """Enable/disable HF Token entry based on segmentation checkbox."""
        if use_segmentation_var.get():
            hf_token_entry.config(state="normal")
        else:
            hf_token_entry.config(state="disabled")
            # Optionally clear the token when disabling
            # hf_token_var.set("")
    # Trigger update when checkbox changes
    use_segmentation_var.trace_add("write", update_hf_token_state)
    update_hf_token_state() # Initial state check


    def update_tts_frame_visibility(*args):
        """Show/hide the TTS playback frame based on whether a trained model exists."""
        character = character_var.get()
        base_output_dir = output_dir_var.get()
        show_tts = False
        # Basic check for valid inputs before checking for model
        if character and character != "No characters found" and base_output_dir and os.path.isdir(base_output_dir):
            try:
                # Check if config and checkpoint exist
                if has_trained_model(character, base_output_dir):
                    show_tts = True
            except (TypeError, Exception) as e: # Catch potential errors during check
                logging.warning(f"Error checking for trained model: {e}")

        # Update GUI elements
        if show_tts:
            tts_frame.grid(row=3, column=0, padx=5, pady=5, sticky="nsew") # Show frame
            speak_button.config(state="normal")
        else:
            tts_frame.grid_remove() # Hide frame
            speak_button.config(state="disabled")

    # Trigger update when character or output directory changes
    character_var.trace_add("write", update_tts_frame_visibility)
    output_dir_var.trace_add("write", update_tts_frame_visibility)
    # Call initially after GUI setup
    window.after(100, update_tts_frame_visibility)


    def set_buttons_state(new_state):
        """Enable/disable control buttons based on state ('normal' or 'disabled')."""
        process_button.config(state=new_state)
        train_button.config(state=new_state)
        test_button.config(state=new_state)
        # Cancel button is enabled only when an operation is running (buttons are disabled)
        cancel_button.config(state="normal" if new_state == "disabled" else "disabled")
        # Update speak button based on model existence only when enabling main buttons
        if new_state == "normal":
            update_tts_frame_visibility() # This handles enabling/disabling speak button
        else:
            speak_button.config(state="disabled") # Always disable speak button during operations


    def update_status_display():
        """Periodically checks the status queue and updates the status label."""
        try:
            while True: # Process all messages currently in queue
                message = status_queue.get_nowait()
                if message == "TASK_COMPLETE":
                    set_buttons_state("normal")
                    status_label.config(text="Ready.") # Reset status
                    update_tts_frame_visibility() # Re-check if model exists now
                elif message == "TASK_CANCELLED":
                     set_buttons_state("normal")
                     status_label.config(text="Operation Cancelled.")
                     update_tts_frame_visibility()
                elif message == "TASK_FAILED":
                     set_buttons_state("normal")
                     # Status label should show the error message put before TASK_FAILED
                     update_tts_frame_visibility()
                else:
                    # Update status label with the message from the queue
                    status_label.config(text=message)
                window.update_idletasks() # Update GUI immediately
        except queue.Empty:
            pass # No messages left in the queue
        finally:
            # Schedule the next check
            window.after(150, update_status_display) # Check again shortly


    def run_task_in_thread(task_func, *args):
        """
        Runs a given function in a separate thread, handling button states and errors.

        Args:
            task_func: The function to run in the background thread.
            *args: Arguments to pass to the task_func.

        Returns:
            bool: True if the task was started, False if another task is already running.
        """
        global current_thread
        if current_thread[0] and current_thread[0].is_alive():
            messagebox.showwarning("Busy", "Another operation is currently running. Please wait or cancel.", parent=window)
            return False

        stop_event.clear() # Clear stop flag before starting new task
        set_buttons_state("disabled") # Disable buttons while task runs

        # Wrapper function to run in the thread
        def thread_wrapper():
            task_failed = False
            try:
                # Execute the provided task function
                task_func(*args)
                # Check if task was cancelled during execution
                if stop_event.is_set():
                     status_queue.put("TASK_CANCELLED")
                else:
                     # If task function completes without error and wasn't cancelled
                     # status_queue.put("TASK_COMPLETE") # Task func should put final status msg
                     pass # Task func should put its final status before TASK_COMPLETE/FAILED
            except (TypeError, Exception) as e:
                task_failed = True
                error_msg = f"Thread Error: {str(e)}"
                logging.exception(f"Error in background thread for {task_func.__name__}") # Log full traceback
                status_queue.put(error_msg) # Put error message for display
            finally:
                # Signal GUI thread to re-enable buttons, unless cancelled
                if not stop_event.is_set():
                    status_queue.put("TASK_FAILED" if task_failed else "TASK_COMPLETE")

                current_thread[0] = None # Clear the current thread reference

        # Create and start the daemon thread
        current_thread[0] = threading.Thread(target=thread_wrapper, daemon=True)
        current_thread[0].start()
        return True

    # --- Task Functions (to be run in threads) ---

    def task_process_voices():
        """Background task for downloading and transcribing voices."""
        character = character_var.get()
        language = language_var.get()
        base_output_dir = output_dir_var.get()
        whisper_model = whisper_model_var.get()
        use_segmentation = use_segmentation_var.get()
        strict_ascii = strict_ascii_var.get()
        hf_token = hf_token_var.get()
        download_wiki_audio = download_wiki_audio_var.get()

        status_queue.put(f"Starting voice processing for {character}...")
        # Call the main processing function
        result_folder = process_character_voices(
            character, language, base_output_dir, download_wiki_audio,
            whisper_model, use_segmentation, hf_token, strict_ascii,
            status_queue=status_queue, stop_event=stop_event
        )
        # Final status message is handled by the function itself or the wrapper
        if result_folder and not stop_event.is_set():
             status_queue.put(f"Processing finished for {character}.") # Final success msg
        elif not stop_event.is_set():
             status_queue.put(f"Processing failed for {character}.") # Final fail msg (error logged elsewhere)


    def task_start_training():
        """Background task for training the TTS model."""
        character = character_var.get()
        base_output_dir = output_dir_var.get()
        selected_tts_model = tts_model_var.get()
        # Parameters validated before calling run_task_in_thread
        params = validate_gui_parameters(batch_size_var, num_epochs_var, learning_rate_var)
        # Wrapper already handles button state, just run the logic

        status_queue.put(f"Preparing training for {character}...")
        # Update config file with current GUI settings
        config_path = update_character_config(
            character, base_output_dir, selected_model=selected_tts_model,
            batch_size=params["batch_size"], num_epochs=params["num_epochs"], learning_rate=params["learning_rate"]
        )
        if not config_path:
            status_queue.put("Error: Failed to create/update config file.")
            raise RuntimeError("Config generation failed.") # Raise error to signal failure in wrapper

        # Check prerequisites again right before training (config path is now validated)
        character_folder = os.path.dirname(config_path)
        if not validate_training_prerequisites(character_folder, config_path):
            # Error message logged by validation function
            status_queue.put("Error: Training prerequisites failed.")
            raise RuntimeError("Prerequisites check failed.") # Raise error

        # Find latest checkpoint to potentially resume from
        resume_checkpoint = find_latest_checkpoint(os.path.join(base_output_dir, "tts_train_output", re.sub(r'[\\/*?:"<>|]', "_", character)))
        if resume_checkpoint:
             status_queue.put(f"Found checkpoint, will attempt to resume: {os.path.basename(resume_checkpoint)}")

        # Start the actual training process
        success = start_tts_training(
            config_path,
            resume_from_checkpoint=resume_checkpoint,
            status_queue=status_queue,
            stop_event=stop_event
        )

        if success and not stop_event.is_set():
            status_queue.put("Training finished. Testing model...")
            # Automatically test the model after successful training
            test_trained_model(config_path, status_queue=status_queue)
            status_queue.put(f"Training & Testing complete for {character}.") # Final success
        elif not stop_event.is_set():
             status_queue.put(f"Training failed for {character}.") # Final fail
             raise RuntimeError("TTS Training function returned failure.") # Raise error


    def task_test_model():
        """Background task for testing the latest trained model."""
        character = character_var.get()
        base_output_dir = output_dir_var.get()
        safe_character_name = re.sub(r'[\\/*?:"<>|]', "_", character)
        character_folder = os.path.join(base_output_dir, safe_character_name)
        config_path = os.path.join(character_folder, f"{safe_character_name}_config.json")

        if not os.path.exists(config_path):
            status_queue.put(f"Error: Config file missing for {character}.")
            raise FileNotFoundError(f"Config file not found: {config_path}")

        status_queue.put(f"Testing latest model for {character}...")
        result_path = test_trained_model(config_path, status_queue=status_queue)
        if result_path and not stop_event.is_set():
             status_queue.put(f"Model test complete. Output: {os.path.basename(result_path)}") # Final success
        elif not stop_event.is_set():
             status_queue.put(f"Model test failed for {character}.") # Final fail
             raise RuntimeError("Model testing function returned failure.")


    def task_speak_model():
        """Background task for synthesizing speech using the trained model."""
        character = character_var.get()
        base_output_dir = output_dir_var.get()
        text_to_speak = tts_text_var.get()
        safe_character_name = re.sub(r'[\\/*?:"<>|]', "_", character)
        character_folder = os.path.join(base_output_dir, safe_character_name)
        config_path = os.path.join(character_folder, f"{safe_character_name}_config.json")

        if not text_to_speak.strip(): # Handled by button click validation, but double-check
            status_queue.put("Error: No text entered to speak.")
            raise ValueError("Empty text provided for synthesis.")

        if not os.path.exists(config_path):
            status_queue.put(f"Error: Config file missing for {character}.")
            raise FileNotFoundError(f"Config file not found: {config_path}")

        status_queue.put(f"Synthesizing speech for {character}...")
        result_path = speak_tts(config_path, text_to_speak, character, status_queue=status_queue, stop_event=stop_event)

        if result_path and not stop_event.is_set():
             status_queue.put(f"Speech synthesis complete. Played: {os.path.basename(result_path)}") # Final success
        elif not stop_event.is_set():
             status_queue.put(f"Speech synthesis failed for {character}.") # Final fail
             raise RuntimeError("Speech synthesis function returned failure.")


    # --- Button Command Lambdas/Functions ---

    def on_process_voices_click():
        character = character_var.get()
        base_output_dir = output_dir_var.get()
        use_segmentation = use_segmentation_var.get()
        hf_token = hf_token_var.get()

        # --- Input Validation ---
        if not character or character == "No characters found":
            messagebox.showerror("Input Error", "Please select a valid character.", parent=window)
            return
        if not base_output_dir or not os.path.isdir(base_output_dir):
            # Attempt to create directory? Or just error? For now, error.
            messagebox.showerror("Input Error", f"Output directory '{base_output_dir}' does not exist or is not a valid directory.", parent=window)
            return
        if use_segmentation and not hf_token:
            messagebox.showerror("Input Error", "Hugging Face token is required when 'Use Audio Segmentation' is checked.", parent=window)
            return

        # Start the task in a background thread
        run_task_in_thread(task_process_voices)

    def on_start_training_click():
        character = character_var.get()
        base_output_dir = output_dir_var.get()

        # --- Input Validation ---
        params = validate_gui_parameters(batch_size_var, num_epochs_var, learning_rate_var)
        if not params:
            return # Validation failed, message shown by validator

        if not character or character == "No characters found":
            messagebox.showerror("Input Error", "Please select a valid character.", parent=window)
            return
        if not base_output_dir or not os.path.isdir(base_output_dir):
            messagebox.showerror("Input Error", f"Output directory '{base_output_dir}' does not exist or is not valid.", parent=window)
            return

        # Check if processing seems to have been done (metadata exists)
        safe_character_name = re.sub(r'[\\/*?:"<>|]', "_", character)
        metadata_path = os.path.join(base_output_dir, safe_character_name, "metadata.csv")
        if not os.path.exists(metadata_path):
            messagebox.showerror("Prerequisite Error", f"Metadata file not found for {character}.\nPlease run 'Process Voices' first.", parent=window)
            return

        # Start the task in a background thread
        run_task_in_thread(task_start_training)

    def on_test_model_click():
        character = character_var.get()
        base_output_dir = output_dir_var.get()

        # --- Input Validation ---
        if not character or character == "No characters found":
            messagebox.showerror("Input Error", "Please select a valid character.", parent=window)
            return
        if not base_output_dir or not os.path.isdir(base_output_dir):
            messagebox.showerror("Input Error", f"Output directory '{base_output_dir}' does not exist or is not valid.", parent=window)
            return
        if not has_trained_model(character, base_output_dir):
            messagebox.showerror("Model Error", f"No trained model (config + checkpoint) found for {character}.\nPlease run training first.", parent=window)
            return

        # Start the task in a background thread
        run_task_in_thread(task_test_model)

    def on_cancel_click():
        """Signals the running background thread to stop."""
        if current_thread[0] and current_thread[0].is_alive():
            logging.info("Cancel button pressed. Setting stop event.")
            stop_event.set()
            cancel_button.config(state="disabled") # Prevent multiple clicks
            status_label.config(text="Cancelling... Please wait for current step to finish.")
            window.update_idletasks()
        else:
            logging.warning("Cancel pressed but no operation appears to be running.")

    def on_speak_model_click():
        character = character_var.get()
        base_output_dir = output_dir_var.get()
        text_to_speak = tts_text_var.get()

         # --- Input Validation ---
        if not character or character == "No characters found":
            messagebox.showerror("Input Error", "Please select a valid character.", parent=window)
            return
        if not base_output_dir or not os.path.isdir(base_output_dir):
            messagebox.showerror("Input Error", f"Output directory '{base_output_dir}' does not exist or is not valid.", parent=window)
            return
        if not text_to_speak.strip():
            messagebox.showerror("Input Error", "Please enter text in the 'Text to Speak' box.", parent=window)
            return
        if not has_trained_model(character, base_output_dir):
             messagebox.showerror("Model Error", f"No trained model found for {character}.", parent=window)
             return

         # Start the task in a background thread
        run_task_in_thread(task_speak_model)


    # --- Assign Commands to Buttons ---
    process_button.configure(command=on_process_voices_click)
    train_button.configure(command=on_start_training_click)
    test_button.configure(command=on_test_model_click)
    cancel_button.configure(command=on_cancel_click)
    speak_button.configure(command=on_speak_model_click)


    # --- Final Setup ---
    update_status_display() # Start the status queue checker loop
    window.mainloop() # Start the Tkinter event loop


# --- Main Execution ---
if __name__ == "__main__":
    # Argument Parser Setup (for CLI execution)
    parser = argparse.ArgumentParser(
        description="Genshin Impact Voice Toolkit: Download, Transcribe, Train TTS.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter # Show defaults in help
    )

    # Subparsers for different actions (process, train, test, speak)
    subparsers = parser.add_subparsers(dest='command', help='Action to perform (leave blank for GUI)')

    # --- Common Arguments for all commands ---
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument(
        "--character", type=str, required=True, help="Character name (e.g., 'Arlecchino')."
    )
    parent_parser.add_argument(
        "--output_dir", type=str, default=BASE_DATA_DIR, help="Base output directory for character data."
    )

    # --- Process Command Arguments ---
    parser_process = subparsers.add_parser('process', help='Download and transcribe voice data.', parents=[parent_parser], formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser_process.add_argument(
        "--language", type=str, default="English", choices=["English", "Japanese", "Chinese", "Korean"], help="Voice language for Wiki category selection."
    )
    parser_process.add_argument(
        "--whisper_model", type=str, default="base", choices=["base", "small", "medium", "large-v2"], help="Whisper model size for transcription."
    )
    parser_process.add_argument(
        "--use_segmentation", action="store_true", help="Use PyAnnote segmentation (requires --hf_token and library install)."
    )
    parser_process.add_argument(
        "--strict_ascii", action="store_true", help="Force ASCII-only transcriptions (may lose data)."
    )
    parser_process.add_argument(
        "--hf_token", type=str, default="", help="Hugging Face token (required if --use_segmentation)."
    )
    parser_process.add_argument(
        "--skip_wiki_download", action="store_true", help="Skip downloading audio from the Wiki (only transcribe existing files)."
    )

    # --- Train Command Arguments ---
    parser_train = subparsers.add_parser('train', help='Train a TTS model for the character.', parents=[parent_parser], formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser_train.add_argument(
        "--tts_model", type=str, default="Fast Tacotron2", choices=list(AVAILABLE_MODELS.keys()), help="TTS base model architecture name."
    )
    parser_train.add_argument(
        "--batch_size", type=int, default=16, help="Batch size for training."
    )
    parser_train.add_argument(
        "--num_epochs", type=int, default=100, help="Number of epochs for training."
    )
    parser_train.add_argument(
        "--learning_rate", type=float, default=0.0001, help="Learning rate for training."
    )
    parser_train.add_argument(
        "--resume", action="store_true", help="Resume training from the latest checkpoint if available."
    )

    # --- Test Command Arguments ---
    parser_test = subparsers.add_parser('test', help='Test the trained TTS model with sample text.', parents=[parent_parser], formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser_test.add_argument(
        "--text", type=str, default="Hello, this is a test.", help="Text to synthesize."
    )
    parser_test.add_argument(
        "--output_wav", type=str, default="cli_test_output.wav", help="Output filename for the test audio (saved in model output dir)."
    )

    # --- Speak Command Arguments ---
    parser_speak = subparsers.add_parser('speak', help='Synthesize and play speech using the trained model.', parents=[parent_parser], formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser_speak.add_argument(
        "--text", type=str, required=True, help="Text to synthesize and speak."
    )

    # Parse arguments
    args = parser.parse_args()

    # --- Execute based on arguments ---
    if args.command is None:
        # No command provided, launch GUI
        logging.info("No command specified, launching GUI...")
        if TTS is None:
             print("\nERROR: Coqui TTS components failed to import. GUI cannot run.")
             print("Please ensure Coqui TTS is installed ('pip install TTS') and dependencies are met.")
             exit(1)
        main_gui()
    else:
        # --- CLI Execution ---
        logging.info(f"Executing command '{args.command}' for character '{args.character}' via CLI...")
        # Use a dummy status queue for CLI or implement simple print handler
        cli_status_queue = queue.Queue()

        # Ensure output directory exists for CLI operations
        if not os.path.isdir(args.output_dir):
             try:
                 os.makedirs(args.output_dir, exist_ok=True)
                 logging.info(f"Created base output directory: {args.output_dir}")
             except OSError as e:
                 logging.error
