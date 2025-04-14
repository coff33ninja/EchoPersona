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
from voice_tools import SpeechToText

# Ensure voice_tools.py is in the same directory or Python path
try:
    from voice_tools import SpeechToText
except ImportError:
    print("Error: voice_tools.py not found. Make sure it's in the same directory.")
    logging.error("Failed to import SpeechToText from voice_tools.py")
    # Optionally exit or disable features that depend on it
    SpeechToText = None  # Set to None to indicate it's unavailable

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# --- Constants ---
BASE_DATA_DIR = "voice_datasets"  # Base directory to store all character datasets
WIKI_API_URL = "https://genshin-impact.fandom.com/api.php"
JMP_API_URL_BASE = "https://genshin.jmp.blue"

# --- Helper Functions ---


def get_category_files(category):
    """Fetches file titles from a specific category on the wiki."""
    params = {
        "action": "query",
        "list": "categorymembers",
        "cmtype": "file",
        "cmtitle": category,
        "cmlimit": 500,  # Fetch up to 500 files per request
        "format": "json",
    }
    files = []
    cmcontinue = None
    while True:
        if cmcontinue:
            params["cmcontinue"] = cmcontinue
        try:
            logging.debug(
                f"Fetching category members for: {category} with params: {params}"
            )
            response = requests.get(WIKI_API_URL, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()
            logging.debug(
                f"Wiki API Response (Category Files): {json.dumps(data, indent=2)}"
            )  # Log response for debugging
        except requests.exceptions.Timeout:
            print(f"Timeout fetching category files from wiki for: {category}")
            logging.error(f"Timeout fetching category files from wiki for: {category}")
            break
        except requests.exceptions.RequestException as e:
            print(f"Error fetching category files from wiki: {e}")
            logging.error(f"Error fetching category files from wiki: {e}")
            break
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON response from wiki: {e}")
            logging.error(f"Error decoding JSON response from wiki: {e}")
            # Log the problematic response text if possible
            logging.error(
                f"Response text: {response.text[:500]}..."
            )  # Log first 500 chars
            break

        if "error" in data:
            print(f"Wiki API Error: {data['error'].get('info', 'Unknown error')}")
            logging.error(f"Wiki API Error: {data['error']}")
            break

        if "query" not in data or "categorymembers" not in data["query"]:
            logging.warning(
                f"Unexpected response structure or no members found for {category}. Response: {data}"
            )
            break  # Exit loop if structure is wrong or no members

        members = data["query"]["categorymembers"]
        # Filter out non-English voice lines based on common patterns
        files.extend(
            [
                member["title"]
                for member in members
                if not re.search(r"Vo (JA|KO|ZH)", member["title"], re.IGNORECASE)
            ]
        )

        # Handle pagination
        if "continue" in data and "cmcontinue" in data["continue"]:
            cmcontinue = data["continue"]["cmcontinue"]
            logging.debug(f"Continuing fetch with cmcontinue: {cmcontinue}")
        else:
            break  # No more pages
    logging.info(f"Found {len(files)} potential files in category '{category}'.")
    return files


def get_file_url(file_title):
    """Fetches the direct download URL for a given file title from the wiki."""
    params = {
        "action": "query",
        "titles": file_title,
        "prop": "imageinfo",
        "iiprop": "url",
        "format": "json",
    }
    try:
        logging.debug(f"Fetching file URL for: {file_title}")
        response = requests.get(WIKI_API_URL, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        logging.debug(
            f"Wiki API Response (File URL): {json.dumps(data, indent=2)}"
        )  # Log response
    except requests.exceptions.Timeout:
        print(f"Timeout fetching file URL from wiki for: {file_title}")
        logging.error(f"Timeout fetching file URL from wiki for: {file_title}")
        return None
    except requests.exceptions.RequestException as e:
        print(f"Error fetching file URL from wiki: {e}")
        logging.error(f"Error fetching file URL from wiki: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON response from wiki: {e}")
        logging.error(f"Error decoding JSON response from wiki: {e}")
        logging.error(f"Response text: {response.text[:500]}...")
        return None

    if "error" in data:
        print(
            f"Wiki API Error getting URL for {file_title}: {data['error'].get('info', 'Unknown error')}"
        )
        logging.error(f"Wiki API Error getting URL for {file_title}: {data['error']}")
        return None

    if "query" in data and "pages" in data["query"]:
        pages = data["query"]["pages"]
        # The page ID can be negative for missing pages or vary otherwise
        page_id = list(pages.keys())[0]  # Get the first (and usually only) page ID
        page_info = pages[page_id]

        if (
            "imageinfo" in page_info
            and isinstance(page_info["imageinfo"], list)
            and page_info["imageinfo"]
        ):
            url = page_info["imageinfo"][0].get("url")
            if url:
                logging.info(f"Found URL for '{file_title}': {url}")
                return url
            else:
                logging.warning(f"No URL found in imageinfo for '{file_title}'.")
                return None
        elif "missing" in page_info:
            logging.warning(f"File '{file_title}' marked as missing on Wiki.")
            return None
        else:
            logging.warning(
                f"Unexpected response structure or missing imageinfo for '{file_title}'. Response: {data}"
            )
            return None
    else:
        logging.warning(
            f"Unexpected response structure for '{file_title}'. Response: {data}"
        )
        return None


def download_and_convert(file_url, output_dir, file_name, status_label=None):
    """Downloads an OGG file, converts it to WAV, and cleans up."""
    # Ensure filename is safe for filesystem
    safe_file_name = re.sub(r'[\\/*?:"<>|]', "_", file_name)  # Replace invalid chars
    ogg_file_name = (
        safe_file_name
        if safe_file_name.lower().endswith(".ogg")
        else f"{safe_file_name}.ogg"
    )
    wav_file_name = ogg_file_name.replace(".ogg", ".wav").replace(
        ".OGG", ".wav"
    )  # Ensure lowercase extension
    ogg_path = os.path.join(output_dir, ogg_file_name)
    wav_path = os.path.join(output_dir, wav_file_name)

    # Skip if WAV already exists
    if os.path.exists(wav_path):
        logging.info(
            f"WAV file already exists, skipping download/conversion: {wav_path}"
        )
        if status_label:
            status_label.config(text=f"Skipped (exists): {wav_file_name}")
        return wav_path  # Return existing path

    try:
        if status_label:
            status_label.config(text=f"Downloading {ogg_file_name}...")
        logging.info(f"Downloading {file_url} to {ogg_path}")
        response = requests.get(file_url, timeout=30)  # Increased timeout for download
        response.raise_for_status()
        with open(ogg_path, "wb") as f:
            f.write(response.content)
        logging.info(f"Downloaded successfully: {ogg_path}")

        if status_label:
            status_label.config(text=f"Converting {ogg_file_name} to WAV...")
        logging.info(f"Converting {ogg_path} to {wav_path} (SR: 22050 Hz)")
        # Use -y to overwrite existing intermediate files if ffmpeg asks
        # Use -loglevel error to suppress verbose ffmpeg output unless error occurs
        # Ensure ffmpeg is in the system PATH or provide full path
        subprocess.run(
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
            capture_output=True,
            check=True,  # Raise exception on non-zero exit code
            text=True,  # Capture output as text
        )
        logging.info(f"Conversion successful: {wav_path}")
        if status_label:
            status_label.config(text=f"Converted: {wav_file_name}")
        return wav_path

    except FileNotFoundError:
        # Specific error if ffmpeg is not found
        err_msg = "Error: 'ffmpeg' command not found. Please ensure ffmpeg is installed and in your system's PATH."
        if status_label:
            status_label.config(text=err_msg)
        print(err_msg)
        logging.error(err_msg)
    except requests.exceptions.Timeout:
        if status_label:
            status_label.config(text=f"Timeout downloading {ogg_file_name}")
        print(f"Timeout downloading {ogg_file_name}")
        logging.error(f"Timeout downloading {ogg_file_name}")
    except requests.exceptions.RequestException as e:
        if status_label:
            status_label.config(text=f"Error downloading {ogg_file_name}: {e}")
        print(f"Error downloading {ogg_file_name}: {e}")
        logging.error(f"Error downloading {ogg_file_name}: {e}")
    except subprocess.CalledProcessError as e:
        if status_label:
            status_label.config(text=f"Error converting {ogg_file_name}")
        print(f"Error converting {ogg_file_name}: {e}")
        logging.error(f"Error converting {ogg_file_name}: {e}")
        logging.error(f"FFmpeg stderr: {e.stderr}")  # Log ffmpeg error output
        logging.error(f"FFmpeg stdout: {e.stdout}")
    except Exception as e:
        # Catch any other unexpected errors during the process
        if status_label:
            status_label.config(text=f"Unexpected error for {ogg_file_name}")
        print(f"Unexpected error processing {ogg_file_name}: {e}")
        logging.exception(
            f"Unexpected error processing {ogg_file_name}"
        )  # Log full traceback
    finally:
        # Cleanup OGG file regardless of success or failure
        if os.path.exists(ogg_path):
            try:
                os.remove(ogg_path)
                logging.info(f"Removed temporary OGG file: {ogg_path}")
            except OSError as e:
                logging.error(f"Error removing temporary OGG file {ogg_path}: {e}")

    return None  # Return None if any error occurred


def fetch_character_list_from_api():
    """Fetches a list of character names from the jmp.blue API."""
    api_url = f"{JMP_API_URL_BASE}/characters"
    try:
        logging.info("Fetching character list from jmp.blue API...")
        response = requests.get(api_url, timeout=10)
        response.raise_for_status()
        # The API returns a list of character slugs/IDs
        character_slugs = response.json()
        character_names = []
        logging.info(f"Found {len(character_slugs)} character slugs. Fetching names...")

        # Fetch details for each character to get the name
        # This can be slow if there are many characters. Consider caching or alternative API endpoints if available.
        for slug in character_slugs:
            char_details_url = f"{JMP_API_URL_BASE}/characters/{slug}"
            try:
                char_response = requests.get(char_details_url, timeout=5)
                char_response.raise_for_status()
                details = char_response.json()
                if "name" in details:
                    character_names.append(details["name"])
                else:
                    logging.warning(
                        f"Character details for slug '{slug}' missing 'name' field."
                    )
            except requests.exceptions.RequestException as e:
                logging.warning(
                    f"Could not fetch details for character slug '{slug}': {e}"
                )
            except json.JSONDecodeError as e:
                logging.warning(
                    f"Could not decode JSON for character slug '{slug}': {e}"
                )

        logging.info(f"Successfully fetched {len(character_names)} character names.")
        return sorted(list(set(character_names)))  # Return sorted unique names

    except requests.exceptions.Timeout:
        print("Timeout fetching character list from API.")
        logging.error("Timeout fetching character list from API.")
    except requests.exceptions.RequestException as e:
        print(f"Error fetching character list from API: {e}")
        logging.error(f"Error fetching character list from API: {e}")
    except json.JSONDecodeError as e:
        print(f"Error decoding character list from API: {e}")
        logging.error(f"Error decoding character list from API: {e}")
    return []  # Return empty list on error


def transcribe_character_audio(character_output_dir, status_label=None):
    """
    Transcribes all WAV audio files in the specific character's output directory
    using SpeechToText and saves the results in 'metadata.csv' within that directory.

    Args:
        character_output_dir: The directory containing the character's audio files.
        status_label: Optional Tkinter label for status updates.
    """
    # Check if SpeechToText is available
    if SpeechToText is None:
        msg = "Transcription skipped: SpeechToText could not be imported from voice_tools.py."
        logging.error(msg)
        if status_label:
            status_label.config(text=msg)
        print(msg)
        return

    metadata_path = os.path.join(character_output_dir, "metadata.csv")
    logging.info(f"Starting transcription for directory: {character_output_dir}")
    if status_label:
        status_label.config(
            text=f"Starting transcription in {os.path.basename(character_output_dir)}..."
        )

    # Check if metadata exists and count lines to potentially resume
    existing_files = set()
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, "r", encoding="utf-8") as mf:
                lines = mf.readlines()
                if len(lines) > 1:  # Check if more than just header exists
                    logging.info(
                        f"Found existing metadata file with {len(lines)-1} entries. Will append new transcriptions."
                    )
                    for line in lines[1:]:  # Skip header
                        # Use pipe delimiter consistent with LJSpeech format
                        parts = line.strip().split("|", 1)
                        if len(parts) == 2:
                            existing_files.add(parts[0])
        except Exception as e:
            logging.error(f"Error reading existing metadata file {metadata_path}: {e}")
            # Proceed assuming no existing data or overwrite

    files_to_transcribe = []
    try:
        for file in os.listdir(character_output_dir):
            if file.lower().endswith(".wav") and file not in existing_files:
                files_to_transcribe.append(file)
    except FileNotFoundError:
        msg = f"Transcription skipped: Directory not found {character_output_dir}"
        logging.error(msg)
        if status_label:
            status_label.config(text=msg)
        print(msg)
        return
    except Exception as e:
        msg = (
            f"Transcription skipped: Error listing files in {character_output_dir}: {e}"
        )
        logging.error(msg)
        if status_label:
            status_label.config(text="Error listing files for transcription.")
        print(msg)
        return

    if not files_to_transcribe:
        logging.info("No new WAV files found to transcribe.")
        if status_label:
            status_label.config(text="Transcription: No new files found.")
        return

    logging.info(f"Found {len(files_to_transcribe)} new WAV files to transcribe.")
    transcribed_count = 0
    failed_count = 0

    # Open metadata file in append mode ('a') or write mode ('w') if it didn't exist or was empty
    file_mode = "a" if len(existing_files) > 0 else "w"
    try:
        with open(metadata_path, file_mode, encoding="utf-8", newline="") as mf:
            # Write header only if creating a new file
            if file_mode == "w":
                mf.write("audio_file|text|normalized_text\n")  # Use pipe separator for LJSpeech format

            for i, file in enumerate(files_to_transcribe):
                wav_path = os.path.join(character_output_dir, file)
                progress_text = f"Transcribing {os.path.basename(character_output_dir)} ({i+1}/{len(files_to_transcribe)}): {file}"
                logging.info(progress_text)
                if status_label:
                    status_label.config(text=progress_text)

                try:
                    # Initialize SpeechToText for the specific file
                    # Using Whisper 'base' model as default - requires installation
                    stt = SpeechToText(
                        use_microphone=False,
                        audio_file=wav_path,
                        engine="whisper",  # Defaulting to whisper
                        whisper_model_size="base",  # Defaulting to base model
                    )
                    # process_audio handles reading the file and transcribing
                    audio_transcript = stt.process_audio(
                        language="en"
                    )  # Assuming English

                    if audio_transcript:
                        # Clean transcript: remove potential pipe characters and leading/trailing whitespace
                        cleaned_transcript = audio_transcript.replace("|", " ").strip()
                        # Format for LJSpeech (relative path | transcript | normalized transcript)
                        metadata_entry = f"{file}|{cleaned_transcript}|{cleaned_transcript.lower().replace('.', '').replace(',', '')}"
                        mf.write(metadata_entry + "\n")
                        logging.info(f"Transcription saved for: {file}")
                        transcribed_count += 1
                    else:
                        logging.warning(
                            f"Failed to transcribe (no text returned): {wav_path}"
                        )
                        # Optionally write a placeholder or skip failed files
                        metadata_entry = f"{file}|<transcription_failed>|<transcription_failed>"
                        mf.write(metadata_entry + "\n")
                        failed_count += 1
                except ImportError as imp_err:
                    # Catch specific error if whisper is selected but not installed
                    if "whisper" in str(imp_err).lower():
                        msg = "Transcription Error: 'openai-whisper' not installed. Please run 'pip install -U openai-whisper'."
                        logging.error(msg)
                        if status_label:
                            status_label.config(text=msg)
                        print(msg)
                        # Stop further transcription attempts if library is missing
                        failed_count += (
                            len(files_to_transcribe) - i
                        )  # Mark remaining as failed
                        break
                    else:
                        # Handle other import errors if necessary
                        logging.error(
                            f"Import error during transcription for {wav_path}: {imp_err}",
                            exc_info=True,
                        )
                        metadata_entry = f"{file}|<transcription_error_import>|<transcription_error_import>"
                        mf.write(metadata_entry + "\n")
                        failed_count += 1
                except Exception as e:
                    logging.error(
                        f"Error during transcription for {wav_path}: {e}", exc_info=True
                    )
                    # Optionally write a placeholder for errors
                    metadata_entry = f"{file}|<transcription_error>|<transcription_error>"
                    mf.write(metadata_entry + "\n")
                    failed_count += 1
                finally:
                    # Force update of the GUI label if provided
                    if status_label and "window" in globals() and window:
                        window.update_idletasks()

        final_status = f"Transcription complete for {os.path.basename(character_output_dir)}. Successful: {transcribed_count}, Failed/Empty: {failed_count}."
        logging.info(final_status)
        if status_label:
            status_label.config(text=final_status)

    except Exception as e:
        final_status = f"Error writing to metadata file {metadata_path}: {e}"
        logging.error(final_status, exc_info=True)
        if status_label:
            status_label.config(text="Error during transcription.")


def validate_metadata_existence(character_output_dir):
    """
    Validate the existence of the metadata file after transcription.

    Args:
        character_output_dir (str): Path to the character's output directory.

    Returns:
        bool: True if the metadata file exists, False otherwise.
    """
    metadata_path = os.path.join(character_output_dir, "metadata.csv")
    if not os.path.exists(metadata_path):
        logging.warning(f"Metadata file does not exist: {metadata_path}")
        return False
    return True


def process_character_voices(
    character, language, base_output_dir, download_wiki_audio=True, status_label=None
):
    """
    Downloads and converts audio files for a specific character into their dedicated folder.

    Args:
        character: The name of the character.
        language: The language (used for Wiki category, e.g., "English").
        base_output_dir: The base directory where all character folders will be created.
        download_wiki_audio: Whether to download audio files from the Wiki.
        status_label: A Tkinter label to update the status.

    Returns:
        The path to the character's specific output directory, or None on error.
    """
    # Sanitize character name for folder creation
    safe_character_name = re.sub(r'[\\/*?:"<>|]', "_", character)
    character_folder = os.path.join(base_output_dir, safe_character_name)

    try:
        os.makedirs(character_folder, exist_ok=True)
        logging.info(f"Ensured character directory exists: {character_folder}")
    except OSError as e:
        error_msg = f"Error creating directory {character_folder}: {e}"
        print(error_msg)
        logging.error(error_msg)
        if status_label:
            status_label.config(text="Error creating directory.")
        return None

    if not download_wiki_audio:
        logging.info("Skipping Wiki audio download as requested.")
        if status_label:
            status_label.config(text="Skipped Wiki download.")
        return character_folder  # Return path even if not downloading

    # --- Wiki Download Logic ---
    category_prefix_map = {
        "English": "",  # English category doesn't usually have a prefix
        "Japanese": "Japanese",
        "Chinese": "Chinese",
        "Korean": "Korean",
    }
    category_prefix = category_prefix_map.get(language)

    # Handle cases where language might not map directly or is English
    if language == "English":
        # Try both with and without "English" prefix as structure varies
        categories_to_try = [
            f"Category:{character} Voice-Overs",
            f"Category:English {character} Voice-Overs",
        ]
    elif category_prefix is not None:
        categories_to_try = [f"Category:{category_prefix} {character} Voice-Overs"]
    else:
        error_msg = f"Invalid language '{language}' selected for Wiki category lookup."
        if status_label:
            status_label.config(text=error_msg)
        print(error_msg)
        logging.warning(error_msg)
        return character_folder  # Return path, but downloads might fail

    files_to_download = []
    for category in categories_to_try:
        logging.info(f"Checking Wiki category: {category}")
        files = get_category_files(category)
        if files:
            logging.info(f"Found {len(files)} files in {category}.")
            files_to_download.extend(files)
            # Optional: break after finding the first non-empty category if preferred
            # break
        else:
            logging.info(f"No files found in {category}.")

    if not files_to_download:
        status_msg = f"No voice files found on Wiki for {character} ({language})."
        if status_label:
            status_label.config(text=status_msg)
        print(status_msg)
        logging.info(status_msg)
        return character_folder  # Still return path, just no files downloaded

    # Remove duplicates if categories overlapped
    unique_files = sorted(list(set(files_to_download)))
    logging.info(f"Total unique files to process from Wiki: {len(unique_files)}")

    downloaded_count = 0
    failed_count = 0
    skipped_count = 0

    for i, file_title in enumerate(unique_files):
        progress_text = (
            f"Processing Wiki file ({i+1}/{len(unique_files)}): {file_title}"
        )
        logging.info(progress_text)
        # Don't update status label here, download_and_convert does it

        # Extract base filename from title (e.g., "File:VO_{character}_{line}.ogg" -> "VO_{character}_{line}.ogg")
        file_name_match = re.match(r"File:(.*)", file_title)
        if not file_name_match:
            logging.warning(f"Could not extract filename from title: {file_title}")
            failed_count += 1
            continue
        file_name = file_name_match.group(1).strip()

        file_url = get_file_url(file_title)
        if file_url:
            wav_file_path = download_and_convert(
                file_url, character_folder, file_name, status_label=status_label
            )
            if wav_file_path:
                downloaded_count += 1
            else:
                # Check if it was skipped vs failed
                wav_file_name = file_name.replace(".ogg", ".wav").replace(
                    ".OGG", ".wav"
                )
                if os.path.exists(os.path.join(character_folder, wav_file_name)):
                    skipped_count += 1
                else:
                    failed_count += 1
        else:
            logging.warning(f"No URL found for {file_title} on Wiki.")
            failed_count += 1
            if status_label:
                status_label.config(text=f"No URL for: {file_name}")

        # Force GUI update
        if status_label and "window" in globals() and window:
            window.update_idletasks()

    final_status = f"Wiki download for {character} ({language}) complete. Downloaded: {downloaded_count}, Skipped: {skipped_count}, Failed: {failed_count}."
    logging.info(final_status)
    if status_label:
        status_label.config(text=final_status)
    print(final_status)

    # After processing all files, update metadata.csv with speaker_id
    metadata_path = os.path.join(character_folder, "metadata.csv")
    if os.path.exists(metadata_path):
        with open(metadata_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        with open(metadata_path, "w", encoding="utf-8") as f:
            for i, line in enumerate(lines):
                if i == 0:
                    # Update header
                    f.write("audio_file|text|speaker_id\n")
                else:
                    # Append speaker_id to each line
                    f.write(line.strip() + "|speaker_1\n")

        logging.info(f"Updated metadata.csv with speaker_id for {character}.")

    return character_folder


# Add reattempt_transcription function
def reattempt_transcription(character_output_dir):
    """
    Re-attempts transcription for audio files marked as 'Validation Needed' in the metadata.

    Args:
        character_output_dir: The directory containing the character's audio files.
    """
    metadata_path = os.path.join(character_output_dir, "metadata.csv")
    voiceless_dir = os.path.join(character_output_dir, "voiceless")
    if not os.path.exists(metadata_path):
        logging.warning(f"Metadata file does not exist: {metadata_path}")
        return

    # Create voiceless directory if it doesn't exist
    if not os.path.exists(voiceless_dir):
        os.makedirs(voiceless_dir)

    try:
        with open(metadata_path, "r", encoding="utf-8") as mf:
            lines = mf.readlines()

        updated_lines = [lines[0]]  # Keep the header

        for line in lines[1:]:  # Skip header
            parts = line.strip().split("|")
            if len(parts) >= 3 and "Validation Needed" in parts[1]:
                wav_path = os.path.join(character_output_dir, parts[0])
                try:
                    stt = SpeechToText(
                        use_microphone=False,
                        audio_file=wav_path,
                        engine="whisper",
                        whisper_model_size="base",
                    )
                    audio_transcript = stt.process_audio(language="en")

                    if audio_transcript:
                        cleaned_transcript = audio_transcript.replace("|", " ").strip()
                        metadata_entry = f"{parts[0]}|{cleaned_transcript}|{cleaned_transcript.lower().replace('.', '').replace(',', '')}"
                        updated_lines.append(metadata_entry + "\n")
                        logging.info(f"Re-transcription saved for: {parts[0]}")
                    else:
                        logging.warning(
                            f"Re-transcription failed (no text returned): {wav_path}"
                        )
                        # Move file to voiceless directory
                        shutil.move(wav_path, os.path.join(voiceless_dir, parts[0]))
                except Exception as e:
                    logging.error(
                        f"Error during re-transcription for {wav_path}: {e}",
                        exc_info=True,
                    )
            else:
                updated_lines.append(line)

        # Write updated metadata back to file
        with open(metadata_path, "w", encoding="utf-8") as mf_update:
            mf_update.writelines(updated_lines)

    except Exception as e:
        logging.error(f"Error reading metadata file for re-transcription: {e}")


# Add validate_metadata function
def validate_metadata(metadata_path):
    """
    Validate the metadata file for placeholder text and log warnings.

    Args:
        metadata_path (str): Path to the metadata file.
    """
    warnings = []
    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="|")
            for line_number, row in enumerate(reader, start=1):
                if len(row) < 2 or row[1].strip() == "<transcription_failed>":
                    warnings.append((line_number, row[0]))

        if warnings:
            logging.warning("[Warning] The following entries in the metadata file contain placeholder text:")
            for line_number, audio_file in warnings:
                logging.warning(f"  [Line {line_number}] {audio_file} ('<transcription_failed>')")
            logging.warning("Please reprocess or manually fix these entries.")
        else:
            logging.info("Metadata validation complete. No placeholder text found.")

    except FileNotFoundError:
        logging.error(f"Error: Metadata file not found at {metadata_path}.")
    except Exception as e:
        logging.error(f"Error validating metadata: {e}")


# --- GUI ---
window = None  # Global variable for main window (needed for update_idletasks)


def main_gui():
    global window
    window = tk.Tk()
    window.title("Genshin Impact Voice Downloader & Transcriber")

    # --- Configuration Frame ---
    config_frame = ttk.LabelFrame(window, text="Configuration")
    config_frame.grid(row=0, column=0, padx=10, pady=10, sticky="ew")

    # Language Selection
    language_label = ttk.Label(config_frame, text="Select Language (for Wiki):")
    language_label.grid(row=0, column=0, padx=5, pady=5, sticky="w")
    languages = ["English", "Japanese", "Chinese", "Korean"]
    language_var = tk.StringVar(window)
    language_var.set(languages[0])  # Default to English
    language_dropdown = ttk.Combobox(
        config_frame,
        textvariable=language_var,
        values=languages,
        state="readonly",
        width=15,
    )
    language_dropdown.grid(row=0, column=1, padx=5, pady=5, sticky="w")

    # Character Selection
    character_label = ttk.Label(config_frame, text="Select Character:")
    character_label.grid(row=1, column=0, padx=5, pady=5, sticky="w")
    character_names = fetch_character_list_from_api()  # Fetch names on startup
    character_var = tk.StringVar(window)
    character_dropdown = ttk.Combobox(
        config_frame,
        textvariable=character_var,
        values=character_names,
        state="readonly",
        width=30,
    )
    if character_names:
        character_var.set(character_names[0])  # Default to first character
    else:
        character_var.set("Could not fetch characters")
        character_dropdown.config(state="disabled")
    character_dropdown.grid(row=1, column=1, columnspan=2, padx=5, pady=5, sticky="ew")

    # Output Directory
    output_dir_label = ttk.Label(config_frame, text="Base Output Directory:")
    output_dir_label.grid(row=2, column=0, padx=5, pady=5, sticky="w")
    output_dir_var = tk.StringVar(window, value=BASE_DATA_DIR)  # Use constant default
    output_dir_entry = ttk.Entry(config_frame, textvariable=output_dir_var, width=40)
    output_dir_entry.grid(row=2, column=1, columnspan=2, padx=5, pady=5, sticky="ew")

    # Download Options
    download_options_label = ttk.Label(config_frame, text="Options:")
    download_options_label.grid(row=3, column=0, padx=5, pady=5, sticky="w")
    download_wiki_audio_var = tk.BooleanVar(window, True)
    download_wiki_audio_check = ttk.Checkbutton(
        window, text="Download Wiki Audio", variable=download_wiki_audio_var
    )
    # Place checkbutton inside the config_frame
    download_wiki_audio_check.grid(row=3, column=1, padx=5, pady=2, sticky="w")

    # --- Control Frame ---
    control_frame = ttk.Frame(window)
    control_frame.grid(row=1, column=0, padx=10, pady=5, sticky="ew")

    # Status Label
    status_label = ttk.Label(
        control_frame, text="Ready", relief=tk.SUNKEN, anchor="w", width=60
    )
    status_label.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5, pady=5)

    # Action Button (Function Definition)
    def start_processing():
        character = character_var.get()
        language = language_var.get()
        base_output_dir = output_dir_var.get()
        download_wiki_audio = download_wiki_audio_var.get()

        if not character or character == "Could not fetch characters":
            messagebox.showerror("Error", "Please select a valid character.")
            return
        if not base_output_dir:
            messagebox.showerror("Error", "Please enter a base output directory.")
            return

        # Disable button during processing
        download_button.config(state="disabled")
        status_label.config(text=f"Starting processing for {character}...")
        window.update_idletasks()  # Ensure GUI updates

        # 1. Process Voice Data (Download/Convert)
        character_folder_path = process_character_voices(
            character, language, base_output_dir, download_wiki_audio, status_label
        )

        # 2. Transcribe if processing was successful (or folder exists)
        if character_folder_path and os.path.isdir(character_folder_path):
            transcribe_character_audio(character_folder_path, status_label)

            # Validate metadata existence
            if not validate_metadata_existence(character_folder_path):
                status_label.config(text=f"Metadata file missing for {character}. Please check transcription.")
                logging.error(f"Metadata file missing for {character} after transcription.")
                return

            # Run the retrainer after transcription
            try:
                retrainer_command = (
                    f"python genshin_voice_retranscriber.py --character_output_dir \"{character_folder_path}\""
                )
                subprocess.run(retrainer_command, shell=True, check=True)
                status_label.config(text=f"Retraining completed for {character}.")
            except subprocess.CalledProcessError as e:
                status_label.config(text=f"Retraining failed: {e}")
                logging.error(f"Retraining failed for {character}: {e}")
        else:
            status_label.config(text="Skipping transcription and retraining due to previous errors.")
            logging.warning("Skipping transcription and retraining due to download/folder errors.")

        # Re-enable button
        download_button.config(state="normal")
        # Final status is set by the last function called

    # Action Button (Creation and Placement - THIS WAS MISSING)
    download_button = ttk.Button(
        control_frame, text="Process Character Voices", command=start_processing
    )
    download_button.pack(
        side=tk.RIGHT, padx=5, pady=5
    )  # Add the button to the control frame

    # Make window columns/rows responsive if needed
    window.grid_columnconfigure(0, weight=1)
    config_frame.grid_columnconfigure(
        1, weight=1
    )  # Allow character dropdown/output dir to expand
    config_frame.grid_columnconfigure(2, weight=1)
    # control_frame.pack_propagate(False) # Prevent control frame from shrinking - might be needed depending on content

    window.mainloop()


# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download and transcribe Genshin Impact voice data for a character."
    )
    parser.add_argument(
        "--character",
        type=str,
        help="Name of the character (e.g., Amber). If omitted, GUI will launch.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=BASE_DATA_DIR,
        help=f"Base directory to save character voice data (default: {BASE_DATA_DIR})",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="English",
        choices=["English", "Japanese", "Chinese", "Korean"],
        help="Language for Wiki voice-over category (default: English)",
    )
    parser.add_argument(
        "--skip_wiki_download",
        action="store_true",
        help="Skip downloading audio from the Wiki (only perform transcription if files exist).",
    )
    parser.add_argument(
        "--skip_transcription",
        action="store_true",
        help="Skip the transcription step.",
    )

    args = parser.parse_args()

    # Launch GUI if no character is specified via CLI
    if args.character is None:
        main_gui()
    else:
        # Run from Command Line
        print(f"--- Running in Command Line Mode for Character: {args.character} ---")
        logging.info(f"Starting CLI processing for character: {args.character}")

        # 1. Process Voice Data (Download/Convert)
        character_folder_path = process_character_voices(
            args.character,
            args.language,
            args.output_dir,
            download_wiki_audio=(
                not args.skip_wiki_download
            ),  # Pass True if not skipped
        )

        # 2. Transcribe if processing was successful and not skipped
        if not args.skip_transcription:
            if character_folder_path and os.path.isdir(character_folder_path):
                transcribe_character_audio(character_folder_path)
            else:
                print(
                    "Skipping transcription because character folder path is invalid or download failed."
                )
                logging.warning("Skipping transcription due to download/folder errors.")
        else:
            print("Skipping transcription as requested by command line argument.")
            logging.info(
                "Skipping transcription as requested by command line argument."
            )

        print(f"--- Finished CLI processing for Character: {args.character} ---")
        logging.info(f"Finished CLI processing for character: {args.character}")
