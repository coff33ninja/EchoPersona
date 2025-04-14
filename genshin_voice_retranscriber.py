import os
import logging
import argparse
from voice_tools import SpeechToText
import shutil
import csv


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
            print("[Warning] The following entries in the metadata file contain placeholder text:")
            for line_number, audio_file in warnings:
                print(f"  [Line {line_number}] {audio_file} ('<transcription_failed>')")
            print("Please reprocess or manually fix these entries.")
        else:
            print("Metadata validation complete. No placeholder text found.")

    except FileNotFoundError:
        print(f"Error: Metadata file not found at {metadata_path}.")
    except Exception as e:
        print(f"Error validating metadata: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Re-attempt transcription for Genshin Impact voice data."
    )
    parser.add_argument(
        "--character_output_dir",
        type=str,
        required=True,
        help="Path to the character's output directory containing the metadata.csv file.",
    )
    args = parser.parse_args()

    reattempt_transcription(args.character_output_dir)
