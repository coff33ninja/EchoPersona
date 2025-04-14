import os
import logging
from voice_tools import SpeechToText

class Retranscriber:
    def __init__(self, character_output_dir):
        self.character_output_dir = character_output_dir
        self.metadata_path = os.path.join(character_output_dir, "metadata.csv")
        self.voiceless_dir = os.path.join(character_output_dir, "voiceless")
        os.makedirs(self.voiceless_dir, exist_ok=True)

    def reattempt_transcription(self):
        if not os.path.exists(self.metadata_path):
            logging.warning(f"Metadata file does not exist: {self.metadata_path}")
            return

        try:
            with open(self.metadata_path, "r", encoding="utf-8") as mf:
                lines = mf.readlines()

            updated_lines = [lines[0]]  # Keep the header

            for line in lines[1:]:  # Skip header
                parts = line.strip().split("|")
                if len(parts) >= 3 and "Validation Needed" in parts[1]:
                    wav_path = os.path.join(self.character_output_dir, parts[0])
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
                            os.rename(wav_path, os.path.join(self.voiceless_dir, parts[0]))
                    except Exception as e:
                        logging.error(
                            f"Error during re-transcription for {wav_path}: {e}",
                            exc_info=True,
                        )
                else:
                    updated_lines.append(line)

            # Write updated metadata back to file
            with open(self.metadata_path, "w", encoding="utf-8") as mf_update:
                mf_update.writelines(updated_lines)

        except Exception as e:
            logging.error(f"Error reading metadata file for re-transcription: {e}")