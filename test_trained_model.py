import os
import argparse
import logging
import torch
from TTS.api import TTS as CoquiTTS # Use the API for easier loading

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Constants ---
BASE_MODEL_DIR = "trained_models"   # Base directory where character models are saved

# --- Argument Parser ---
def parse_arguments():
    parser = argparse.ArgumentParser(description="Test a trained TTS model for a specific character.")
    parser.add_argument("--character", type=str, required=True,
                        help="Name of the character whose trained model you want to test.")
    parser.add_argument("--text", type=str, required=True,
                        help="Text to synthesize using the character's model.")
    parser.add_argument("--output_file", type=str, default=None,
                        help="Optional: Path to save the generated WAV file. If not provided, defaults to 'character_test_output.wav'.")
    parser.add_argument("--base_model_dir", type=str, default=BASE_MODEL_DIR,
                        help=f"Base directory containing all trained character models (default: {BASE_MODEL_DIR})")
    # Add argument for GPU preference if needed
    parser.add_argument("--cpu", action="store_true", help="Force using CPU even if GPU is available.")

    return parser.parse_args()

# --- Main Function ---
def main():
    args = parse_arguments()

    # --- Determine Paths ---
    # Sanitize character name similar to how VoiceTrainer does it
    character_name_sanitized = args.character # Assume CLI passes sanitized name or handle here if needed
    character_model_dir = os.path.join(args.base_model_dir, character_name_sanitized)

    # Expecting standard output structure from training (best_model.pth, config.json)
    trained_model_path = os.path.join(character_model_dir, "best_model.pth")
    trained_config_path = os.path.join(character_model_dir, "config.json")

    # Determine output filename
    output_filename = args.output_file
    if not output_filename:
        output_filename = f"{character_name_sanitized}_test_output.wav"
        # Save default output in the character's model directory for organization
        output_path = os.path.join(character_model_dir, output_filename)
    else:
        # If user specified a path, use it directly
        output_path = output_filename
        # Ensure the directory for the custom output path exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)


    # --- Check if Model Files Exist ---
    if not os.path.exists(trained_model_path) or not os.path.exists(trained_config_path):
        logging.error(f"Trained model files not found for character '{args.character}' in directory: {character_model_dir}")
        logging.error(f"Expected model: {trained_model_path}")
        logging.error(f"Expected config: {trained_config_path}")
        print("Error: Trained model files not found. Please ensure training completed successfully.")
        return

    # --- Load Trained Model using Coqui TTS API ---
    print(f"\n--- Loading Custom Trained TTS Model for Character: {args.character} ---")
    logging.info(f"Model Path: {trained_model_path}")
    logging.info(f"Config Path: {trained_config_path}")

    if CoquiTTS is None:
         print("Error: Coqui TTS library not found. Please install it: pip install TTS")
         logging.critical("Coqui TTS library not found.")
         return

    try:
        # Determine GPU usage
        use_gpu = torch.cuda.is_available() and not args.cpu
        logging.info(f"Initializing TTS model (GPU: {use_gpu})")

        # Initialize TTS engine with the trained model paths
        tts_engine = CoquiTTS(
            model_path=trained_model_path,
            config_path=trained_config_path,
            gpu=use_gpu
        )
        logging.info("TTS model loaded successfully.")

    except Exception as e:
        logging.error(f"An error occurred while loading the custom trained model: {e}", exc_info=True)
        print(f"Error loading model: {e}")
        return

    # --- Synthesize Speech ---
    print(f"Synthesizing text: '{args.text[:100]}...'") # Print snippet
    logging.info(f"Synthesizing to output file: {output_path}")

    try:
        # Generate speech using tts_to_file
        tts_engine.tts_to_file(
            text=args.text,
            file_path=output_path
            # speaker=None, # Not needed for single-speaker models loaded this way
            # language=None, # Language should be inferred from the model config
        )
        print(f"\nCustom trained speech saved successfully to: {output_path}")
        logging.info(f"Speech synthesis complete for character '{args.character}'.")

        # Optionally play the audio
        play_choice = input("Play the generated audio? (y/n): ").lower()
        if play_choice == 'y':
             from voice_tools import play_audio # Import playback helper
             play_audio(output_path)


    except Exception as e:
        logging.error(f"An error occurred during speech synthesis: {e}", exc_info=True)
        print(f"Error during synthesis: {e}")


if __name__ == "__main__":
    main()
