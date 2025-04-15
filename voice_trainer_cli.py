import argparse
import os
import re
import logging

try:
    from voice_tools import VoiceTrainer, BASE_DATASET_DIR, BASE_MODEL_DIR
except ImportError:
    print("Error: voice_tools.py not found. Make sure it's in the same directory.")
    logging.error("Failed to import from voice_tools.py")
    exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Voice Trainer CLI Tool - Manages datasets and training for specific characters.",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # Required Arguments
    parser.add_argument(
        "--character",
        type=str,
        required=True,
        help="Name of the character to work with. This determines the dataset and model paths.",
    )
    parser.add_argument(
        "--action",
        type=str,
        required=True,
        choices=[
            "record",
            "provide",
            "validate",
            "stats",
            "augment",
            "trim",
            "quality",
            "train",
            "test",
            "use",
        ],
        help="""Action to perform:
    --- Dataset Management ---
    record:    Record a new training sample for the character (prompts for text).
    provide:   Add an existing audio file (WAV/MP3) to the character's dataset. Requires --file.
    validate:  Validate the character's metadata file (checks format, file existence).
    stats:     Show statistics for the character's dataset.
    augment:   Apply random augmentation to a specific audio file in the dataset. Requires --file (relative path).
    trim:      Trim leading/trailing silence from a specific audio file in the dataset. Requires --file (relative path).
    quality:   Perform a basic quality check on a specific audio file in the dataset. Requires --file (relative path).
    --- Training & Usage ---
    train:     Start training a TTS model for the character using their dataset.
    test:      Test the character's trained TTS model. Requires --text.
    use:       Generate speech using the character's trained TTS model. Requires --text.
""",
    )

    # Optional Arguments
    parser.add_argument("--text", type=str, help="Text for 'test' or 'use' actions.")
    parser.add_argument(
        "--file",
        type=str,
        help="Path to an audio file. Usage depends on action:\n"
        "  provide: Full path to the source WAV/MP3 file.\n"
        "  augment, trim, quality: RELATIVE path within the character's dataset (e.g., 'sample_char_time.wav').",
    )
    parser.add_argument(
        "--base_dataset_dir",
        type=str,
        default=BASE_DATASET_DIR,
        help=f"Base directory containing all character datasets (default: {BASE_DATASET_DIR})",
    )
    parser.add_argument(
        "--base_model_dir",
        type=str,
        default=BASE_MODEL_DIR,
        help=f"Base directory containing all trained character models (default: {BASE_MODEL_DIR})",
    )

    # Training-specific Arguments
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of epochs for training (default: 100).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for training (default: 16).",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.001,
        help="Learning rate for training (default: 0.001).",
    )
    parser.add_argument(
        "--continue_path",
        type=str,
        default=None,
        help="Path to a previous checkpoint to continue training from (e.g., path/to/run-XXX).",
    )
    # Optional: Add more parameters for flexibility
    parser.add_argument(
        "--sample_rate",
        type=int,
        default=22050,
        help="Target sample rate for training (default: 22050).",
    )
    parser.add_argument(
        "--use_phonemes",
        type=str,
        default="true",
        choices=["true", "false"],
        help="Use phonemes for training (default: true).",
    )
    parser.add_argument(
        "--phoneme_language",
        type=str,
        default="en-us",
        help="Phoneme language if phonemes are used (default: en-us).",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="false",
        choices=["true", "false"],
        help="Use mixed precision training (default: false).",
    )

    args = parser.parse_args()

    # Input Validation
    if args.action in ["test", "use"] and not args.text:
        parser.error(f"--text is required for action '{args.action}'.")
    if args.action in ["provide", "augment", "trim", "quality"] and not args.file:
        parser.error(f"--file is required for action '{args.action}'.")
    if args.use_phonemes == "true" and not args.phoneme_language:
        parser.error("--phoneme_language is required when --use_phonemes is true.")

    # Initialize Trainer
    try:
        safe_character_name = re.sub(r'[\\/*?:"<>|]', "_", args.character)
        if safe_character_name != args.character:
            print(
                f"Warning: Character name sanitized to '{safe_character_name}' for path usage."
            )
        print(f"Initializing trainer for character: '{safe_character_name}'...")
        trainer = VoiceTrainer(
            character_name=safe_character_name,
            base_dataset_dir=args.base_dataset_dir,
            base_model_dir=args.base_model_dir,
        )
    except ValueError as ve:
        print(f"Error initializing trainer: {ve}")
        exit(1)
    except Exception as e:
        print(f"An unexpected error occurred during trainer initialization: {e}")
        logging.exception("Trainer initialization failed.")
        exit(1)

    # Perform Action
    print(
        f"\nPerforming action '{args.action}' for character '{trainer.character_name}'..."
    )

    # Pre-checks
    actions_requiring_dataset = [
        "validate",
        "stats",
        "augment",
        "trim",
        "quality",
        "train",
    ]
    if args.action in actions_requiring_dataset and not os.path.isdir(
        trainer.dataset_path
    ):
        print(
            f"Error: Dataset directory does not exist for character '{trainer.character_name}': {trainer.dataset_path}"
        )
        print("Please use 'record' or 'provide' first to create the dataset.")
        exit(1)

    actions_requiring_model = ["test", "use"]
    if args.action in actions_requiring_model:
        model_file = os.path.join(trainer.output_path, "best_model.pth")
        config_file = os.path.join(trainer.output_path, "config.json")
        if not os.path.exists(model_file) or not os.path.exists(config_file):
            print(
                f"Error: Trained model files ('best_model.pth', 'config.json') not found for character '{trainer.character_name}' in {trainer.output_path}"
            )
            print(
                "Please ensure training was completed successfully using the 'train' action."
            )
            exit(1)

    actions_requiring_file_in_dataset = ["augment", "trim", "quality"]
    if args.action in actions_requiring_file_in_dataset:
        target_file_path = os.path.join(trainer.dataset_path, args.file)
        if not os.path.exists(target_file_path):
            print(
                f"Error: Specified file '{args.file}' not found within the dataset directory: {trainer.dataset_path}"
            )
            exit(1)

    # Execute Action
    try:
        if args.action == "record":
            trainer.record_training_sample()
        elif args.action == "provide":
            trainer.provide_voice_data(args.file)
        elif args.action == "validate":
            trainer.validate_metadata()
        elif args.action == "stats":
            trainer.dataset_statistics()
        elif args.action == "augment":
            trainer.augment_audio(args.file)
        elif args.action == "trim":
            trainer.trim_silence(args.file)
        elif args.action == "quality":
            trainer.check_audio_quality(args.file)
        elif args.action == "train":
            trainer.train_voice(
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                continue_path=args.continue_path,
                sample_rate=args.sample_rate,
                use_phonemes=args.use_phonemes.lower() == "true",
                phoneme_language=args.phoneme_language,
                mixed_precision=args.mixed_precision.lower() == "true",
            )
        elif args.action == "test":
            trainer.test_trained_voice(args.text)
        elif args.action == "use":
            trainer.use_trained_voice(args.text)
        else:
            print(f"Error: Unknown action '{args.action}'.")
        print(
            f"\nAction '{args.action}' completed successfully for character '{trainer.character_name}'."
        )
    except FileNotFoundError as fnf_error:
        print(f"\nError: File not found during action '{args.action}': {fnf_error}")
        logging.error(f"FileNotFoundError during action {args.action}: {fnf_error}")
    except Exception as e:
        print(f"\nAn unexpected error occurred during action '{args.action}': {e}")
        logging.exception(f"Error during action {args.action}")


if __name__ == "__main__":
    main()
