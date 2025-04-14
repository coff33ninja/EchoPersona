import os
import argparse
import logging
from trainer import Trainer, TrainerArgs

# Removed CharactersConfig from this import as it's likely moved/unused
from TTS.config import BaseAudioConfig, BaseDatasetConfig

# Removed BaseTTSConfig import
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.datasets import load_tts_samples

# Removed VitsArgs import
from TTS.tts.models.vits import Vits
from TTS.utils.audio import AudioProcessor

# Removed ModelManager import as it wasn't used for training setup
# from TTS.utils.manage import ModelManager # Keep if needed elsewhere, but not for this core logic

# --- Constants ---
METADATA_FILENAME = "metadata.csv"  # Expected metadata filename in dataset folder


# --- Argument Parser ---
def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Train a VITS TTS model for a specific character."
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to the character's dataset directory (containing metadata.csv and wav files).",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save the trained model, logs, and other outputs for this character.",
    )
    # Add other training parameters as needed (e.g., epochs, batch_size, learning_rate)
    parser.add_argument(
        "--epochs", type=int, default=1000, help="Number of training epochs."
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Training batch size."
    )
    parser.add_argument(
        "--eval_batch_size", type=int, default=16, help="Evaluation batch size."
    )
    parser.add_argument(
        "--num_loader_workers",
        type=int,
        default=4,
        help="Number of workers for data loading.",
    )
    parser.add_argument(
        "--learning_rate", type=float, default=0.0002, help="Initial learning rate."
    )  # VITS default lr
    parser.add_argument(
        "--language",
        type=str,
        default="en",
        help="Language code for the dataset (e.g., 'en', 'es').",
    )
    parser.add_argument(
        "--text_cleaner",
        type=str,
        default="english_cleaners",
        help="Text cleaner to use.",
    )
    parser.add_argument(
        "--use_phonemes",
        action="store_true",
        default=True,
        help="Use phonemes for training.",
    )  # Default to True for VITS
    parser.add_argument(
        "--no_phonemes",
        action="store_false",
        dest="use_phonemes",
        help="Do not use phonemes for training.",
    )
    parser.add_argument(
        "--phoneme_language",
        type=str,
        default="en-us",
        help="Phoneme language (if using phonemes).",
    )
    parser.add_argument(
        "--sample_rate", type=int, default=22050, help="Target sample rate."
    )
    parser.add_argument(
        "--run_eval",
        action="store_true",
        default=True,
        help="Run evaluation during training.",
    )
    parser.add_argument(
        "--no_eval",
        action="store_false",
        dest="run_eval",
        help="Do not run evaluation during training.",
    )
    parser.add_argument(
        "--mixed_precision",
        action="store_true",
        default=True,
        help="Use mixed precision training.",
    )
    parser.add_argument(
        "--no_mixed_precision",
        action="store_false",
        dest="mixed_precision",
        help="Do not use mixed precision training.",
    )
    parser.add_argument(
        "--continue_path",
        type=str,
        default=None,
        help="Path to a previous training output directory to continue from.",
    )

    return parser.parse_args()


# --- Main Training Function ---
def main():
    args = parse_arguments()

    # --- Validate Paths ---
    if not os.path.isdir(args.dataset_path):
        logging.error(
            f"Dataset path not found or is not a directory: {args.dataset_path}"
        )
        return
    metadata_path = os.path.join(args.dataset_path, METADATA_FILENAME)
    if not os.path.isfile(metadata_path):
        logging.error(f"Metadata file not found in dataset path: {metadata_path}")
        return

    # Ensure output path exists
    os.makedirs(args.output_path, exist_ok=True)
    logging.info(f"Output path: {args.output_path}")

    # --- Configuration ---
    # Use arguments passed from CLI

    # Update logging to use a structured format
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(args.output_path, "training.log")),
            logging.StreamHandler()
        ]
    )

    # Audio Configuration (Adjust defaults if needed)
    audio_config = BaseAudioConfig(
        sample_rate=args.sample_rate,
        resample=False,  # Assume data is already at target sample rate
        num_mels=80,  # VITS default
        log_func="np.log10",
        min_level_db=-100,
        frame_shift_ms=None,
        frame_length_ms=None,
        ref_level_db=20,
        fft_size=1024,  # VITS default
        power=1.5,  # VITS default
        preemphasis=0.97,  # VITS default
        hop_length=256,  # VITS default
        win_length=1024,  # VITS default
    )

    # Dataset Configuration
    # Ensure dataset path uses the character folder directly
    dataset_config = BaseDatasetConfig(
        formatter="ljspeech",
        meta_file_train=METADATA_FILENAME,  # Use the standard filename relative to dataset_path
        path=args.dataset_path,  # Use character folder directly
        language=args.language,
    )

    # --- REMOVED Unused Config Definitions ---
    # characters_config = CharactersConfig(...) # Removed - parameters set in VitsConfig
    # base_tts_config = BaseTTSConfig(...) # Removed - parameters set in VitsConfig

    # VITS Model Configuration
    # Adapt based on Coqui TTS library version and VITS requirements
    config = VitsConfig(
        audio=audio_config,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        num_loader_workers=args.num_loader_workers,
        num_eval_loader_workers=args.num_loader_workers,  # Can be same or different
        run_eval=args.run_eval,
        test_delay_epochs=-1,  # Start eval immediately
        epochs=args.epochs,
        text_cleaner=args.text_cleaner,
        use_phonemes=args.use_phonemes,
        phoneme_language=args.phoneme_language,
        phoneme_cache_path=os.path.join(
            args.output_path, "phoneme_cache"
        ),  # Cache in output dir
        compute_input_seq_cache=True,  # Cache text processing results
        print_step=50,  # Log training step every 50 steps
        print_eval=True,  # Print evaluation results
        mixed_precision=args.mixed_precision,
        output_path=args.output_path,  # CRITICAL: Use the provided output path
        datasets=[dataset_config],  # Pass the configured dataset
        lr=args.learning_rate,
        # Add other VITS specific parameters if needed:
        # use_speaker_embedding=False, # Usually False for single speaker finetuning
        # num_speakers=0, # Set to 0 for single speaker
    )

    # --- REMOVED Unused VitsArgs definition ---
    # vits_args = VitsArgs(...) # Removed - parameters should be in VitsConfig if needed

    # --- Preprocessing Steps (Moved from original script - Handled by VoiceTrainer now) ---
    # trainer = VoiceTrainer() # VoiceTrainer is now handled by the CLI script
    # print("Validating metadata...")
    # trainer.validate_metadata()
    # print("Preprocessing dataset...")
    # ... (augmentation/trimming logic removed as it's done via CLI before training) ...

    # --- Initialize AudioProcessor ---
    # Used for text processing (tokenization, phonemization) based on config
    try:
        ap = AudioProcessor.init_from_config(config)
        logging.info("AudioProcessor initialized successfully.")
    except Exception as e:
        logging.error(f"Failed to initialize AudioProcessor: {e}", exc_info=True)
        return

    # --- Load Dataset Samples ---
    try:
        logging.info(f"Loading dataset samples from: {args.dataset_path}")
        # eval_split_size can be adjusted, e.g., 0.01 for 1% evaluation data
        train_samples, eval_samples = load_tts_samples(
            datasets=[dataset_config],  # Pass the dataset configuration as a list
            eval_split=True,  # Enable evaluation split
            eval_split_size=0.01  # Use 1% of the dataset for evaluation
        )
        if not train_samples:
            logging.error(
                "No training samples loaded. Check metadata file format and content."
            )
            return
        logging.info(f"Loaded {len(train_samples)} training samples.")
        if eval_samples:
            logging.info(f"Loaded {len(eval_samples)} evaluation samples.")
        else:
            logging.warning(
                "No evaluation samples loaded. Evaluation might be skipped."
            )

        # Filter out entries with failed transcriptions
        train_samples = [sample for sample in train_samples if sample.get('text') != '<transcription_failed>']
        eval_samples = [sample for sample in eval_samples if sample.get('text') != '<transcription_failed>']

        # Validate dataset paths
        for sample in train_samples:
            if not os.path.isfile(sample["audio_file"]):
                logging.error(f"Missing audio file: {sample['audio_file']}")
                return

    except Exception as e:
        logging.error(f"Failed to load dataset samples: {e}", exc_info=True)
        return

    # --- REMOVED Unused ModelManager implementation ---
    # model_manager = ModelManager()
    # model_path = model_manager.download_model(...)
    # print(f"Model downloaded to: {model_path}")

    # --- Initialize Model ---
    # Initialize VITS model from the configuration
    try:
        model = Vits.init_from_config(config)
        logging.info("VITS model initialized successfully.")
    except Exception as e:
        logging.error(f"Failed to initialize VITS model: {e}", exc_info=True)
        return

    # --- Initialize Trainer ---
    try:
        # TrainerArgs can be used for fine-grained control over optimizer, scheduler etc.
        # Using default TrainerArgs() for now.
        trainer = Trainer(
            args=TrainerArgs(
                continue_path=args.continue_path
            ),  # Pass continue_path if provided
            config=config,
            output_path=args.output_path,  # Pass output path again
            model=model,
            train_samples=train_samples,
            eval_samples=eval_samples,
            training_assets={
                "audio_processor": ap
            },  # Pass the initialized AudioProcessor
        )
        logging.info("Trainer initialized successfully.")
    except Exception as e:
        logging.error(f"Failed to initialize Trainer: {e}", exc_info=True)
        return

    # --- Start Training ---
    try:
        logging.info(">>> Starting Training <<<")
        if args.continue_path:
            logging.info(
                f"Attempting to continue training from checkpoint specified in TrainerArgs: {args.continue_path}"
            )
        # Trainer's `fit` method handles finding the latest checkpoint in output_path
        # or the one specified via continue_path in TrainerArgs.
        trainer.fit()
        logging.info(">>> Training Finished <<<")
        print(f"\nTraining complete. Model files saved in: {args.output_path}")

    except Exception as e:
        logging.error("An error occurred during training.", exc_info=True)
        print(f"An error occurred during training: {e}")


if __name__ == "__main__":
    main()
