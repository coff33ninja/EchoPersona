import os
import argparse
import logging
from trainer import Trainer, TrainerArgs
from TTS.config import BaseAudioConfig, BaseDatasetConfig
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.vits import Vits
from TTS.utils.audio import AudioProcessor

# --- Constants ---
METADATA_FILENAME = "metadata.csv"  # Expected metadata filename in dataset folder

# Configure logging
logging.basicConfig(filename="training.log", level=logging.INFO)

# --- Custom Formatter ---
def custom_formatter(root_path, meta_file, **kwargs):
    """
    Custom formatter to load dataset samples where audio filenames in metadata.csv
    include the .wav extension and are located directly in the dataset directory.

    Args:
        root_path (str): Path to the dataset directory.
        meta_file (str): Name of the metadata file (e.g., 'metadata.csv').
        **kwargs: Additional arguments (ignored).

    Returns:
        list: List of dictionaries containing 'text', 'audio_file', and 'speaker_name'.
    """
    items = []
    with open(os.path.join(root_path, meta_file), "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("|")
            if len(parts) < 2 or parts[0] == "audio_file":  # Skip header or malformed lines
                continue
            audio_file = os.path.join(root_path, parts[0])  # Full filename from metadata
            text = clean_text(parts[1])  # Clean text to remove unsupported characters
            items.append(
                {
                    "text": text,
                    "audio_file": audio_file,
                    "speaker_name": "speaker1",
                    "root_path": root_path,
                }
            )
    return items

# Fix for UnicodeEncodeError: Add a preprocessing step to clean unsupported characters.
def clean_text(text):
    """Removes unsupported characters from the text."""
    supported_chars = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?-'")
    return ''.join(c for c in text if c in supported_chars)

# Add a vocabulary update function to include missing characters
def update_vocabulary(vocabulary_path, new_characters):
    """
    Update the vocabulary file to include new characters.

    Args:
        vocabulary_path (str): Path to the vocabulary file.
        new_characters (set): Set of characters to add to the vocabulary.
    """
    try:
        # Read existing vocabulary
        if os.path.exists(vocabulary_path):
            with open(vocabulary_path, "r", encoding="utf-8") as f:
                existing_vocab = set(f.read().strip())
        else:
            existing_vocab = set()

        # Add new characters
        updated_vocab = existing_vocab.union(new_characters)

        # Write updated vocabulary back to file
        with open(vocabulary_path, "w", encoding="utf-8") as f:
            f.write("".join(sorted(updated_vocab)))

        logging.info("Vocabulary updated successfully.")
    except Exception as e:
        logging.error(f"Failed to update vocabulary: {e}", exc_info=True)

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
        help="Number of workers for data loading."
    )
    parser.add_argument(
        "--learning_rate", type=float, default=0.0002, help="Initial learning rate."
    )
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
        help="Text cleaner to use."
    )
    parser.add_argument(
        "--use_phonemes",
        action="store_true",
        default=True,
        help="Use phonemes for training."
    )
    parser.add_argument(
        "--no_phonemes",
        action="store_false",
        dest="use_phonemes",
        help="Do not use phonemes for training."
    )
    parser.add_argument(
        "--phoneme_language",
        type=str,
        default="en-us",
        help="Phoneme language (if using phonemes)."
    )
    parser.add_argument(
        "--sample_rate", type=int, default=22050, help="Target sample rate."
    )
    parser.add_argument(
        "--run_eval",
        action="store_true",
        default=True,
        help="Run evaluation during training."
    )
    parser.add_argument(
        "--no_eval",
        action="store_false",
        dest="run_eval",
        help="Do not run evaluation during training."
    )
    parser.add_argument(
        "--mixed_precision",
        action="store_true",
        default=True,
        help="Use mixed precision training."
    )
    parser.add_argument(
        "--no_mixed_precision",
        action="store_false",
        dest="mixed_precision",
        help="Do not use mixed precision training."
    )
    parser.add_argument(
        "--continue_path",
        type=str,
        default=None,
        help="Path to a previous training output directory to continue from."
    )
    return parser.parse_args()

# --- Main Training Function ---
def main():
    args = parse_arguments()

    # --- Validate Paths ---
    if not os.path.isdir(args.dataset_path):
        logging.error(f"Dataset path not found or is not a directory: {args.dataset_path}")
        return
    metadata_path = os.path.join(args.dataset_path, METADATA_FILENAME)
    if not os.path.isfile(metadata_path):
        logging.error(f"Metadata file not found in dataset path: {metadata_path}")
        return

    # Ensure output path exists
    os.makedirs(args.output_path, exist_ok=True)
    logging.info(f"Output path: {args.output_path}")

    # --- Configure Logging ---
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(args.output_path, "training.log")),
            logging.StreamHandler()
        ]
    )

    # --- Audio Configuration ---
    audio_config = BaseAudioConfig(
        sample_rate=args.sample_rate,
        resample=False,
        num_mels=80,
        log_func="np.log10",
        min_level_db=-100,
        ref_level_db=20,
        fft_size=1024,
        power=1.5,
        preemphasis=0.97,
        hop_length=256,
        win_length=1024,
    )

    # --- Dataset Configuration ---
    dataset_config = BaseDatasetConfig(
        formatter="ljspeech",  # Overridden by custom_formatter below
        meta_file_train=METADATA_FILENAME,
        path=args.dataset_path,
        language=args.language,
    )

    # --- VITS Model Configuration ---
    config = VitsConfig(
        audio=audio_config,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        num_loader_workers=args.num_loader_workers,
        num_eval_loader_workers=args.num_loader_workers,
        run_eval=args.run_eval,
        test_delay_epochs=-1,
        epochs=args.epochs,
        text_cleaner=args.text_cleaner,
        use_phonemes=args.use_phonemes,
        phoneme_language=args.phoneme_language,
        phoneme_cache_path=os.path.join(args.output_path, "phoneme_cache"),
        compute_input_seq_cache=True,
        print_step=50,
        print_eval=True,
        mixed_precision=args.mixed_precision,
        output_path=args.output_path,
        datasets=[dataset_config],
        lr=args.learning_rate,
    )

    # --- Initialize AudioProcessor ---
    try:
        ap = AudioProcessor.init_from_config(config)
        logging.info("AudioProcessor initialized successfully.")
    except Exception as e:
        logging.error(f"Failed to initialize AudioProcessor: {e}", exc_info=True)
        return

    # --- Load Dataset Samples ---
    try:
        logging.info(f"Loading dataset samples from: {args.dataset_path}")
        train_samples, eval_samples = load_tts_samples(
            datasets=[dataset_config],
            eval_split=True,
            eval_split_size=0.01,
            formatter=custom_formatter,  # Use custom formatter
        )
        if not train_samples:
            logging.error("No training samples loaded. Check metadata file format and content.")
            return
        logging.info(f"Loaded {len(train_samples)} training samples.")
        if eval_samples:
            logging.info(f"Loaded {len(eval_samples)} evaluation samples.")
        else:
            logging.warning("No evaluation samples loaded. Evaluation might be skipped.")

        # Filter out failed transcriptions
        train_samples = [sample for sample in train_samples if sample.get('text') != '<transcription_failed>']
        eval_samples = [sample for sample in eval_samples if sample.get('text') != '<transcription_failed>']

        # Validate all samples
        all_samples = train_samples + eval_samples
        for sample in all_samples:
            if not sample["audio_file"].lower().endswith(".wav"):
                logging.error(f"Non-WAV file found in dataset: {sample['audio_file']}")
                return
            if not os.path.isfile(sample["audio_file"]):
                logging.error(f"Missing audio file: {sample['audio_file']}")
                return

    except Exception as e:
        logging.error(f"Failed to load dataset samples: {e}", exc_info=True)
        return

    # --- Initialize Model ---
    try:
        model = Vits.init_from_config(config)
        logging.info("VITS model initialized successfully.")
    except Exception as e:
        logging.error(f"Failed to initialize VITS model: {e}", exc_info=True)
        return

    # Ensure the phoneme cache directory is created before training.
    phoneme_cache_dir = os.path.join(args.output_path, 'phoneme_cache')
    os.makedirs(phoneme_cache_dir, exist_ok=True)

    # --- Initialize Trainer ---
    try:
        trainer = Trainer(
            args=TrainerArgs(continue_path=args.continue_path),
            config=config,
            output_path=args.output_path,
            model=model,
            train_samples=train_samples,
            eval_samples=eval_samples,
            training_assets={"audio_processor": ap},
        )
        logging.info("Trainer initialized successfully.")
    except Exception as e:
        logging.error(f"Failed to initialize Trainer: {e}", exc_info=True)
        return

    # Update the vocabulary with a comprehensive set of characters
    # Define the path to the vocabulary file
    vocabulary_path = os.path.join(args.output_path, "vocabulary.log")

    # Define a comprehensive set of characters for the vocabulary
    comprehensive_characters = set(
        "abcdefghijklmnopqrstuvwxyz"  # Lowercase letters
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"  # Uppercase letters
        "0123456789"                  # Numbers
        ".,?!'\":;-()"               # Punctuation
        "ðʃʒŋæɔɪʊɛɑʌɚɝɹɾɫɡ͡ "         # Phonetic symbols and space
    )

    # Update the vocabulary
    update_vocabulary(vocabulary_path, comprehensive_characters)

    # Log the vocabulary update
    logging.info(f"Updated vocabulary with comprehensive characters: {comprehensive_characters}")

    # --- Start Training ---
    try:
        logging.info(">>> Starting Training <<<")
        if args.continue_path:
            logging.info(f"Continuing training from: {args.continue_path}")
        trainer.fit()
        logging.info(">>> Training Finished <<<")
        print(f"\nTraining complete. Model files saved in: {args.output_path}")
    except Exception as e:
        logging.error("An error occurred during training.", exc_info=True)
        print(f"An error occurred during training: {e}")

if __name__ == "__main__":
    main()
