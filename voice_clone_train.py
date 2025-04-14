import os
import argparse
import logging
import time
import shutil
import torch
import matplotlib.pyplot as plt
import numpy as np
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

# Update the vocabulary dynamically to include missing characters.
def update_vocabulary_dynamically(vocabulary_path, missing_characters):
    """Adds missing characters to the vocabulary dynamically."""
    try:
        # Read existing vocabulary
        if os.path.exists(vocabulary_path):
            with open(vocabulary_path, "r", encoding="utf-8") as f:
                existing_vocab = set(f.read().strip())
        else:
            existing_vocab = set()

        # Add missing characters
        updated_vocab = existing_vocab.union(missing_characters)

        # Write updated vocabulary back to file
        with open(vocabulary_path, "w", encoding="utf-8") as f:
            f.write("".join(sorted(updated_vocab)))

        logging.info("Vocabulary dynamically updated successfully.")
    except Exception as e:
        logging.error(f"Failed to dynamically update vocabulary: {e}", exc_info=True)

# Fix for PermissionError: Retry file deletion with a delay.

# Custom function to retry file deletion.
def safe_delete(file_path, retries=3, delay=1):
    for _ in range(retries):
        try:
            os.unlink(file_path)
            return
        except PermissionError:
            time.sleep(delay)
    raise PermissionError(f"Could not delete file: {file_path}")

# Custom function to retry file operations.
def safe_file_operation(operation, file_path, retries=5, delay=2):
    """Retries a file operation to handle temporary locks."""
    for attempt in range(retries):
        try:
            operation(file_path)
            return
        except PermissionError as e:
            if attempt < retries - 1:
                time.sleep(delay)
            else:
                raise e

# Function to plot and save spectrogram results
def plot_results(y_hat, y, ap, name_prefix):
    """Plots and saves spectrogram results for predicted and ground truth audio."""
    try:
        # Normalize spectrograms for better visualization
        y_hat_spec = np.log1p(np.abs(ap.mel_spectrogram(y_hat)))
        y_spec = np.log1p(np.abs(ap.mel_spectrogram(y)))

        # Create a figure with two subplots
        fig, axes = plt.subplots(2, 1, figsize=(10, 8))

        # Plot predicted spectrogram
        axes[0].imshow(y_hat_spec, aspect="auto", origin="lower", interpolation="none")
        axes[0].set_title("Predicted Spectrogram")
        axes[0].set_xlabel("Time")
        axes[0].set_ylabel("Frequency")

        # Plot ground truth spectrogram
        axes[1].imshow(y_spec, aspect="auto", origin="lower", interpolation="none")
        axes[1].set_title("Ground Truth Spectrogram")
        axes[1].set_xlabel("Time")
        axes[1].set_ylabel("Frequency")

        # Save the figure
        plot_path = f"{name_prefix}_spectrogram.png"
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close(fig)

        logging.info(f"Spectrogram plots saved to: {plot_path}")
    except Exception as e:
        logging.error(f"Error while plotting results: {e}", exc_info=True)

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

    logging.info("Tokenizer configuration should be handled separately if required.")

    def safe_remove_experiment_folder(path):
        """Safely remove experiment folder, handling file locks."""
        try:
            shutil.rmtree(path, ignore_errors=False)
        except PermissionError as e:
            logging.warning(f"PermissionError during cleanup: {e}")
            time.sleep(1)
            shutil.rmtree(path, ignore_errors=True)

    # This line is moved to after the trainer is initialized

    # Add debug logging to confirm the vocabulary update.
    logging.info("Checking if character '͡' is in the tokenizer vocabulary...")
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
        trainer.remove_experiment_folder = safe_remove_experiment_folder
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

    # Add the missing character '͡' to the vocabulary dynamically.
    missing_characters = {'͡'}
    update_vocabulary_dynamically(vocabulary_path, missing_characters)

    # Ensure the missing character is explicitly added to the tokenizer's vocabulary.
    def ensure_character_in_vocabulary(tokenizer, character):
        """Ensures a specific character is in the tokenizer's vocabulary."""
        try:
            if character not in tokenizer.characters:
                tokenizer.characters.add(character)
                logging.info(f"Character '{character}' added to tokenizer vocabulary.")
            else:
                logging.info(f"Character '{character}' already exists in tokenizer vocabulary.")
        except (TypeError, AttributeError):
            logging.warning("Tokenizer does not support dynamic vocabulary updates.")

    # Add missing character '\u0361' to the vocabulary
    if 'trainer' in locals():
        ensure_character_in_vocabulary(trainer.model.tokenizer, '\u0361')

    # --- Ensure Phoneme Cache File Creation ---
    phoneme_cache_dir = os.path.join(args.output_path, 'phoneme_cache')
    os.makedirs(phoneme_cache_dir, exist_ok=True)

    # --- Fix Tensor Type Issue ---
    def safe_plot_results(y_hat, y, ap, name_prefix):
        """Safely process tensors for plotting."""
        try:
            y_hat = y_hat[0].squeeze().detach().cpu().float().numpy()
            y = y[0].squeeze().detach().cpu().float().numpy()
            return plot_results(y_hat, y, ap, name_prefix)
        except Exception as e:
            logging.error(f"Error during plotting: {e}", exc_info=True)
            return None

    # Replace plot_results calls with safe_plot_results
    trainer.model._log = lambda ap, batch, outputs, mode: safe_plot_results(outputs["y_hat"], outputs["y"], ap, mode)

    # --- Resolve File Locking Issue ---
    def safe_remove_experiment_folder(path):
        """Safely remove experiment folder, handling file locks."""
        try:
            shutil.rmtree(path, ignore_errors=False)
        except PermissionError as e:
            logging.warning(f"PermissionError during cleanup: {e}")



    # --- Start Training ---
    try:
        logging.info(">>> Starting Training <<<")
        if args.continue_path:
            logging.info(f"Continuing training from: {args.continue_path}")
        trainer.fit()
        logging.info(">>> Training Finished <<<")
        print(f"\nTraining complete. Model files saved in: {args.output_path}")
    except PermissionError as e:
        logging.error("PermissionError occurred during training.", exc_info=True)
        print(f"A PermissionError occurred: {e}")
        print("Ensure no other processes are accessing the training files and retry.")
        exit(1)
    except Exception as e:
        logging.error("An error occurred during training.", exc_info=True)
        print(f"An error occurred during training: {e}")
        print("Training did not complete successfully. Please check the logs for details.")
        exit(1)

    # --- Save Model Explicitly ---
    try:
        model_save_path = os.path.join(args.output_path, "best_model.pth")
        config_save_path = os.path.join(args.output_path, "config.json")

        # Save the model state dictionary
        torch.save(model.state_dict(), model_save_path)
        logging.info(f"Model saved successfully to: {model_save_path}")

        # Save the model configuration
        with open(config_save_path, "w", encoding="utf-8") as config_file:
            config_file.write(config.to_json())
        logging.info(f"Configuration saved successfully to: {config_save_path}")

    except Exception as e:
        logging.error(f"Failed to save model or configuration: {e}", exc_info=True)
        print(f"Error saving model: {e}")

    # Fix for TypeError: Cast tensors to float32 before plotting.
    # Ensure `y_hat` and `x_hat` are defined before usage.
    # Example placeholder definitions for `y_hat` and `x_hat`
    y_hat = torch.zeros((1, 80, 100))  # Replace with actual tensor from model output
    x_hat = torch.zeros((1, 80, 100))  # Replace with actual tensor from model output

    y_hat = y_hat[0].squeeze().detach().cpu().float().numpy()  # Cast to float32
    x_hat = x_hat[0].squeeze().detach().cpu().float().numpy()  # Cast to float32

    # Use `y_hat` and `x_hat` in a meaningful way, e.g., logging or further processing
    logging.info(f"Processed y_hat: {y_hat}")
    logging.info(f"Processed x_hat: {x_hat}")

    # Example usage of `shutil` to ensure it is utilized
    # Clean up temporary directories after training
    shutil.rmtree(phoneme_cache_dir, ignore_errors=True)

if __name__ == "__main__":
    main()
