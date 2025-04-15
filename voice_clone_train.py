# voice_clone_train.py (Modified to remove skip_disk_logging and enhance file lock handling)

import os
import logging
import argparse
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

# --- Import from enhanced_logger ---
try:
    from enhanced_logger import setup_logger, get_logger
except ImportError:
    print("Error: enhanced_logger.py not found. Using basic logging.")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )

    def get_logger(name):
        return logging.getLogger(name)


# Initialize module-level logger
logger = get_logger(__name__)

# --- Constants ---
METADATA_FILENAME = "metadata.csv"


# --- Custom Formatter ---
def custom_formatter(root_path, meta_file, **kwargs):
    """
    Custom formatter to load dataset samples where audio filenames in metadata.csv
    include the .wav extension and are located directly in the dataset directory.
    """
    items = []
    try:
        with open(os.path.join(root_path, meta_file), "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("|")
                if len(parts) < 2 or parts[0] == "audio_file":
                    continue
                audio_file = os.path.join(root_path, parts[0])
                text = clean_text(parts[1])
                items.append(
                    {
                        "text": text,
                        "audio_file": audio_file,
                        "speaker_name": "speaker1",
                        "root_path": root_path,
                    }
                )
        logger.info(f"Loaded {len(items)} items from metadata using custom_formatter.")
    except FileNotFoundError:
        logger.error(f"Metadata file not found: {os.path.join(root_path, meta_file)}")
    except Exception as e:
        logger.exception(f"Error reading metadata file in custom_formatter: {e}")
    return items


# --- Text Cleaning ---
def clean_text(text):
    """Removes unsupported characters from the text."""
    supported_chars = set(
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?'\"-:"
    )
    original_length = len(text)
    cleaned_text = "".join(c for c in text if c in supported_chars)
    if len(cleaned_text) != original_length:
        logger.debug(f"Cleaned text: Original='{text}', Cleaned='{cleaned_text}'")
    return cleaned_text


# --- Vocabulary Management ---
def update_vocabulary(vocabulary_path, new_characters):
    """Update the vocabulary file to include new characters."""
    try:
        if os.path.exists(vocabulary_path):
            with open(vocabulary_path, "r", encoding="utf-8") as f:
                existing_vocab = set(f.read().strip())
        else:
            existing_vocab = set()

        updated_vocab = existing_vocab.union(new_characters)
        if updated_vocab != existing_vocab:
            with open(vocabulary_path, "w", encoding="utf-8") as f:
                f.write("".join(sorted(list(updated_vocab))))
            logger.info(
                f"Vocabulary updated at {vocabulary_path} with new characters: {new_characters - existing_vocab}"
            )
        else:
            logger.debug("Vocabulary already contains all required characters.")
    except Exception as e:
        logger.exception(f"Failed to update vocabulary: {e}")


def ensure_phoneme_vocabulary(config):
    """Ensures the VITS tokenizer vocabulary includes all gruut phonemes."""
    logger = get_logger("ensure_phoneme_vocabulary")
    try:
        gruut_phonemes = set(
            [
                "a",
                "b",
                "d",
                "e",
                "f",
                "g",
                "h",
                "i",
                "j",
                "k",
                "l",
                "m",
                "n",
                "o",
                "p",
                "r",
                "s",
                "t",
                "u",
                "v",
                "w",
                "z",
                "æ",
                "ʧ",
                "ð",
                "ɛ",
                "ɪ",
                "ŋ",
                "ɔ",
                "ɹ",
                "ʃ",
                "θ",
                "ʊ",
                "ʒ",
                "ɑ",
                "ɒ",
                "ʌ",
                "ː",
                "ɡ",
                "ɨ",
                "ʔ",
                "ɚ",
                "ɝ",
                "ɒ",
                "ɪ",
                "ʉ",
                "ʲ",
                "ʷ",
                "ᵊ",
                "ⁿ",
                "̃",
                "̩",
                "̯",
                "̮",
                "̪",
                "̺",
                "̻",
                "͡",
            ]
        )
        vocab_file = os.path.join(config.output_path, "vocabulary.txt")
        update_vocabulary(vocab_file, gruut_phonemes)
        logger.info("Phoneme vocabulary ensured for gruut en-us.")
    except Exception as e:
        logger.exception(f"Failed to ensure phoneme vocabulary: {e}")


# --- File Deletion Utilities ---
def safe_delete(file_path, retries=10, delay=3):
    """Retries deleting a file, useful for temporary locks."""
    for attempt in range(retries):
        try:
            if os.path.exists(file_path):
                os.unlink(file_path)
                logger.info(f"Deleted file: {file_path}")
            else:
                logger.debug(f"File not found, no deletion needed: {file_path}")
            return
        except PermissionError as e:
            logger.warning(
                f"Attempt {attempt + 1}/{retries}: PermissionError deleting {file_path}. Retrying in {delay}s..."
            )
            time.sleep(delay)
        except Exception as e:
            logger.exception(
                f"Unexpected error deleting file {file_path} on attempt {attempt + 1}"
            )
            if attempt == retries - 1:
                raise
            time.sleep(delay)
    logger.error(f"Could not delete file after {retries} attempts: {file_path}")


def safe_remove_experiment_folder(path, retries=10, delay=3):
    """Retries removing a directory tree, handling temporary locks."""
    for attempt in range(retries):
        try:
            if os.path.exists(path):
                shutil.rmtree(path, ignore_errors=False)
                logger.info(f"Removed directory tree: {path}")
            else:
                logger.debug(f"Directory not found, no removal needed: {path}")
            return
        except PermissionError as e:
            logger.warning(
                f"Attempt {attempt + 1}/{retries}: PermissionError removing {path}. Retrying in {delay}s..."
            )
            time.sleep(delay)
        except Exception as e:
            logger.exception(
                f"Unexpected error removing directory {path} on attempt {attempt + 1}"
            )
            if attempt == retries - 1:
                raise
            time.sleep(delay)
    logger.error(f"Could not remove directory after {retries} attempts: {path}")


# --- Plotting Results ---
def plot_results(y_hat, y, ap, name_prefix):
    """Plots and saves spectrogram results for predicted and ground truth audio."""
    try:
        if isinstance(y_hat, torch.Tensor):
            y_hat = y_hat.squeeze().detach().cpu().float().numpy()
        if isinstance(y, torch.Tensor):
            y = y.squeeze().detach().cpu().float().numpy()

        if y_hat is None or y is None or y_hat.size == 0 or y.size == 0:
            logger.warning("Invalid or empty audio data received for plotting.")
            return

        try:
            y_hat_spec = (
                np.log1p(np.abs(ap.mel_spectrogram(torch.tensor(y_hat).unsqueeze(0))))
                .squeeze()
                .numpy()
            )
            y_spec = (
                np.log1p(np.abs(ap.mel_spectrogram(torch.tensor(y).unsqueeze(0))))
                .squeeze()
                .numpy()
            )
        except Exception as spec_e:
            logger.error(f"Error generating spectrogram for {name_prefix}: {spec_e}")
            return

        fig, axes = plt.subplots(2, 1, figsize=(10, 8))
        im0 = axes[0].imshow(
            y_hat_spec, aspect="auto", origin="lower", interpolation="none"
        )
        axes[0].set_title("Predicted Spectrogram")
        axes[0].set_xlabel("Time Frames")
        axes[0].set_ylabel("Mel Bins")
        fig.colorbar(im0, ax=axes[0])

        im1 = axes[1].imshow(
            y_spec, aspect="auto", origin="lower", interpolation="none"
        )
        axes[1].set_title("Ground Truth Spectrogram")
        axes[1].set_xlabel("Time Frames")
        axes[1].set_ylabel("Mel Bins")
        fig.colorbar(im1, ax=axes[1])

        plot_path = f"{name_prefix}_spectrogram.png"
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close(fig)
        logger.info(f"Spectrogram plots saved to: {plot_path}")
    except Exception as e:
        logger.exception(f"Error while plotting results: {e}")


# --- Argument Parser ---
def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Train a VITS TTS model for a specific character."
    )
    parser.add_argument(
        "--dataset_path", type=str, required=True, help="Path to dataset directory."
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save trained model and logs.",
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
        help="Number of workers for data loading.",
    )
    parser.add_argument(
        "--learning_rate", type=float, default=0.0002, help="Initial learning rate."
    )
    parser.add_argument(
        "--language", type=str, default="en", help="Language code for dataset."
    )
    parser.add_argument(
        "--use_phonemes",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use phonemes.",
    )
    parser.add_argument(
        "--phoneme_language", type=str, default="en-us", help="Phoneme language."
    )
    parser.add_argument(
        "--sample_rate", type=int, default=22050, help="Target sample rate."
    )
    parser.add_argument(
        "--run_eval",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run evaluation.",
    )
    parser.add_argument(
        "--mixed_precision",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use mixed precision.",
    )
    parser.add_argument(
        "--continue_path",
        type=str,
        default=None,
        help="Path to continue training from.",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level.",
    )
    args = parser.parse_args()
    if args.use_phonemes and not args.phoneme_language:
        parser.error("--phoneme_language is required when --use_phonemes is enabled.")
    return args


def main():
    args = parse_arguments()

    # Setup Logging
    log_level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }
    log_level = log_level_map.get(args.log_level.upper(), logging.INFO)
    output_path = (
        args.continue_path if args.continue_path else args.output_path
    )  # Use continue_path if provided
    log_file = os.path.join(output_path, "training.log")

    os.makedirs(output_path, exist_ok=True)
    setup_logger(log_file_path=log_file, level=log_level)
    logger = get_logger(__name__)

    # Validate Paths
    logger.info(f"Using Dataset Path: {args.dataset_path}")
    logger.info(f"Using Output Path: {output_path}")
    if not os.path.isdir(args.dataset_path):
        logger.critical(f"Dataset path not found: {args.dataset_path}")
        return
    metadata_path = os.path.join(args.dataset_path, METADATA_FILENAME)
    if not os.path.isfile(metadata_path):
        logger.critical(f"Metadata file not found: {metadata_path}")
        return

    # Clean Up Stale Logs
    run_dirs = [d for d in os.listdir(output_path) if d.startswith("run-")]
    for run_dir in run_dirs:
        run_path = os.path.join(output_path, run_dir)
        logger.info(f"Cleaning up stale run directory: {run_path}")
        safe_remove_experiment_folder(run_path)

    # Audio Configuration
    audio_config = BaseAudioConfig(
        sample_rate=args.sample_rate,
        num_mels=80,
        fft_size=1024,
        hop_length=256,
        win_length=1024,
        resample=False,
    )

    # Dataset Configuration
    dataset_config = BaseDatasetConfig(
        formatter="ljspeech",
        meta_file_train=METADATA_FILENAME,
        path=args.dataset_path,
        language=args.language,
    )

    # VITS Model Configuration
    phoneme_cache_path = os.path.join(output_path, "phoneme_cache")
    os.makedirs(phoneme_cache_path, exist_ok=True)

    config = VitsConfig(
        audio=audio_config,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        num_loader_workers=args.num_loader_workers,
        num_eval_loader_workers=max(1, args.num_loader_workers // 2),
        run_eval=args.run_eval,
        test_delay_epochs=-1,
        epochs=args.epochs,
        use_phonemes=args.use_phonemes,
        phoneme_language=args.phoneme_language if args.use_phonemes else None,
        phoneme_cache_path=phoneme_cache_path,
        compute_input_seq_cache=True,
        print_step=50,
        print_eval=True,
        mixed_precision=args.mixed_precision,
        output_path=output_path,
        datasets=[dataset_config],
        lr=args.learning_rate,
    )

    # Ensure Phoneme Vocabulary
    ensure_phoneme_vocabulary(config)

    # Load Dataset Samples
    try:
        logger.info(f"Loading dataset samples from: {args.dataset_path}")
        train_samples, eval_samples = load_tts_samples(
            datasets=config.datasets,
            eval_split=True,
            eval_split_size=0.01,
            formatter=custom_formatter,
        )
        if not train_samples:
            logger.critical("No training samples loaded.")
            return
        logger.info(f"Loaded {len(train_samples)} training samples.")
        if eval_samples:
            logger.info(f"Loaded {len(eval_samples)} evaluation samples.")
        else:
            logger.warning("No evaluation samples loaded.")
            config.run_eval = False
    except FileNotFoundError as fnf:
        logger.error(f"Phoneme cache file missing or inaccessible: {fnf}")
        logger.info("Clearing phoneme cache and retrying...")
        shutil.rmtree(phoneme_cache_path, ignore_errors=True)
        os.makedirs(phoneme_cache_path, exist_ok=True)
        train_samples, eval_samples = load_tts_samples(
            datasets=config.datasets,
            eval_split=True,
            eval_split_size=0.01,
            formatter=custom_formatter,
        )
    except Exception as e:
        logger.exception(f"Failed to load dataset samples: {e}")
        return

    # Vocabulary Check
    logger.info("Checking dataset vocabulary against model configuration...")
    all_text = "".join(
        sample["text"] for sample in (train_samples + (eval_samples or []))
    )
    dataset_chars = set(all_text)
    logger.debug(f"Characters found in dataset: {sorted(list(dataset_chars))}")
    if not config.use_phonemes:
        common_chars = set(
            "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?'\"-:"
        )
        uncommon_chars = dataset_chars - common_chars
        if uncommon_chars:
            logger.warning(
                f"Dataset contains characters not in basic set: {uncommon_chars}."
            )

    # Initialize AudioProcessor
    try:
        ap = AudioProcessor.init_from_config(config)
        logger.info("AudioProcessor initialized successfully.")
    except Exception as e:
        logger.exception(f"Failed to initialize AudioProcessor: {e}")
        return

    # Initialize Tokenizer
    try:
        from TTS.tts.utils.text.tokenizer import TTSTokenizer

        tokenizer, config = TTSTokenizer.init_from_config(config)
        logger.info("Tokenizer initialized successfully.")
    except Exception as e:
        logger.exception(f"Failed to initialize Tokenizer: {e}")
        return

    # Initialize Model
    try:
        model = Vits(config, ap, tokenizer)
        logger.info("VITS model initialized successfully.")
    except Exception as e:
        logger.exception(f"Failed to initialize VITS model: {e}")
        return

    # Initialize Trainer
    try:
        trainer = Trainer(
            args=TrainerArgs(),  # Use default TrainerArgs, handle continue_path via output_path
            config=config,
            output_path=output_path,
            model=model,
            train_samples=train_samples,
            eval_samples=eval_samples if config.run_eval else None,
            training_assets={"audio_processor": ap},
        )
        trainer.remove_experiment_folder = safe_remove_experiment_folder
        logger.info("Trainer initialized successfully.")
    except TypeError as e:
        logger.exception(f"Failed to initialize Trainer due to invalid arguments: {e}")
        print(f"\nError: Invalid Trainer arguments: {e}")
        print("Check your trainer package version or configuration.")
        print("Check log file:", log_file)
        return
    except Exception as e:
        logger.exception(f"Failed to initialize Trainer: {e}")
        print(f"\nError initializing Trainer: {e}")
        print("Check log file:", log_file)
        return

    # Save Final Model and Config
    final_model_save_path = os.path.join(output_path, "final_model.pth")
    final_config_save_path = os.path.join(output_path, "final_config.json")

    # Start Training
    try:
        logger.info(">>> Starting Training <<<")
        logger.warning(
            "On Windows, file locking issues may occur. Close other programs accessing training files."
        )
        if args.continue_path:
            logger.info(f"Continuing training from: {args.continue_path}")
        trainer.fit()
        logger.info(">>> Training Finished <<<")
        print(f"\nTraining complete. Model files saved in: {output_path}")

        # Save final model and config after training
        torch.save(model.state_dict(), final_model_save_path)
        with open(final_config_save_path, "w") as f:
            f.write(config.to_json())
        logger.info(f"Saved final model to: {final_model_save_path}")
        logger.info(f"Saved final config to: {final_config_save_path}")
    except PermissionError as e:
        logger.exception(f"PermissionError during training: {e}")
        print(f"\nPermissionError: {e}")
        print(
            "Ensure no other processes access training files and you have write permissions."
        )
        print("Check log file:", log_file)
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            print("\nCUDA out of memory. Try reducing --batch_size.")
        elif "cuDNN error" in str(e):
            print("\ncuDNN error. Check CUDA/cuDNN compatibility.")
        else:
            print(f"\nRuntimeError during training: {e}")
        print("Check log file:", log_file)
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user.")
        print("\nTraining interrupted. Model may not have saved properly.")
    except Exception as e:
        logger.exception(f"Unexpected error during training: {e}")
        print(f"\nUnexpected error: {e}")
        print("Check log file:", log_file)

if __name__ == "__main__":
    main()
