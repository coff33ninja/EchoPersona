# voice_clone_train.py (Modified for enhanced_logger)

import os
import logging
import argparse
# Removed 'logging' import here, will get logger from enhanced_logger
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
# Make sure enhanced_logger.py is in the same directory or Python path
try:
    from enhanced_logger import setup_logger, get_logger
except ImportError:
    print("Error: enhanced_logger.py not found. Logging will not work correctly.")
    # Fallback to basic logging if module not found
    import logging
import os
import logging
import shutil
import torch
import numpy as np
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# Initialize module-level logger for use in functions before main()
logger = get_logger(__name__)
    # Define a dummy get_logger if needed elsewhere, or handle missing loggerdef get_logger(name): return logging.getLogger(name)
    # No setup_logger available in fallback

# --- Constants ---
METADATA_FILENAME = "metadata.csv"  # Expected metadata filename in dataset folder

# --- Custom Formatter ---
def custom_formatter(root_path, meta_file, **kwargs):
    """
    Custom formatter to load dataset samples where audio filenames in metadata.csv
    include the .wav extension and are located directly in the dataset directory.
    (Logging calls changed to use logger)
    """
    # logger = get_logger('custom_formatter') # Or use module logger directly
    items = []
    try:
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
                        "speaker_name": "speaker1", # Assuming single speaker dataset
                        "root_path": root_path,
                    }
                )
        logger.info(f"Loaded {len(items)} items from metadata using custom_formatter.")
    except FileNotFoundError:
        logger.error(f"Metadata file not found: {os.path.join(root_path, meta_file)}")
    except Exception as e:
        logger.exception(f"Error reading metadata file in custom_formatter: {e}")
    return items

# Fix for UnicodeEncodeError: Add a preprocessing step to clean unsupported characters.
def clean_text(text):
    """Removes unsupported characters from the text."""
    # logger = get_logger('clean_text') # Or use module logger directly
    # Increased character set based on VITS needs
    supported_chars = set(
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?'\"-:" # Basic English + Punctuation
        # Add phoneme characters if needed, VITS might handle this internally
        # Depending on your VITS configuration and language, more might be needed
    )
    original_length = len(text)
    cleaned_text = ''.join(c for c in text if c in supported_chars)
    if len(cleaned_text) != original_length:
        logger.debug(f"Cleaned text: Original='{text}', Cleaned='{cleaned_text}'")
    return cleaned_text

# Add a vocabulary update function to include missing characters
def update_vocabulary(vocabulary_path, new_characters):
    """
    Update the vocabulary file to include new characters.
    (Logging calls changed to use logger)
    """
    # logger = get_logger('update_vocabulary') # Or use module logger directly
    try:
        # Read existing vocabulary
        if os.path.exists(vocabulary_path):
            with open(vocabulary_path, "r", encoding="utf-8") as f:
                existing_vocab = set(f.read().strip())
        else:
            existing_vocab = set()

        # Add new characters
        updated_vocab = existing_vocab.union(new_characters)

        # Write updated vocabulary back to file if changed
        if updated_vocab != existing_vocab:
            with open(vocabulary_path, "w", encoding="utf-8") as f:
                f.write("".join(sorted(list(updated_vocab)))) # Sort for consistency
            logger.info(f"Vocabulary updated at {vocabulary_path} with new characters: {new_characters - existing_vocab}")
        else:
            logger.debug("Vocabulary already contains all required characters.")
    except Exception as e:
        logger.exception(f"Failed to update vocabulary: {e}")


# --- Vocabulary check moved inside main where config and samples are available ---


# Fix for PermissionError: Retry file deletion with a delay.

# Custom function to retry file deletion.
def safe_delete(file_path, retries=3, delay=1):
    """Retries deleting a file, useful for temporary locks."""
    # logger = get_logger('safe_delete') # Or use module logger directly
    for attempt in range(retries):
        try:
            if os.path.exists(file_path):
                os.unlink(file_path)
                logger.info(f"Deleted file: {file_path}")
            else:
                logger.debug(f"File not found, no deletion needed: {file_path}")
            return # Success
        except PermissionError as e:
            logger.warning(f"Attempt {attempt + 1}/{retries}: PermissionError deleting {file_path}. Retrying in {delay}s...")
            time.sleep(delay)
        except Exception as e:
            logger.exception(f"Unexpected error deleting file {file_path} on attempt {attempt + 1}")
            # Decide if to continue retrying on other errors
            if attempt == retries - 1:
                raise # Re-raise the last exception if all retries fail
            time.sleep(delay)

    logger.error(f"Could not delete file after {retries} attempts: {file_path}")
    # Optionally raise the last PermissionError if needed by caller
    # raise PermissionError(f"Could not delete file after retries: {file_path}")

# Custom function to retry directory removal.
def safe_remove_experiment_folder(path, retries=5, delay=2):
    """Retries removing a directory tree, handling temporary locks."""
    # logger = get_logger('safe_remove_experiment_folder') # Or use module logger directly
    for attempt in range(retries):
        try:
            if os.path.exists(path):
                shutil.rmtree(path, ignore_errors=False)
                logger.info(f"Removed directory tree: {path}")
            else:
                 logger.debug(f"Directory not found, no removal needed: {path}")
            return # Success
        except PermissionError as e:
            logger.warning(f"Attempt {attempt + 1}/{retries}: PermissionError removing {path}. Retrying in {delay}s...")
            time.sleep(delay)
        except Exception as e:
             logger.exception(f"Unexpected error removing directory {path} on attempt {attempt + 1}")
             if attempt == retries - 1:
                  raise # Re-raise the last exception if all retries fail
             time.sleep(delay)

    logger.error(f"Could not remove directory after {retries} attempts: {path}")
    # Optionally raise the last PermissionError if needed by caller
    # raise PermissionError(f"Could not remove directory after retries: {path}")


# Function to plot and save spectrogram results
def plot_results(y_hat, y, ap, name_prefix):
    """Plots and saves spectrogram results for predicted and ground truth audio."""
    # logger = get_logger('plot_results') # Or use module logger directly
    try:
        # Ensure tensors are numpy arrays on CPU
        if isinstance(y_hat, torch.Tensor):
             y_hat = y_hat.squeeze().detach().cpu().float().numpy()
        if isinstance(y, torch.Tensor):
             y = y.squeeze().detach().cpu().float().numpy()

        # Check if arrays are valid before processing
        if y_hat is None or y is None or y_hat.size == 0 or y.size == 0:
            logger.warning("Invalid or empty audio data received for plotting.")
            return

        # Normalize spectrograms for better visualization
        # Use try-except for spectrogram generation as it might fail on invalid data
        try:
             y_hat_spec = np.log1p(np.abs(ap.mel_spectrogram(torch.tensor(y_hat).unsqueeze(0)))) # Add batch dim back temporarily
             y_spec = np.log1p(np.abs(ap.mel_spectrogram(torch.tensor(y).unsqueeze(0))))
             y_hat_spec = y_hat_spec.squeeze().numpy() # Remove batch dim
             y_spec = y_spec.squeeze().numpy()
        except Exception as spec_e:
             logger.error(f"Error generating spectrogram for {name_prefix}: {spec_e}")
             return


        # Create a figure with two subplots
        fig, axes = plt.subplots(2, 1, figsize=(10, 8))

        # Plot predicted spectrogram
        im0 = axes[0].imshow(y_hat_spec, aspect="auto", origin="lower", interpolation="none")
        axes[0].set_title("Predicted Spectrogram")
        axes[0].set_xlabel("Time Frames")
        axes[0].set_ylabel("Mel Bins")
        fig.colorbar(im0, ax=axes[0])


        # Plot ground truth spectrogram
        im1 = axes[1].imshow(y_spec, aspect="auto", origin="lower", interpolation="none")
        axes[1].set_title("Ground Truth Spectrogram")
        axes[1].set_xlabel("Time Frames")
        axes[1].set_ylabel("Mel Bins")
        fig.colorbar(im1, ax=axes[1])

        # Save the figure
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
        default=4, # Adjust based on your system's cores/RAM
        help="Number of workers for data loading."
    )
    parser.add_argument(
        "--learning_rate", type=float, default=0.0002, help="Initial learning rate."
    )
    parser.add_argument(
        "--language",
        type=str,
        default="en",
        help="Language code for the dataset (e.g., 'en', 'es'). Affects text processing.",
    )
    # Text cleaner argument removed - VITS internal cleaner is usually preferred/required
    # parser.add_argument(
    #     "--text_cleaner",
    #     type=str,
    #     # default="english_cleaners", # Let VitsConfig handle default based on phonemes/language
    #     help="Text cleaner to use (often determined by phoneme settings)."
    # )
    parser.add_argument(
        "--use_phonemes",
        action=argparse.BooleanOptionalAction, # Allows --use-phonemes / --no-phonemes
        default=True,
        help="Use phonemes for training (default: True)."
    )
    parser.add_argument(
        "--phoneme_language",
        type=str,
        default="en-us",
        help="Phoneme language (if using phonemes, e.g., 'en-us', 'es')."
    )
    parser.add_argument(
        "--sample_rate", type=int, default=22050, help="Target sample rate for audio."
    )
    parser.add_argument(
        "--run_eval",
        action=argparse.BooleanOptionalAction, # Allows --run-eval / --no-run-eval
        default=True,
        help="Run evaluation during training (default: True)."
    )
    parser.add_argument(
        "--mixed_precision",
        action=argparse.BooleanOptionalAction, # Allows --mixed-precision / --no-mixed-precision
        default=False, # Defaulting to False as requested by user previously
        help="Use mixed precision training (default: False)."
    )
    parser.add_argument(
        "--continue_path",
        type=str,
        default=None,
        help="Path to a previous training output directory to continue from."
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level (default: INFO)."
    )
    args = parser.parse_args()
    # Basic validation
    if args.use_phonemes and not args.phoneme_language:
        parser.error("--phoneme_language is required when --use_phonemes is enabled.")


    # logger.info("Arguments parsed successfully.")
    return args

def main():
    args = parse_arguments()
    # --- Setup Logging USING enhanced_logger
    log_level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,    "CRITICAL": logging.CRITICAL
    }
    log_level = log_level_map.get(args.log_level.upper(), logging.INFO)
    log_file = os.path.join(args.output_path, "training.log") # Central log file

    # Ensure output path exists BEFORE setting up logger to write there
    try:
         os.makedirs(args.output_path, exist_ok=True)
    except OSError as e:
         # Use print because logger isn't set up yet
         print(f"CRITICAL: Could not create output directory {args.output_path}: {e}", file=sys.stderr)
         return # Cannot proceed without output directory

    # Now setup the logger using the function from enhanced_logger
    setup_logger(log_file_path=log_file, level=log_level)
    logger = get_logger(__name__)


    # --- Validate Paths ---
    logger.info(f"Using Dataset Path: {args.dataset_path}")
    logger.info(f"Using Output Path: {args.output_path}")
    if not os.path.isdir(args.dataset_path):
        logger.critical(f"Dataset path not found or is not a directory: {args.dataset_path}")
        return
    metadata_path = os.path.join(args.dataset_path, METADATA_FILENAME)
    if not os.path.isfile(metadata_path):
        logger.critical(f"Metadata file not found in dataset path: {metadata_path}")
        return


    # --- Audio Configuration (Standard VITS) ---
    audio_config = BaseAudioConfig(
        sample_rate=args.sample_rate,
        # VITS uses specific values, overriding some BaseAudioConfig defaults if needed
        # Often these are baked into VitsConfig anyway
        num_mels=80,
        fft_size=1024,
        hop_length=256,
        win_length=1024,
        # Ensure other params match expected VITS defaults if needed
        # min_level_db=-100, # VITS often uses different normalization
        # ref_level_db=20,
        # power=1.5,
        # preemphasis=0.97,
        # log_func="np.log10", # VITS might use natural log implicitly
        resample=False, # Resampling should happen during data prep if needed
    )

    # --- Dataset Configuration ---
    dataset_config = BaseDatasetConfig(
        formatter="ljspeech",  # Will be overridden by custom_formatter if provided to load_tts_samples
        meta_file_train=METADATA_FILENAME, # Relative to dataset path
        path=args.dataset_path,
        language=args.language,
        # ignored_speakers=None, # Set if you have multi-speaker metadata but want single speaker
    )

    # --- VITS Model Configuration ---
    # Make sure phoneme cache path is inside the output directory
    phoneme_cache_path = os.path.join(args.output_path, "phoneme_cache")
    os.makedirs(phoneme_cache_path, exist_ok=True) # Ensure cache dir exists

    config = VitsConfig(
        audio=audio_config,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        num_loader_workers=args.num_loader_workers,
        num_eval_loader_workers=max(1, args.num_loader_workers // 2), # Eval often needs fewer workers
        run_eval=args.run_eval,
        test_delay_epochs=-1, # Disable testing during training by default
        epochs=args.epochs,
        # text_cleaner=args.text_cleaner, # Let VITS handle cleaner based on phonemes/language
        use_phonemes=args.use_phonemes,
        phoneme_language=args.phoneme_language if args.use_phonemes else None,
        phoneme_cache_path=phoneme_cache_path,
        compute_input_seq_cache=True, # Speeds up training start after first epoch
        print_step=50, # Log training loss every 50 steps
        print_eval=True, # Print evaluation results
        mixed_precision=args.mixed_precision,
        output_path=args.output_path,
        datasets=[dataset_config], # Pass dataset config list
        lr=args.learning_rate, # Learning rate
        # VITS specific parameters (can often use defaults, but check documentation)
        # e.g., data_dep_init=True, use_sdp=True can sometimes help
        # Check Coqui TTS VitsConfig documentation for details
    )


    # --- Load Dataset Samples ---
    try:
        logger.info(f"Loading dataset samples from: {args.dataset_path}")
        # Use the custom formatter defined earlier
        train_samples, eval_samples = load_tts_samples(
            datasets=config.datasets, # Use datasets from the config object
            eval_split=True,
            eval_split_size=0.01, # Use 1% for evaluation
            formatter=custom_formatter,
        )
        if not train_samples:
            logger.critical("No training samples loaded. Check metadata file format, content, and paths.")
            return
        logger.info(f"Loaded {len(train_samples)} training samples.")
        if eval_samples:
            logger.info(f"Loaded {len(eval_samples)} evaluation samples.")
        else:
            # Training can proceed without eval samples, but evaluation steps will be skipped
            logger.warning("No evaluation samples loaded (eval_split_size might be too small or dataset too small). Evaluation will be skipped.")
            config.run_eval = False # Disable evaluation if no samples

        # --- Vocabulary Check ---
        # Collect all unique characters from the loaded dataset texts
        # Only needs to be done once after loading samples
        logger.info("Checking dataset vocabulary against model configuration...")
        all_text = "".join(sample["text"] for sample in (train_samples + (eval_samples or [])))
        dataset_chars = set(all_text)
        logger.debug(f"Characters found in dataset: {sorted(list(dataset_chars))}")

        # Check against VITS vocabulary (VITS handles its internal chars/phonemes)
        # We mainly need to ensure our text cleaner doesn't remove essential chars unexpectedly
        # and that the input text doesn't contain completely unsupported chars if NOT using phonemes.
        if not config.use_phonemes:
             # If not using phonemes, the model directly uses characters.
             # VITS has a default character set. Check if dataset chars are outside common sets.
             # This is a basic check; VITS might have its own validation.
             common_chars = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?'\"-:")
             uncommon_chars = dataset_chars - common_chars
             if uncommon_chars:
                  logger.warning(f"Dataset contains characters not in the basic set (when not using phonemes): {uncommon_chars}. Ensure your VITS model supports these or clean the text further.")
        else:
             # If using phonemes, the text is converted, so character set issues are less common,
             # unless the phonemizer itself fails on certain characters/words.
             logger.info("Phonemes are enabled. Character issues are less likely, but check phonemizer warnings if they occur.")


    except Exception as e:
        logger.exception(f"Failed to load or process dataset samples: {e}")
        return

    # --- Initialize AudioProcessor ---
    # This needs to be done AFTER the config is finalized and potentially AFTER checking dataset
    try:
        ap = AudioProcessor.init_from_config(config)
        logger.info("AudioProcessor initialized successfully.")
    except Exception as e:
        logger.exception(f"Failed to initialize AudioProcessor: {e}")
        return


    # --- Initialize Model ---
    try:
        # This will also initialize the tokenizer based on config (phonemes, language, etc.)
        model = Vits.init_from_config(config) # Removed ap argument as it is not accepted
        logger.info("VITS model initialized successfully.")
    except Exception as e:
        logger.exception(f"Failed to initialize VITS model: {e}")
        return


    # --- Initialize Trainer ---
    try:
        trainer = Trainer(
            args=TrainerArgs(
                continue_path=args.continue_path,
                #restore_path=None, # Use continue_path for continuing training runs
                #rank=0, # For distributed training
                #group_id="group_id", # For distributed training
                #use_ddp=False # Set to True for distributed training
                ),
            config=config,
            output_path=args.output_path,
            model=model,
            train_samples=train_samples,
            eval_samples=eval_samples if config.run_eval else None, # Pass None if eval disabled
            training_assets={"audio_processor": ap}, # Pass ap if needed by Trainer hooks/callbacks
        )
        logger.info("Trainer initialized successfully.")

        # --- Replace default folder removal with safer version ---
        trainer.remove_experiment_folder = safe_remove_experiment_folder

    except Exception as e:
        logger.exception(f"Failed to initialize Trainer: {e}")
        return


    # --- Start Training ---
    try:
        logger.info(">>> Starting Training <<<")
        if args.continue_path:
            logger.info(f"Attempting to continue training from: {args.continue_path}")
        trainer.fit()
        logger.info(">>> Training Finished <<<")
        print(f"\nTraining complete. Model files saved in: {args.output_path}")
    except PermissionError as e:
        logger.exception(f"PermissionError occurred during training: {e}")
        print(f"\nA PermissionError occurred: {e}")
        print("Ensure no other processes are accessing the training files (output dir, dataset) and that you have write permissions.")
        print("Check the log file for more details:", log_file)
        # exit(1) # Optional: exit with error code
    except RuntimeError as e:
         # Catch common GPU errors
         if "CUDA out of memory" in str(e):
              logger.exception(f"CUDA out of memory: {e}")
              print("\nCUDA out of memory. Try reducing --batch_size.")
              print("If using mixed precision (--mixed_precision), ensure your GPU supports it well.")
              print("Check the log file for more details:", log_file)
         elif "cuDNN error" in str(e):
              logger.exception(f"cuDNN error: {e}")
              print("\nA cuDNN error occurred. This often indicates an issue with your CUDA/cuDNN installation or GPU driver compatibility.")
              print("Ensure PyTorch, CUDA, and cuDNN versions are compatible.")
              print("Check the log file for more details:", log_file)
         else:
              # Re-raise other RuntimeErrors
              logger.exception(f"A RuntimeError occurred during training: {e}")
              print(f"\nA RuntimeError occurred during training: {e}")
              print("Check the log file for more details:", log_file)
         # exit(1) # Optional: exit with error code
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user (KeyboardInterrupt).")
        print("\nTraining interrupted. Model might not have been saved properly.")
    except Exception as e:
        # Catch any other unexpected errors
        logger.exception(f"An unexpected error occurred during training: {e}")
        print(f"\nAn unexpected error occurred during training: {e}")
        print("Check the log file for details:", log_file)
        # exit(1) # Optional: exit with error code

    # --- Save Model Explicitly (Optional - Trainer usually saves best/last) ---
    # The Trainer typically saves the best model and checkpoints automatically.
    # Explicit saving might be redundant unless you want to force-save the *very* final state.
    # try:
    #     final_model_save_path = os.path.join(args.output_path, "final_model.pth")
    #     final_config_save_path = os.path.join(args.output_path, "final_config.json") # Use final_ if different from trainer's config.json
    #
    #     # Save the model state dictionary
    #     if trainer.model: # Check if model exists
    #         torch.save(trainer.model.state_dict(), final_model_save_path)
    #         logger.info(f"Final model state saved explicitly to: {final_model_save_path}")
    #     else:
    #          logger.warning("Trainer model object not found, cannot save final model state explicitly.")
    #
    #     # Save the model configuration (usually same as trainer's config.json)
    #     # config.save_json(final_config_save_path) # Use the config object's save method
    #     # logger.info(f"Final configuration saved explicitly to: {final_config_save_path}")
    #
    # except Exception as e:
    #     logger.exception(f"Failed during explicit final model/config saving: {e}")
    #     print(f"Error during explicit final save: {e}")


if __name__ == "__main__":
    main()
