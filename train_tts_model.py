import os
import logging
import argparse
import tkinter as tk
from tkinter import filedialog, ttk
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.models.vits import Vits
from TTS.tts.datasets import load_tts_samples
from trainer import Trainer, TrainerArgs

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("train_tts_model.log"),
        logging.StreamHandler(),
    ],
)


def load_config(config_path):
    """Load configuration using VitsConfig."""
    config = VitsConfig()
    config.load_json(config_path)
    return config


def adjust_metadata_paths(dataset_path, meta_file_train, meta_file_val):
    """Adjust metadata file paths."""
    meta_file_train_basename = os.path.basename(meta_file_train)
    meta_file_val_basename = os.path.basename(meta_file_val)
    meta_file_train_path = os.path.join(dataset_path, meta_file_train_basename)
    meta_file_val_path = os.path.join(dataset_path, meta_file_val_basename)

    if not os.path.exists(meta_file_train_path) or not os.path.exists(
        meta_file_val_path
    ):
        parent_dir = os.path.dirname(dataset_path)
        meta_file_train_path = os.path.join(parent_dir, meta_file_train_basename)
        meta_file_val_path = os.path.join(parent_dir, meta_file_val_basename)

    if not os.path.exists(meta_file_train_path) or not os.path.exists(
        meta_file_val_path
    ):
        raise FileNotFoundError(
            f"Metadata files not found at {meta_file_train_path} or {meta_file_val_path}"
        )

    logging.info(f"Resolved training metadata file: {meta_file_train_path}")
    logging.info(f"Resolved validation metadata file: {meta_file_val_path}")
    return meta_file_train_path, meta_file_val_path


def train_model(config_path, dataset_path, output_dir, character):
    """Train a VITS model for a character."""
    try:
        # Load configuration
        config = load_config(config_path)

        # Set multi-speaker parameters
        config.use_speaker_embedding = True
        config.num_speakers = 1  # Single character for now

        # Adjust metadata paths
        dataset_config = config.datasets[0]
        meta_file_train_path, meta_file_val_path = adjust_metadata_paths(
            dataset_path, dataset_config.meta_file_train, dataset_config.meta_file_val
        )

        # Load dataset
        dataset_obj = {
            "path": os.path.dirname(dataset_path),
            "meta_file_train": os.path.basename(meta_file_train_path),
            "meta_file_val": os.path.basename(meta_file_val_path),
            "formatter": "ljspeech",
            "dataset_name": f"{character}_dataset",
            "ignored_speakers": [],
            "language": "en",
            "audio_path": "wavs",
            "meta_file_attn_mask": None,
        }
        train_samples, eval_samples = load_tts_samples(
            datasets=[dataset_obj],
            eval_split=True,
            eval_split_max_size="50%",
            eval_split_size=1000,
        )

        # Initialize model
        model = Vits(config)

        # Set up trainer
        trainer_args = TrainerArgs()
        trainer = Trainer(
            trainer_args,
            config,
            output_path=output_dir,
            model=model,
            train_samples=train_samples,
            eval_samples=eval_samples,
        )

        # Start training
        trainer.fit()

        logging.info(f"Training completed for {character}")
    except Exception as e:
        logging.error(f"Training failed for {character}: {e}")
        raise


def run_gui():
    """Run the GUI interface."""
    window = tk.Tk()
    window.title("TTS Model Trainer")

    tk.Label(window, text="Base Directory:").grid(row=0, column=0, padx=5, pady=5)
    base_dir_var = tk.StringVar()
    tk.Entry(window, textvariable=base_dir_var, width=50).grid(
        row=0, column=1, padx=5, pady=5
    )
    tk.Button(
        window,
        text="Browse",
        command=lambda: base_dir_var.set(filedialog.askdirectory()),
    ).grid(row=0, column=2, padx=5, pady=5)

    tk.Label(window, text="Character:").grid(row=1, column=0, padx=5, pady=5)
    character_var = tk.StringVar()
    character_dropdown = ttk.Combobox(window, textvariable=character_var)
    character_dropdown.grid(row=1, column=1, padx=5, pady=5)

    tk.Label(window, text="Output Directory:").grid(row=2, column=0, padx=5, pady=5)
    output_dir_var = tk.StringVar()
    tk.Entry(window, textvariable=output_dir_var, width=50).grid(
        row=2, column=1, padx=5, pady=5
    )
    tk.Button(
        window,
        text="Browse",
        command=lambda: output_dir_var.set(filedialog.askdirectory()),
    ).grid(row=2, column=2, padx=5, pady=5)

    tk.Button(
        window,
        text="Train",
        command=lambda: train_model(
            os.path.join(
                base_dir_var.get(),
                character_var.get(),
                f"{character_var.get()}_config.json",
            ),
            os.path.join(base_dir_var.get(), character_var.get(), "wavs"),
            output_dir_var.get(),
            character_var.get(),
        ),
    ).grid(row=3, column=0, columnspan=3, pady=10)

    def update_characters():
        base_dir = base_dir_var.get()
        if base_dir and os.path.exists(base_dir):
            characters = [
                d
                for d in os.listdir(base_dir)
                if os.path.isdir(os.path.join(base_dir, d))
            ]
            character_dropdown["values"] = characters
            if characters:
                character_var.set(characters[0])

    base_dir_var.trace("w", lambda *args: update_characters())
    window.mainloop()


def main():
    parser = argparse.ArgumentParser(description="Train TTS models for characters.")
    parser.add_argument(
        "--base-dir", type=str, help="Base directory containing character datasets"
    )
    parser.add_argument("--character", type=str, help="Character to train")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="tts_output",
        help="Output directory for trained models",
    )
    parser.add_argument("--gui", action="store_true", help="Run in GUI mode")

    args = parser.parse_args()

    if args.gui:
        run_gui()
    else:
        if not all([args.base_dir, args.character]):
            parser.error("base-dir and character are required in CLI mode")

        config_path = os.path.join(
            args.base_dir, args.character, f"{args.character}_config.json"
        )
        dataset_path = os.path.join(args.base_dir, args.character, "wavs")
        train_model(config_path, dataset_path, args.output_dir, args.character)


if __name__ == "__main__":
    main()
