import os
import json
import logging
import argparse
import threading
import queue
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import torch
import traceback
from pathlib import Path
from typing import Optional, Dict, Any
from TTS.utils.radam import RAdam
import torch.serialization

# --- Logging Configuration ---
logger = logging.getLogger()
logger.setLevel(logging.INFO)

log_format = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(log_format)
logger.addHandler(console_handler)

file_handler = logging.FileHandler("train_tts_model.log", encoding="utf-8")
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(log_format)
logger.addHandler(file_handler)


# --- Conditional Imports ---
def import_trainer_and_args():
    """Attempt to import Trainer, TrainerArgs, and related functions."""
    try:
        from TTS.bin.train_tts import (
            Trainer,
            TrainerArgs,
            load_tts_samples,
            setup_model,
        )

        logging.info(
            "Successfully imported Trainer, TrainerArgs, load_tts_samples, setup_model from TTS.bin.train_tts"
        )
        return Trainer, TrainerArgs, load_tts_samples, setup_model, "bin.train_tts"
    except ImportError as e:
        logging.warning(f"Failed to import from TTS.bin.train_tts: {e}")
        return None, None, None, None, "api"


try:
    Trainer, TrainerArgs, load_tts_samples, setup_model, import_source = (
        import_trainer_and_args()
    )
    from TTS.utils.manage import ModelManager
    from TTS.utils.synthesizer import Synthesizer
    from TTS.api import TTS
except ImportError as e:
    logging.error(f"Critical import failure: {e}")
    raise

# --- Constants ---
DEFAULT_BASE_DIR = "voice_datasets"
DEFAULT_OUTPUT_DIR = "tts_output"
SUPPORTED_MODELS = [
    "tts_models/en/ljspeech/tacotron2-DDC",
    "tts_models/en/ljspeech/vits",
    "tts_models/en/ek1/tacotron2",
]
CACHE_DIR = os.path.expanduser("~/.cache/tts/")


# --- Helper Functions ---
def find_character_configs(base_dir: str) -> Dict[str, str]:
    """Find all character folders with a JSON config file."""
    configs = {}
    base_path = Path(base_dir)
    if not base_path.exists():
        logging.error(f"Base directory does not exist: {base_dir}")
        return configs
    for character_dir in base_path.iterdir():
        if character_dir.is_dir():
            for file in character_dir.glob("*.json"):
                if file.name.endswith("_config.json"):
                    configs[character_dir.name] = str(file)
                    logging.info(f"Found config for {character_dir.name}: {file}")
    logging.info(f"Found {len(configs)} character configs in {base_dir}")
    return configs


def validate_data_paths(config: Dict[str, Any]) -> bool:
    """Validate that all required data paths exist."""
    required_paths = []
    for dataset in config.get("datasets", []):
        required_paths.extend(
            [
                dataset.get("path"),
                dataset.get("meta_file_train"),
                dataset.get("meta_file_val"),
            ]
        )
    for path in required_paths:
        if not path or not os.path.exists(path):
            logging.error(f"Required path does not exist: {path}")
            return False
    return True


def load_config(config_path: str) -> Dict[str, Any]:
    """Load and validate the JSON configuration file."""
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        logging.info(f"Loaded configuration from {config_path}")
        if not validate_data_paths(config):
            raise ValueError(
                "Invalid configuration: Missing or non-existent data paths."
            )
        return config
    except Exception as e:
        logging.error(f"Failed to load configuration from {config_path}: {e}")
        raise


def setup_device(use_gpu: bool, gpu_type: str) -> torch.device:
    """Set up the computation device (CPU, CUDA, or MPS)."""
    if not use_gpu:
        logging.info("Using CPU for training.")
        return torch.device("cpu")
    if gpu_type == "cuda" and torch.cuda.is_available():
        logging.info(f"Using CUDA GPU (Device: {torch.cuda.get_device_name(0)})")
        return torch.device("cuda")
    elif gpu_type == "mps" and torch.backends.mps.is_available():
        logging.info("Using MPS (Apple Silicon GPU)")
        return torch.device("mps")
    else:
        logging.warning(
            f"Requested GPU type '{gpu_type}' not available. Falling back to CPU."
        )
        return torch.device("cpu")


def pre_download_model(
    model_id: str, status_queue: Optional[queue.Queue] = None
) -> bool:
    """Check if the Coqui TTS model exists in cache and download if necessary."""
    try:
        # Construct expected model path in cache
        model_dir = os.path.join(CACHE_DIR, model_id.replace("/", "--"))
        model_exists = os.path.exists(model_dir) and any(Path(model_dir).iterdir())

        if model_exists:
            logging.info(f"Model {model_id} already exists at {model_dir}")
            if status_queue:
                status_queue.put(f"Model {model_id} already exists")
            return True

        logging.info(f"Model {model_id} not found in cache. Attempting to download.")
        if status_queue:
            status_queue.put(f"Downloading model: {model_id}")

        manager = ModelManager()
        manager.download_model(model_id)
        logging.info(f"Successfully downloaded model: {model_id} to {model_dir}")
        if status_queue:
            status_queue.put(f"Model {model_id} downloaded successfully")
        return True
    except Exception as e:
        logging.error(f"Failed to process model {model_id}: {e}")
        if status_queue:
            status_queue.put(f"Error processing model {model_id}: {e}")
        return False


def initialize_model(
    model_id: str, config: Dict[str, Any], device: torch.device, import_source: str
) -> Any:
    try:
        if import_source == "api":
            # Allowlist RAdam for PyTorch 2.6+
            torch.serialization.add_safe_globals([RAdam])
            tts = TTS(model_id).to(device)
            logging.info(f"Initialized TTS model from TTS.api: {model_id}")
            return tts
        else:
            tts_config = load_config(config["config_path"])
            trainer_args = TrainerArgs(
                continue_path=config.get("restore_path"),
            )
            tts = Trainer(trainer_args, config=tts_config).to(device)
            logging.info(f"Initialized Trainer from TTS.bin.train_tts: {model_id}")
            return tts
    except Exception as e:
        logging.error(
            f"Failed to initialize model {model_id}: {e}\n{traceback.format_exc()}"
        )
        raise


def adjust_metadata_paths(dataset_path: str, meta_file_train: str, meta_file_val: str) -> tuple[str, str]:
    """Adjust metadata file paths to ensure compatibility with the expected structure."""
    # Normalize meta file names to basename only to avoid double joining
    meta_file_train_basename = os.path.basename(meta_file_train)
    meta_file_val_basename = os.path.basename(meta_file_val)

    # Check if metadata files exist in the dataset path
    meta_file_train_path = os.path.join(dataset_path, meta_file_train_basename)
    meta_file_val_path = os.path.join(dataset_path, meta_file_val_basename)

    if not os.path.exists(meta_file_train_path) or not os.path.exists(meta_file_val_path):
        # If not found, check one directory above
        parent_dir = os.path.dirname(dataset_path)
        meta_file_train_path = os.path.join(parent_dir, meta_file_train_basename)
        meta_file_val_path = os.path.join(parent_dir, meta_file_val_basename)

    if not os.path.exists(meta_file_train_path) or not os.path.exists(meta_file_val_path):
        raise FileNotFoundError(
            f"Metadata files not found in expected locations: {meta_file_train_path}, {meta_file_val_path}"
        )

    return meta_file_train_path, meta_file_val_path


def train_model(
    config_path: str,
    output_dir: str,
    use_gpu: bool,
    gpu_type: str,
    model_id: str,
    restore_path: Optional[str] = None,
    status_queue: Optional[queue.Queue] = None,
) -> str:
    """Train a TTS model using the provided configuration with enhanced error handling."""
    try:
        config = load_config(config_path)
        if status_queue:
            status_queue.put("Pre-downloading TTS model...")
        if not pre_download_model(model_id, status_queue):
            raise ValueError(f"Failed to pre-download model {model_id}")

        device = setup_device(use_gpu, gpu_type)
        config["config_path"] = config_path
        config["output_path"] = output_dir
        config["restore_path"] = restore_path

        os.makedirs(output_dir, exist_ok=True)
        checkpoint_dir = os.path.join(output_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Configure training parameters
        dataset_config = config.get("datasets", [{}])[0]
        batch_size = config.get("batch_size", 16)
        epochs = config.get("num_epochs", 100)
        run_eval = config.get("run_eval", True)

        # Verify dataset paths
        dataset_path = dataset_config.get("path")
        meta_file_train = dataset_config.get("meta_file_train")
        meta_file_val = dataset_config.get("meta_file_val")
        dataset_name = dataset_config.get("dataset_name", "default_dataset")
        if not all([dataset_path, meta_file_train, meta_file_val]):
            raise ValueError(
                "Dataset configuration missing required fields: path, meta_file_train, meta_file_val"
            )
        from pathlib import Path
        dataset_path_abs = str(Path(dataset_path).resolve())
        if not os.path.exists(dataset_path_abs):
            raise FileNotFoundError(f"Dataset path not found: {dataset_path_abs}")
        # Adjust meta_file_train and meta_file_val to be basenames only and join properly
        meta_file_train_basename = Path(meta_file_train).name
        meta_file_val_basename = Path(meta_file_val).name

        # Fix metadata file path resolution
        meta_file_train_path, meta_file_val_path = adjust_metadata_paths(dataset_path, meta_file_train, meta_file_val)
        logging.info(f"Using training metadata file at: {meta_file_train_path}")
        logging.info(f"Using validation metadata file at: {meta_file_val_path}")

        # Ensure the config object is properly formatted for Trainer
        if isinstance(config, dict):
            from coqpit import Coqpit
            # Remove keys not expected by Coqpit
            filtered_config = {k: v for k, v in config.items() if k not in ["config_path", "output_path", "restore_path"]}
            config = Coqpit(**filtered_config)

        # Training logic
        if import_source == "bin.train_tts" and load_tts_samples and setup_model:
            logging.info(
                "Using Trainer.fit with dataset preprocessing from TTS.bin.train_tts"
            )
            try:
                # Preprocess dataset
                train_samples, eval_samples = load_tts_samples(
                    datasets=[
                        {
                            "path": dataset_path,
                            "meta_file_train": meta_file_train_basename,
                            "meta_file_val": meta_file_val_basename,
                            "formatter": "ljspeech",
                            "dataset_name": dataset_name,
                            "ignored_speakers": [],
                            "language": "en",
                        }
                    ],
                    eval_split=True,
                )
                if not train_samples or not eval_samples:
                    raise ValueError("No training or evaluation samples loaded")
                logging.info(
                    f"Loaded {len(train_samples)} training samples and {len(eval_samples)} evaluation samples"
                )

                # Initialize model
                model_config = load_config(config_path)
                model = setup_model(model_config, samples=train_samples + eval_samples)
                logging.info(f"Model initialized with setup_model: {model_id}")

                # Setup Trainer
                trainer_args = TrainerArgs(
                    continue_path=restore_path,
                )
                trainer = Trainer(
                    trainer_args,
                    output_path=output_dir,  # Pass output_path here
                    config=model_config,
                ).to(device)

                # Train using fit
                trainer.fit(
                    train_samples=train_samples,
                    eval_samples=eval_samples,
                    batch_size=batch_size,
                    epochs=epochs,
                    checkpoint_dir=checkpoint_dir,
                    restore_path=restore_path,
                    evaluate=run_eval,
                )
            except Exception as e:
                logging.warning(
                    f"Trainer.fit with preprocessing failed: {e}\n{traceback.format_exc()}"
                )
                logging.info("Attempting manual dataset handling with Trainer.fit")
                # Fallback: Manual dataset handling
                try:
                    trainer_args = TrainerArgs(
                        continue_path=restore_path,
                    )
                    trainer = Trainer(
                        trainer_args,
                        output_path=output_dir,  # Pass output_path here
                        config=load_config(config_path),
                    ).to(device)
                    trainer.fit(
                        dataset_path=dataset_path,
                        meta_file_train=meta_file_train_basename,
                        meta_file_val=meta_file_val_basename,
                        batch_size=batch_size,
                        epochs=epochs,
                        checkpoint_dir=checkpoint_dir,
                        restore_path=restore_path,
                        evaluate=run_eval,
                    )
                except Exception as e:
                    logging.error(
                        f"Manual dataset handling failed: {e}\n{traceback.format_exc()}"
                    )
                    raise RuntimeError("All Trainer.fit attempts failed")
        else:
            raise RuntimeError(
                "Required TTS.bin.train_tts imports (Trainer, load_tts_samples, setup_model) are unavailable"
            )

        if status_queue:
            status_queue.put("Starting training process...")

        final_model_path = os.path.join(output_dir, "final_model.pth")
        torch.save(model.state_dict(), final_model_path)
        logging.info(f"Final model saved to {final_model_path}")
        if status_queue:
            status_queue.put(f"Final model saved to {final_model_path}")

        config_save_path = os.path.join(output_dir, "config.json")
        with open(config_save_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=4)
        logging.info(f"Configuration saved to {config_save_path}")
        if status_queue:
            status_queue.put(f"Configuration saved to {config_save_path}")

        return final_model_path
    except Exception as e:
        logging.error(f"Training failed: {e}\n{traceback.format_exc()}")
        if status_queue:
            status_queue.put(f"Training failed: {e}")
        raise


def test_model(
    model_path: str,
    config_path: str,
    text: str,
    output_wav: str,
    status_queue: Optional[queue.Queue] = None,
) -> None:
    """Test the trained model by synthesizing a sample text."""
    try:
        config = load_config(config_path)
        device = torch.device("cpu")
        model = initialize_model(config["model"], config, device, import_source)
        model.model.load_state_dict(torch.load(model_path, map_location=device))
        model.model.eval()

        if status_queue:
            status_queue.put(f"Synthesizing text: {text}")
        if import_source == "api":
            wav = model.tts(text=text, speaker=None, language=None)
            model.save_wav(wav, output_wav)
        else:
            synthesizer = Synthesizer(model.model, config)
            wav = synthesizer.tts(text=text, speaker_name=None, language_name=None)
            synthesizer.save_wav(wav, output_wav)
        logging.info(f"Synthesized audio saved to {output_wav}")
        if status_queue:
            status_queue.put(f"Synthesized audio saved to {output_wav}")
    except Exception as e:
        logging.error(f"Failed to synthesize audio: {e}\n{traceback.format_exc()}")
        if status_queue:
            status_queue.put(f"Failed to synthesize audio: {e}")
        raise


# --- GUI ---
def main_gui(base_dir: str = DEFAULT_BASE_DIR) -> None:
    window = tk.Tk()
    window.title("TTS Model Trainer")
    window.geometry("600x500")

    # Variables
    base_dir_var = tk.StringVar(value=base_dir)
    character_var = tk.StringVar()
    output_dir_var = tk.StringVar(value=DEFAULT_OUTPUT_DIR)
    use_gpu_var = tk.BooleanVar(value=False)
    gpu_type_var = tk.StringVar(value="cuda")
    restore_path_var = tk.StringVar(value="")
    test_text_var = tk.StringVar(
        value="Hiya, I’m Hu Tao, director of the Wangsheng Funeral Parlor! Now talking from this computer!"
    )
    test_output_var = tk.StringVar(value="hu_tao_test.wav")
    status_queue = queue.Queue()
    current_thread = [None]
    character_configs = [find_character_configs(base_dir)]

    # Frames
    main_frame = ttk.Frame(window, padding="10")
    main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
    window.columnconfigure(0, weight=1)
    window.rowconfigure(0, weight=1)

    # Input Frame
    input_frame = ttk.Frame(main_frame)
    input_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)

    ttk.Label(input_frame, text="Base Directory:").grid(
        row=0, column=0, sticky=tk.W, pady=2
    )
    ttk.Entry(input_frame, textvariable=base_dir_var, width=30).grid(
        row=0, column=1, sticky=(tk.W, tk.E), pady=2
    )
    ttk.Button(
        input_frame, text="Browse", command=lambda: browse_directory(base_dir_var)
    ).grid(row=0, column=2, padx=5)

    ttk.Label(input_frame, text="Character:").grid(row=1, column=0, sticky=tk.W, pady=2)
    character_combobox = ttk.Combobox(
        input_frame, textvariable=character_var, width=30, state="readonly"
    )
    character_combobox.grid(row=1, column=1, columnspan=2, sticky=(tk.W, tk.E), pady=2)
    character_combobox["values"] = list(character_configs[0].keys())

    ttk.Label(input_frame, text="Output Directory:").grid(
        row=2, column=0, sticky=tk.W, pady=2
    )
    ttk.Entry(input_frame, textvariable=output_dir_var, width=30).grid(
        row=2, column=1, sticky=(tk.W, tk.E), pady=2
    )
    ttk.Button(
        input_frame, text="Browse", command=lambda: browse_directory(output_dir_var)
    ).grid(row=2, column=2, padx=5)

    ttk.Checkbutton(input_frame, text="Use GPU", variable=use_gpu_var).grid(
        row=3, column=0, columnspan=3, sticky=tk.W, pady=2
    )
    ttk.Label(input_frame, text="GPU Type:").grid(row=4, column=0, sticky=tk.W, pady=2)
    ttk.Combobox(
        input_frame, textvariable=gpu_type_var, values=["cuda", "mps"], state="readonly"
    ).grid(row=4, column=1, columnspan=2, sticky=(tk.W, tk.E), pady=2)

    ttk.Label(input_frame, text="Restore Path (optional):").grid(
        row=5, column=0, sticky=tk.W, pady=2
    )
    ttk.Entry(input_frame, textvariable=restore_path_var, width=30).grid(
        row=5, column=1, sticky=(tk.W, tk.E), pady=2
    )
    ttk.Button(
        input_frame,
        text="Browse",
        command=lambda: browse_file(restore_path_var, [("PyTorch files", "*.pth")]),
    ).grid(row=5, column=2, padx=5)

    ttk.Label(input_frame, text="Test Text:").grid(row=6, column=0, sticky=tk.W, pady=2)
    ttk.Entry(input_frame, textvariable=test_text_var, width=30).grid(
        row=6, column=1, columnspan=2, sticky=(tk.W, tk.E), pady=2
    )

    ttk.Label(input_frame, text="Test Output WAV:").grid(
        row=7, column=0, sticky=tk.W, pady=2
    )
    ttk.Entry(input_frame, textvariable=test_output_var, width=30).grid(
        row=7, column=1, sticky=(tk.W, tk.E), pady=2
    )
    ttk.Button(
        input_frame,
        text="Browse",
        command=lambda: browse_file(test_output_var, [("WAV files", "*.wav")]),
    ).grid(row=7, column=2, padx=5)

    input_frame.columnconfigure(1, weight=1)

    # Status Frame
    status_frame = ttk.Frame(main_frame)
    status_frame.grid(
        row=1, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=10
    )
    status_text = tk.Text(status_frame, height=10, width=50, wrap=tk.WORD)
    status_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
    scroll = ttk.Scrollbar(status_frame, orient=tk.VERTICAL, command=status_text.yview)
    scroll.grid(row=0, column=1, sticky=(tk.N, tk.S))
    status_text["yscrollcommand"] = scroll.set
    status_frame.columnconfigure(0, weight=1)
    status_frame.rowconfigure(0, weight=1)

    def browse_file(var, filetypes):
        file = filedialog.askopenfilename(filetypes=filetypes)
        if file:
            var.set(file)

    def browse_directory(var):
        directory = filedialog.askdirectory()
        if directory:
            var.set(directory)
            character_configs[0] = find_character_configs(directory)
            character_combobox["values"] = list(character_configs[0].keys())
            character_var.set("")
            status_text.delete(1.0, tk.END)
            status_text.insert(
                tk.END,
                f"Found {len(character_configs[0])} character configs in {directory}\n",
            )
            status_text.see(tk.END)

    def start_training():
        if current_thread[0] and current_thread[0].is_alive():
            messagebox.showwarning(
                "Process Running", "A training process is already running!"
            )
            return
        character = character_var.get()
        if not character or character not in character_configs[0]:
            messagebox.showerror("Input Error", "Please select a valid character.")
            return
        config_path = character_configs[0][character]
        if not os.path.exists(config_path):
            messagebox.showerror("Input Error", f"Config file not found: {config_path}")
            return
        if not output_dir_var.get():
            messagebox.showerror("Input Error", "Please specify an output directory.")
            return
        if not test_text_var.get():
            messagebox.showerror("Input Error", "Please specify test text.")
            return
        if not test_output_var.get():
            messagebox.showerror(
                "Input Error", "Please specify a test output WAV path."
            )
            return
        status_text.delete(1.0, tk.END)

        config = load_config(config_path)
        model_id = config.get("model", SUPPORTED_MODELS[0])

        def training_thread():
            try:
                model_path = train_model(
                    config_path=config_path,
                    output_dir=output_dir_var.get(),
                    use_gpu=use_gpu_var.get(),
                    gpu_type=gpu_type_var.get(),
                    model_id=model_id,
                    restore_path=restore_path_var.get() or None,
                    status_queue=status_queue,
                )
                test_model(
                    model_path=model_path,
                    config_path=config_path,
                    text=test_text_var.get(),
                    output_wav=test_output_var.get(),
                    status_queue=status_queue,
                )
                status_queue.put(
                    f"Training and testing completed successfully for {character}."
                )
            except Exception as e:
                status_queue.put(f"Process failed for {character}: {e}")

        current_thread[0] = threading.Thread(target=training_thread, daemon=True)
        current_thread[0].start()

    def stop_training():
        if current_thread[0] and current_thread[0].is_alive():
            status_text.insert(tk.END, "Stopping training (may take a moment)...\n")
            status_text.see(tk.END)
            current_thread[0] = None

    def update_status():
        while True:
            try:
                message = status_queue.get_nowait()
                status_text.insert(tk.END, message + "\n")
                status_text.see(tk.END)
            except queue.Empty:
                break
        window.after(100, update_status)

    # Buttons
    button_frame = ttk.Frame(main_frame)
    button_frame.grid(row=2, column=0, columnspan=2, pady=10)
    ttk.Button(button_frame, text="Start Training", command=start_training).grid(
        row=0, column=0, padx=5
    )
    ttk.Button(button_frame, text="Stop", command=stop_training).grid(
        row=0, column=1, padx=5
    )

    main_frame.columnconfigure(0, weight=1)
    main_frame.rowconfigure(1, weight=1)
    update_status()
    window.mainloop()


# --- CLI ---
def main():
    parser = argparse.ArgumentParser(
        description="Train custom TTS models for characters using Coqui TTS.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default=DEFAULT_BASE_DIR,
        help="Base directory containing character folders with JSON configs.",
    )
    parser.add_argument(
        "--character",
        type=str,
        default=None,
        help="Specific character to train (if not specified, trains all found characters).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save training checkpoints and final models.",
    )
    parser.add_argument(
        "--use-gpu", action="store_true", help="Use GPU for training if available."
    )
    parser.add_argument(
        "--gpu-type",
        type=str,
        choices=["cuda", "mps"],
        default="cuda",
        help="Type of GPU to use (cuda for NVIDIA, mps for Apple Silicon).",
    )
    parser.add_argument(
        "--restore-path",
        type=str,
        default=None,
        help="Path to a checkpoint to restore training from (optional).",
    )
    parser.add_argument(
        "--test-text",
        type=str,
        default="Hiya, I’m Hu Tao, director of the Wangsheng Funeral Parlor!",
        help="Text to synthesize after training for testing.",
    )
    parser.add_argument(
        "--test-output",
        type=str,
        default="hu_tao_test.wav",
        help="Path to save the test synthesized audio.",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Launch the GUI instead of running in CLI mode.",
    )

    args = parser.parse_args()

    if args.gui:
        main_gui(base_dir=args.base_dir)
    else:
        character_configs = find_character_configs(args.base_dir)
        if not character_configs:
            logging.error("No character configs found. Exiting.")
            exit(1)

        characters_to_train = (
            [args.character] if args.character else character_configs.keys()
        )
        for character in characters_to_train:
            if character not in character_configs:
                logging.error(f"Character {character} not found in configs. Skipping.")
                continue
            config_path = character_configs[character]
            try:
                config = load_config(config_path)
                model_id = config.get("model", SUPPORTED_MODELS[0])
                logging.info(f"Training model for {character}...")
                model_path = train_model(
                    config_path=config_path,
                    output_dir=args.output_dir,
                    use_gpu=args.use_gpu,
                    gpu_type=args.gpu_type,
                    model_id=model_id,
                    restore_path=args.restore_path,
                    status_queue=None,
                )
                test_model(
                    model_path=model_path,
                    config_path=config_path,
                    text=args.test_text,
                    output_wav=args.test_output,
                    status_queue=None,
                )
                logging.info(f"Successfully trained and tested model for {character}")
            except Exception as e:
                logging.error(f"Failed to process {character}: {e}")
                continue
        if not characters_to_train:
            logging.error("No valid characters to train. Exiting.")
            exit(1)


if __name__ == "__main__":
    main()
