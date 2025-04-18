import importlib
import inspect
import logging
import pkgutil
import TTS
import tensorboard
from typing import List, Tuple, Dict

# --- Logging Configuration ---
logger = logging.getLogger()
logger.setLevel(logging.INFO)

log_format = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(log_format)
logger.addHandler(console_handler)

file_handler = logging.FileHandler("tts_inspection.log", encoding="utf-8")
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(log_format)
logger.addHandler(file_handler)

# --- Target Classes and Methods ---
TARGET_CLASSES = [
    "Trainer",
    "TrainerArgs",
    "TrainTTSArgs",
    "TTS",
    "ModelManager",
    "Synthesizer",
]
TARGET_METHODS = [
    "train",
    "fit",
    "tts",
    "save_wav",
    "download_model",
    "load_config",
    "load_tts_samples",
    "setup_model",
]


def list_tts_modules(package, prefix: str = "") -> List[Tuple[str, bool]]:
    """Recursively list all modules and submodules in the TTS package."""
    modules = []
    for _, modname, ispkg in pkgutil.walk_packages(
        package.__path__, prefix=package.__name__ + "."
    ):
        modules.append((modname, ispkg))
        try:
            module = importlib.import_module(modname)
            if ispkg:
                modules.extend(list_tts_modules(module, prefix=modname + "."))
        except Exception as e:
            logging.warning(f"Failed to import {modname}: {e}")
    return modules


def inspect_module(module_name: str) -> Dict[str, List[str]]:
    """Inspect a module for target classes and methods, including signatures."""
    result = {"classes": [], "methods": []}
    try:
        module = importlib.import_module(module_name)
        logging.info(f"\nInspecting module: {module_name}")

        # Inspect classes
        for name, obj in inspect.getmembers(module, inspect.isclass):
            if name in TARGET_CLASSES:
                result["classes"].append(name)
                logging.info(f"  Class: {name}")
                # Inspect methods in the class
                for method_name, method in inspect.getmembers(obj, inspect.isfunction):
                    if method_name in TARGET_METHODS:
                        sig = str(inspect.signature(method))
                        result["methods"].append(f"{name}.{method_name}{sig}")
                        logging.info(f"    Method: {method_name}{sig}")

        # Inspect top-level functions
        for name, obj in inspect.getmembers(module, inspect.isfunction):
            if name in TARGET_METHODS:
                sig = str(inspect.signature(obj))
                result["methods"].append(f"{name}{sig}")
                logging.info(f"  Function: {name}{sig}")

        if not result["classes"]:
            logging.info("  No target classes found.")
        if not result["methods"]:
            logging.info("  No target methods found.")

        return result
    except Exception as e:
        logging.error(f"Failed to inspect module {module_name}: {e}")
        return result


def detect_tensorboard_conflicts():
    """Detect if TensorBoard is running and causing conflicts."""
    import psutil
    for process in psutil.process_iter(['pid', 'name']):
        if 'tensorboard' in process.info['name'].lower():
            logging.warning(f"TensorBoard is already running with PID {process.info['pid']}. This might cause conflicts.")
            return True
    return False


def main():
    logging.info("Inspecting TTS modules for training-related classes and methods...")

    # Detect TensorBoard conflicts
    if detect_tensorboard_conflicts():
        logging.warning("TensorBoard conflicts detected. Consider stopping existing TensorBoard instances.")

    modules = list_tts_modules(TTS)

    # Filter to relevant modules
    relevant_modules = [
        modname
        for modname, _ in modules
        if any(
            part in modname
            for part in ["bin.train_tts", "api", "utils.manage", "utils.synthesizer"]
        )
    ]

    # Inspect each relevant module
    training_components = {}
    for modname in relevant_modules:
        training_components[modname] = inspect_module(modname)

    # Summarize findings
    logging.info("\nSummary of Training Components:")
    for modname, components in training_components.items():
        if components["classes"] or components["methods"]:
            logging.info(f"\nModule: {modname}")
            if components["classes"]:
                logging.info("  Classes:")
                for cls in components["classes"]:
                    logging.info(f"    - {cls}")
            if components["methods"]:
                logging.info("  Methods/Functions:")
                for method in components["methods"]:
                    logging.info(f"    - {method}")


if __name__ == "__main__":
    main()
