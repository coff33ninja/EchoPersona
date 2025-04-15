# enhanced_logger.py
import logging
import sys
import os
import time

# Explicitly import logging.config to avoid attribute errors
import logging.config


# Custom FileHandler with retry logic for Windows file locking
class RetryFileHandler(logging.FileHandler):
    def __init__(
        self, filename, mode="a", encoding=None, delay=False, retries=5, delay_s=1
    ):
        self.retries = retries
        self.delay_s = delay_s
        super().__init__(filename, mode, encoding, delay)

    def _open(self):
        for attempt in range(self.retries):
            try:
                return super()._open()
            except PermissionError as e:
                if attempt == self.retries - 1:
                    raise
                print(
                    f"PermissionError opening log file {self.baseFilename}, retrying in {self.delay_s}s... ({attempt + 1}/{self.retries})",
                    file=sys.stderr,
                )
                time.sleep(self.delay_s)


# Basic configuration dictionary
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
    },
    "handlers": {
        "console": {
            "formatter": "standard",
            "class": "logging.StreamHandler",
            "stream": sys.stdout,
            # Level set dynamically in setup_logger
        },
        "file": {
            "formatter": "standard",
            "class": __name__ + ".RetryFileHandler",  # Use custom handler
            "filename": "app.log",  # Overridden in setup_logger
            "mode": "a",
            "encoding": "utf-8",
            "retries": 5,
            "delay_s": 1,
            # Level set dynamically
        },
    },
    "loggers": {
        "": {
            "handlers": ["console", "file"],
            # Level set dynamically
            "propagate": True,
        }
    },
}


def setup_logger(log_file_path="app.log", level=logging.INFO):
    """
    Configures the logging system.

    Args:
        log_file_path (str): The full path for the log file.
        level (int): The logging level (e.g., logging.INFO, logging.DEBUG).
    """
    # Ensure UTF-8 console encoding on Windows
    if sys.platform.startswith("win"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception as e:
            print(f"Warning: Could not set console to UTF-8: {e}", file=sys.stderr)

    # Ensure the directory for the log file exists with retries
    log_dir = os.path.dirname(log_file_path)
    if log_dir:
        for attempt in range(3):
            try:
                os.makedirs(log_dir, exist_ok=True)
                break
            except OSError as e:
                if attempt == 2:
                    print(
                        f"Error creating log directory {log_dir}: {e}", file=sys.stderr
                    )
                    # Fallback to current directory
                    log_file_path = (
                        os.path.basename(log_file_path) or "fallback_app.log"
                    )
                    print(
                        f"Warning: Falling back to log file: {log_file_path}",
                        file=sys.stderr,
                    )
                    break
                time.sleep(1)

    # Validate write access to log file path
    try:
        with open(log_file_path, "a", encoding="utf-8") as f:
            pass
    except (PermissionError, OSError) as e:
        print(f"Error: Cannot write to log file {log_file_path}: {e}", file=sys.stderr)
        log_file_path = "fallback_app.log"
        print(f"Warning: Falling back to log file: {log_file_path}", file=sys.stderr)

    # Update the config dictionary
    LOGGING_CONFIG["handlers"]["file"]["filename"] = log_file_path
    LOGGING_CONFIG["handlers"]["file"]["level"] = level
    LOGGING_CONFIG["handlers"]["console"][
        "level"
    ] = level  # Match console to specified level
    LOGGING_CONFIG["loggers"][""]["level"] = level

    try:
        logging.config.dictConfig(LOGGING_CONFIG)
        logger = logging.getLogger(__name__)
        logger.info(
            f"Logging configured. Log file: {log_file_path}, Level: {logging.getLevelName(level)}"
        )
    except Exception as e:
        print(
            f"Error setting up logger with dictConfig: {e}. Falling back to basic config.",
            file=sys.stderr,
        )
        try:
            logging.basicConfig(
                level=level,
                format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                handlers=[
                    logging.StreamHandler(sys.stdout),
                    RetryFileHandler(log_file_path, mode="a", encoding="utf-8"),
                ],
            )
            logger = logging.getLogger(__name__)
            logger.error("Fell back to basic logging configuration.")
        except Exception as fallback_e:
            print(
                f"Error in fallback logging setup: {fallback_e}. Using console only.",
                file=sys.stderr,
            )
            logging.basicConfig(
                level=level,
                format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                handlers=[logging.StreamHandler(sys.stdout)],
            )
            logger = logging.getLogger(__name__)
            logger.error("Fell back to console-only logging.")


def get_logger(name):
    """
    Retrieves a logger instance. Call setup_logger first.

    Args:
        name (str): The name for the logger (usually __name__).

    Returns:
        logging.Logger: The logger instance.
    """
    return logging.getLogger(name)


# Example usage
if __name__ == "__main__":
    setup_logger("example.log", level=logging.DEBUG)
    logger = get_logger("example_module")
    logger.debug("This is a debug message.")
    logger.info("This is an info message.")
    logger.warning("This is a warning message.")
    logger.error("This is an error message.")
    logger.critical("This is a critical message.")
    try:
        1 / 0
    except ZeroDivisionError:
        logger.exception("Caught an exception.")
    print("Check example.log for output.")
