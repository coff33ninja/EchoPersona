# enhanced_logger.py
import logging
import sys
import os

# Basic configuration dictionary
LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False, # Keep other loggers if any
    'formatters': {
        'standard': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            'datefmt': '%Y-%m-%d %H:%M:%S',
        },
    },
    'handlers': {
        'console': {
            'level': 'INFO',
            'formatter': 'standard',
            'class': 'logging.StreamHandler',
            'stream': sys.stdout, # Log INFO and above to console
        },
        'file': {
            # Level will be set in setup_logger
            'formatter': 'standard',
            'class': 'logging.FileHandler',
            'filename': 'app.log', # Default filename, will be overridden
            'mode': 'a',           # Append mode
            'encoding': 'utf-8',
        }
    },
    'loggers': {
        '': { # Root logger
            'handlers': ['console', 'file'],
            # Level will be set in setup_logger
            'propagate': True
        }
    }
}

def setup_logger(log_file_path='app.log', level=logging.INFO):
    """
    Configures the logging system.

    Args:
        log_file_path (str): The full path for the log file.
        level (int): The logging level (e.g., logging.INFO, logging.DEBUG).
    """
    # Ensure the directory for the log file exists
    log_dir = os.path.dirname(log_file_path)
    if log_dir and not os.path.exists(log_dir):
        try:
            os.makedirs(log_dir)
        except OSError as e:
            # Use basic print since logger might not be fully configured yet
            print(f"Error creating log directory {log_dir}: {e}", file=sys.stderr)
            # Fallback: try logging to current directory
            log_file_path = os.path.basename(log_file_path)
            if not log_file_path: # Handle edge case of only directory path given
                 log_file_path = 'fallback_app.log'
            print(f"Warning: Falling back to log file: {log_file_path}", file=sys.stderr)


    # Update the config dictionary with the desired file path and level
    LOGGING_CONFIG['handlers']['file']['filename'] = log_file_path
    LOGGING_CONFIG['handlers']['file']['level'] = level
    LOGGING_CONFIG['loggers']['']['level'] = level

    try:
        logging.config.dictConfig(LOGGING_CONFIG)
        logging.info(f"Logging configured. Log file: {log_file_path}, Level: {logging.getLevelName(level)}")
    except Exception as e:
        # Fallback to basic config if dictConfig fails
        print(f"Error setting up logger with dictConfig: {e}. Falling back to basic config.", file=sys.stderr)
        logging.basicConfig(level=level,
                            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                            handlers=[logging.StreamHandler(sys.stdout),
                                      logging.FileHandler(log_file_path, mode='a', encoding='utf-8')])
        logging.error("Fell back to basic logging configuration.")


def get_logger(name):
    """
    Retrieves a logger instance. Call setup_logger first.

    Args:
        name (str): The name for the logger (usually __name__).

    Returns:
        logging.Logger: The logger instance.
    """
    return logging.getLogger(name)

# Example basic usage (if you run this file directly)
if __name__ == '__main__':
    # Example setup call - in real use, this would be called from the main script
    setup_logger('example.log', level=logging.DEBUG)

    # Get a logger instance
    logger = get_logger('example_module')

    # Log messages
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

