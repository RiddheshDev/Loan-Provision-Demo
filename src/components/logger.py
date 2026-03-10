import logging
import os
import sys
from datetime import datetime


# Create log file name and path
LOG_FILE = f"{datetime.now().strftime('%d_%m_%Y_%H')}.log"
log_path = os.path.join(os.getcwd(), "logs")
os.makedirs(log_path, exist_ok=True)
log_file_path = os.path.join(log_path, LOG_FILE)

# Define log format
log_format = "[ %(asctime)s ] [line:%(lineno)d] [%(name)s] - %(levelname)s - %(message)s"

# Create handlers
file_handler = logging.FileHandler(log_file_path)
console_handler = logging.StreamHandler(sys.stdout)

# Set format for both handlers
formatter = logging.Formatter(log_format)
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Get root logger and configure it
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Avoid adding duplicate handlers if file re-imports
if not logger.hasHandlers():
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
else:
    # Clear old handlers and re-add to avoid duplicates
    logger.handlers.clear()
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)


class CustomLogger:    
# Create log file name and path
    def __init__(self,timestamp,run_path):
        self.timestamp = timestamp
        self.LOG_FILE = f"{self.timestamp}.log"
        self.log_path = os.path.join(run_path, "logs")
        os.makedirs(self.log_path, exist_ok=True)
        self.log_file_path = os.path.join(self.log_path, LOG_FILE)

    def get_logger(self):
            # Define log format
        log_format = "[ %(asctime)s ] [line:%(lineno)d] [%(name)s] - %(levelname)s - %(message)s"

        # Create handlers
        file_handler = logging.FileHandler(self.log_file_path)
        console_handler = logging.StreamHandler(sys.stdout)

        # Set format for both handlers
        formatter = logging.Formatter(log_format)
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # Get root logger and configure it
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)

        # Avoid adding duplicate handlers if file re-imports
        if not logger.hasHandlers():
            logger.addHandler(file_handler)
            logger.addHandler(console_handler)
        else:
            # Clear old handlers and re-add to avoid duplicates
            logger.handlers.clear()
            logger.addHandler(file_handler)
            logger.addHandler(console_handler)

        return logger
