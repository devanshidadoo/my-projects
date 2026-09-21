import logging
import os
from datetime import datetime


# Logger function to dynamically create a logger per conversation ID
def get_logger(conv_id: str) -> logging.Logger:
    """
    Get a logger instance for a specific conversation.
    
    Args:
        conv_id: Conversation ID to use in the log filename
        
    Returns:
        Logger instance configured for the conversation
    """
    # Create logs directory if it doesn't exist
    logs_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
    os.makedirs(logs_dir, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger(f"conversation_{conv_id}")
    logger.setLevel(logging.INFO)
    
    # Create file handler
    log_file = os.path.join(logs_dir, f"{conv_id}.log")
    handler = logging.FileHandler(log_file)
    handler.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(handler)
    
    return logger
