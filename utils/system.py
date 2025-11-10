import os
import sys
from loguru import logger

class SysUtils:
    def __init__(
            self, 
            log_file_path: str,
            log_cons_level: str='INFO',
            log_file_level: str='DEBUG',
    ):
        self.logger = logger
        self.logger.remove()
    
        # Add console handler
        self.logger.add(
            sys.stdout,
            level=log_cons_level,
            colorize=True,
        )
        
        # Add file handler
        self.logger.add(
            log_file_path,
            level=log_file_level,
            rotation="10 MB",  # Rotate when file reaches 10MB
            retention="1 week",  # Keep logs for 1 week
            compression="zip"  # Compress rotated logs
        )
        self.logger.info(f"Logger initialized. Logs saved to: {log_file_path}")

    def from_env(self, env_var: str, default=None):
        try:
            var = os.environ[env_var]
            return var
        except KeyError:
            self.logger.info(
                f'Variable {env_var} not found in environment, using default value {default}'
            )
            return default