"""
FIDEL-Sync Centralized Logging Configuration.
Provides verbose debugging across all components
"""
import logging
import sys
import os
from datetime import datetime

class VerboseLogger:
  '''
  Enhanced logging utility with multiple verbosity levels
  '''
  def __init__(self, name, log_file = None, verbosity_level = logging.DEBUG):
    # Setting the name and verbosity level of the logger
    self.logger = logging.getLogger(name)
    self.logger.setLevel(verbosity_level)
    
    # Remove existing handlers that might be there in the code, we want log data to show on console
    self.logger.handlers = []
    
    # Console handler (sending logs to screen)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(verbosity_level)

    # Detailed formatter with timestamps, file, function, and line number
    formatter = logging.Formatter(
      '%(asctime)s - [%(levelname)-8s] - %(name)s:%(funcName)s():%(lineno)d - %(message)s', datefmt = '%Y-%m-%d %H:%M:%S'
    )
    '''
    Format example:
    2026-02-12 10:30:11 - [INFO    ] - MyApp:main():12 - App started
    '''
    console_handler.setFormatter(formatter)
    self.logger.addHandler(console_handler) 

    # File handler (optional)
    '''
    To save logs into a file, use the function as:
    `VerboseLogger("MyApp", log_file = "logs/app.log")`, the logs will be saved in a file `logs/app.log`
    '''
    if log_file:
      log_dir = os.path.dirname(log_file)
      if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)
      file_handler = logging.FileHandler(log_file)
      file_handler.setLevel(verbosity_level)
      file_handler.setFormatter(formatter)
      self.logger.addHandler(file_handler)

# Shortcuts: instead of always typing `self.logger.xyz(param)`, we can simply write `self.xyz(param)`
def debug(self, message):  # DEBUG level - Detailed internal information
  self.logger.debug(message)

def info(self, message):  # Prints normal informative messages
  self.logger.info(message)

def warning(self, message):  # Used when something that is not an error yet but might become one, e.g., we might want to use it when disk usage exceeds 85%
  self.logger.warning(message)

def error(self, message, exc_info = False):  # This prints an error, and if we set exc_info = True, it will print full exception traceback
  self.logger.error(message, exc_info = exc_info)

def critical(self, message, exc_info = False):  # Major failure
  self.logger.critical(message, exc_info = exc_info)

def func_enter(self, func_name, **kwargs):
  '''
  Logs function entry with parameters.
  Let's say the function is: logger.func_enter("calculate", x = 10, y = 20), 
  the code `", ".join([f"{k} = {type(v).__name__}" for k, v in kwargs.items()]) if kwargs else ""`
  for each parameter:
    Take the name (k), its type (int, str, list), and convert it to a string
  The output will be:
  >>> ENTER calculate(x = int, y = int)
  '''
  args_string = ", ".join([f"{k} = {type(v).__name__}" for k, v in kwargs.items()]) if kwargs else ""
  self.logger.debug(f">>> ENTER {func_name} returns {args_string}")

def func_exit(self, func_name, result = None):
  '''
  Logs function exit with a return type. If you understood the previous function, try finding out what this one does. 
  '''
  result_type = type(result).__name__ if result is not None else "None"
  self.logger.debug(f"<<< EXIT {func_name} returns {result_type}")

def step(self, step_num, description):
  '''
  Log numbered steps. E.g.,
  logger.step(1, "Loading Configuration")
  output: [Step 1] Loading Configuration
  '''
  self.logger.info(f"[Step {step_num}] {description}")
    
def wait(self, wait_reason, timeout_sec = None):
  '''
  Logs wait for events
  '''
  message = f"WAIT: {wait_reason}"
  if timeout_sec:
    message += f"(timeout = {timeout_sec}s)"
  self.logger.debug(message)

def checkpoint(self, checkpoint_name, status = "OK"):
  '''
  Logs checkpoints
  '''
  self.logger.info(f"CHECKPOINT: {checkpoint_name} - Status: {status}")

# Global logger instance
logger = None

def initialize_logger(name = "FIDEL", log_file = "logs/fidel_debug.log", verbosity_level = logging.DEBUG):
  '''
  Initialize global debugger
  '''
  global logger
  logger = VerboseLogger(name, log_file = log_file, verbosity_level = verbosity_level)
  logger.info("=" * 80)
  logger.info(f"LOGGING INITIALIZED: {name}")
  logger.info("=" * 80)
  return logger

def get_logger():
  '''
  Get global logger instance
  '''
  global logger
  if logger is None:
    initialize_logger()
  return logger
