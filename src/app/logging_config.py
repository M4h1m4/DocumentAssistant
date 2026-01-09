import logging
import sys 

# WHY: Direct imports - dependencies are declared in pyproject.toml
# - python-json-logger should be installed via uv sync
# - No runtime capability detection - fail fast if dependency is missing
from pythonjsonlogger import jsonlogger

JsonFormatter = jsonlogger.JsonFormatter

def setup_json_logging(log_level: str="INFO") -> None:
    root_logger = logging.getLogger() # Get root logger
    root_logger.handlers=[] #remove existing loggers 
    handler = logging.StreamHandler(sys.stdout) # creating a console handler

    formatter = JsonFormatter(
        fmt='%(asctime)s %(name)s %(levelname)s %(message)s %(pathname)s %(lineno)d',
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    handler.setFormatter(formatter) #attaching the formatter to the handler
    root_logger.addHandler(handler) # attaching the handler to the logger 
    root_logger.setLevel(getattr(logging, log_level.upper())) # setting the log level
    
    # Prevent propagation to avoid duplicate logs
    root_logger.propagate = False

def get_logger(name: str) -> None:
    return logging.getLogger(name)

