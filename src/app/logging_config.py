import logging
import sys 
import pythonjsonlogger as jsonlogger

def setup_json_logging(log_level: str="INFO") -> None:
    root_logger = logging.getLogger() # Get root logger
    root_logger.handlers=[] #remove existing loggers 
    handler = logging.StreamHandler(sys.stdout) # creating a consolde handler

    formatter = jsonlogger.JsonFormatter(
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

