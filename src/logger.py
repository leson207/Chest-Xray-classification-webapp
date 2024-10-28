import os
import sys
import logging
from datetime import datetime

logging_format = '[%(asctime)s: %(levelname)s]: %(module)s: %(message)s'

log_dir = 'logs'
log_file = 'running_logs.log'
# log_file = f'{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.log'
log_filepath=os.path.join(log_dir, log_file)
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level = logging.INFO,
    format = logging_format,
    handlers=[
        logging.FileHandler(log_filepath),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger('KidneyLogger')