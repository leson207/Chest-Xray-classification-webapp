import os
import yaml
from box import ConfigBox

from src.logger import logger

def read_yaml(path):
    with open(path) as yaml_file:
        content = yaml.safe_load(yaml_file)
        logger.info(f'yaml file: {path} loaded successfully')
        return ConfigBox(content)

def create_directories(paths, verbose=True):
    for path in paths:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f'Create directory at {path}')