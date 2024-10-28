import os
import gdown
import zipfile

from src.logger import logger
from src.Xray.entity.entity import DataIngestorConfig

class DataIngestor:
    def __init__(self, config:DataIngestorConfig):
        self.config = config
    
    def sync_to_s3(self):
        command = f'aws s3 sync {self.config.ARTIFACT_DIR} s3://{self.config.BUCKET_NAME}/{self.config.CLOUD_DIR}'
        os.system(command)
        logger.info(f"Sync data from {self.config.LOCAL_DIR} to s3://{self.config.BUCKET_NAME}/{self.config.CLOUD_DIR}")
    
    def sync_from_s3(self):
        command = f'aws s3 sync s3://{self.config.BUCKET_NAME}/{self.config.CLOUD_DIR} {self.config.ARTIFACT_DIR}'
        os.system(command)
        logger.info(f"Sync data from s3://{self.config.BUCKET_NAME}/{self.config.CLOUD_DIR} to {self.config.CLOUD_DIR}")

    def download_data_file(self):
        dataset_url = self.config.DATA_SOURCE
        download_path = self.config.ARTIFACT_DIR
        os.makedirs(download_path, exist_ok=True)
        logger.info(f'Downloading data from {dataset_url} into {download_path}')

        file_id = dataset_url.split("/")[-2]
        prefix = 'https://drive.google.com/uc?id='
        output_file = os.path.join(download_path, 'data_file')
        gdown.download(prefix+file_id, download_path+'/data.zip', quiet=False)

        logger.info(f'Downloaded data from {dataset_url} into file {output_file}')

    def extract_data_file(self):
        extract_path = self.config.ARTIFACT_DIR
        with zipfile.ZipFile(self.config.DATA_FILE_PATH, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
