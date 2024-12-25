import os
import urllib.request as request
from src import logger
import zipfile
from src.entity.config_entity import (DataIngestionConfig)

## component-Data Ingestion

class DataIngestion:
    def __init__(self,config:DataIngestionConfig):
        self.config=config
    
    # Downloading the zip file
    def download_file(self):
        if not os.path.exists(self.config.local_data_file):
            filename, headers = request.urlretrieve(
                url = self.config.source_URL,
                filename = self.config.local_data_file
            )
            logger.info(f"{filename} download! with following info: \n{headers}")
        else:
            logger.info(f"File already exists")

    def extract_zip_file(self):
        """
        zip_file_path: str
        Extracts the zip file into the data directory
        Function returns None
        """
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)
        with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
            zip_ref.extractall(unzip_path)
            logger.info(f"Extracted {self.config.local_data_file} to {unzip_path}")
        
        # Replace spaces in extracted file names
        for item in os.listdir(unzip_path):
            old_path = os.path.join(unzip_path, item)
            new_path = os.path.join(unzip_path, item.replace(" ", "_"))
            if old_path != new_path:
                os.rename(old_path, new_path)
                logger.info(f"Renamed {old_path} to {new_path}")