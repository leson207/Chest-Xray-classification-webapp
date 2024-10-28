from src.Xray.entity.configuration import ConfigurationManager
from src.Xray.components.data_ingestion import DataIngestor

if __name__=="__main__":
    config_manager=ConfigurationManager()
    data_ingestor_config=config_manager.get_data_ingestor_config()
    data_ingestor=DataIngestor(data_ingestor_config)
    data_ingestor.download_data_file()
    data_ingestor.extract_data_file()
    # data_ingestor.sync_to_s3()