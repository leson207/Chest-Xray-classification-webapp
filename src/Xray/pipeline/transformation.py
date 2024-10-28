from src.Xray.entity.configuration import ConfigurationManager
from src.Xray.components.data_preparation import DataModule

if __name__=="__main__":
    config_manager=ConfigurationManager()

    data_module_config=config_manager.get_data_module_config()
    data_module=DataModule(data_module_config)
    data_module.prepare_data()