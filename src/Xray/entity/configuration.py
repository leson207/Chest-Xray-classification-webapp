from src.Xray.entity.entity import *
from src.Xray.utils.common import read_yaml, create_directories

class ConfigurationManager:
    def __init__(self, config_file_path='params.yaml'):
        self.config = read_yaml(config_file_path)

        create_directories([self.config.artifacts_root])

    def get_data_ingestor_config(self) -> DataIngestorConfig:
        config= self.config.data_ingestion
        create_directories([config.ARTIFACT_DIR])

        data_ingestor_config= DataIngestorConfig(
            **{k: v for k, v in config.items()}
        )

        return data_ingestor_config

    def get_transform_config(self) -> TransformConfig:
        config= self.config.transform
        transform_config= TransformConfig(
            **{k: v for k, v in config.items()}
        )

        return transform_config

    def get_data_module_config(self) -> DataModuleConfig:
        config= self.config.data_preparation
        data_module_config= DataModuleConfig(
            **{k: v for k, v in config.items()}
        )
        data_module_config.TRANSFORM=self.get_transform_config()
        return data_module_config
    
    def get_model_config(self) -> ModelConfig:
        config= self.config.model
        model_config= ModelConfig(
            **{k: v for k, v in config.items()}
        )

        return model_config

    def get_model_module_config(self) -> ModelModuleConfig:
        config= self.config.model_preparation
        model_module_config= ModelModuleConfig(
            **{k: v for k, v in config.items()}
        )
        model_module_config.ARGS=self.get_model_config()
        return model_module_config

    def get_trainer_config(self) -> TrainerConfig:
        config= self.config.model_trainining
        trainer_config= TrainerConfig(
            **{k: v for k, v in config.items()}
        )

        return trainer_config
    
    def get_evaluator_config(self) -> EvaluatorConfig:
        config= self.config.model_evaluation
        evaluator_config= EvaluatorConfig(
            **{k: v for k, v in config.items()}
        )
        evaluator_config.ARGS=self.get_model_config()
        return evaluator_config