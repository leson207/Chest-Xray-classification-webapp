from src.Xray.entity.configuration import ConfigurationManager
from src.Xray.components.data_ingestion import DataIngestor
from src.Xray.components.data_preparation import DataModule
from src.Xray.components.model_preparation import ModelModule
from src.Xray.components.model_training import Trainer
from src.Xray.components.model_evaluation import Evaluator

if __name__=="__main__":
    config_manager=ConfigurationManager()
    data_ingestor_config=config_manager.get_data_ingestor_config()
    data_ingestor=DataIngestor(data_ingestor_config)
    # data_ingestor.download_data_file()
    # data_ingestor.extract_data_file()
    # data_ingestor.sync_to_s3()

    data_module_config=config_manager.get_data_module_config()
    data_module=DataModule(data_module_config)
    data_module.setup()
    train_loader=data_module.create_train_dataloader()
    val_loader=data_module.create_val_dataloader()
    test_loader=data_module.create_test_dataloader()
    batch=next(iter(test_loader))
    print(batch[0].shape)

    model_module_config=config_manager.get_model_module_config()
    model_module=ModelModule(model_module_config)
    model_module.visualize()
    model_module.summary()
    print(model_module.model(batch[0]).shape)

    trainer_config=config_manager.get_trainer_config()
    trainer=Trainer(trainer_config)
    trainer.setup(model_module, data_module)
    trainer.fit()

    evaluator_config=config_manager.get_evaluator_config()
    evaluator=Evaluator(evaluator_config)
    evaluator.eval()