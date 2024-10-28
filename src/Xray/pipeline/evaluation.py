from src.Xray.entity.configuration import ConfigurationManager
from src.Xray.components.model_evaluation import Evaluator

if __name__=="__main__":
    config_manager=ConfigurationManager()

    evaluator_config=config_manager.get_evaluator_config()
    evaluator=Evaluator(evaluator_config)
    evaluator.eval()