import os
import torch
import bentoml

from torchview import draw_graph
from torchsummary import summary

from src.Xray.model.architecture import Net
from src.Xray.entity.entity import ModelModuleConfig

from src.logger import logger


class ModelModule:
    def __init__(self, config: ModelModuleConfig):
        self.config=config
        self.model=Net(config.ARGS)
        self.input_shape=config.ARGS.INPUT_SHAPE
    
    def save(self, transform=None):
        os.makedirs(self.config.ARTIFACT_DIR, exist_ok=True)

        self.model.eval()
        scripted_model = torch.jit.script(self.model)
        scripted_model.save(self.config.SCRIPTED_MODEL_PATH)
        torch.save(self.model, self.config.MODEL_PATH)
        torch.save(self.model.state_dict(), self.config.STATE_DICT_PATH)
        bentoml.pytorch.save_model(
            name='model',
            model=self.model,
            custom_objects={
                'test_transformation': transform
            },
            signatures={
                'predict': {
                    'batchable': True,
                    'input_schema': {
                        'image': {
                            'type': 'ndarray',
                            'shape': self.input_shape,
                            'dtype': 'float32'
                        }
                    }
                }
            }
        )
        logger.info('Model saved')
    
    def load_state_dict(self):
        self.model.load_state_dict(torch.load(self.config.STATE_DICT_PATH, weights_only=True))
    
    def load_model(self):
        self.model=torch.load(self.config.MODEL_PATH, weights_only=False)

    def summary(self):
        summary(self.model.to_empty(device='cpu'),self.input_shape)
    
    def visualize(self):
        model_graph = draw_graph(self.model, input_size=self.input_shape, device='meta',
                                 filename='architecture',directory=self.config.ARTIFACT_DIR, save_graph=True)
