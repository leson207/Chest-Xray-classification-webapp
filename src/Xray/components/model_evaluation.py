import joblib
from tqdm.auto import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from src.logger import logger
from src.Xray.model.architecture import Net
from src.Xray.entity.entity import EvaluatorConfig


class Evaluator:
    def __init__(self, config:EvaluatorConfig):
        self.models={}
        model=torch.load(config.MODEL_PATH,weights_only=False)
        self.models['model']=model

        model=torch.jit.load(config.SCRIPTED_MODEL_PATH)
        self.models['scripted_model']=model

        model=Net(config.ARGS)
        model.load_state_dict(torch.load(config.STATE_DICT_PATH, weights_only=True))
        self.models['state_dict_model']=model

        transform=joblib.load(config.TEST_TRANFORM_PATH)
        dataset=ImageFolder(config.TEST_DATA_DIR, transform=transform)
        self.data_loader=DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=False, drop_last=False)
    
    def eval(self):
        for name, model in self.models.items():
            self.single_eval(model)
            logger.info(f'Finnish Evaluation for {name}')


    def single_eval(self, model):
        model.eval()
        p_bar=tqdm(self.data_loader)
        for imgs, labels in p_bar:
            with torch.no_grad():
                pred=model(imgs)
            loss=F.cross_entropy(pred,labels)
            correct=(pred.argmax(dim=-1)==labels).sum().item()
            acc=correct/labels.size(0)
            p_bar.set_postfix(loss=loss.item(), acc=acc)
    