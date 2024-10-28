from tqdm.auto import tqdm

import torch
import torch.nn.functional as F

from src.logger import logger
from src.Xray.model.metric import Metrics

from src.Xray.entity.entity import TrainerConfig
from src.Xray.components.data_preparation import DataModule
from src.Xray.components.model_preparation import ModelModule


class Trainer:
    def __init__(self, config: TrainerConfig):
        super().__init__()
        self.config=config
        self.metrics=Metrics()

    def criterion(self, pred, y):
        return F.cross_entropy(pred, y)

    def setup(self, model_module: ModelModule, data_module: DataModule):
        logger.info("Prepare Training")
        
        data_module.setup()
        self.train_loader=data_module.create_train_dataloader()
        self.val_loader=data_module.create_val_dataloader()

        self.model_module=model_module

        self.optimizer=torch.optim.Adam(self.model_module.model.parameters(), lr=self.config.LEARNING_RATE)
        self.scheduler=torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.config.STEP_SIZE, gamma=0.8)

    
    def fit(self):
        logger.info("Enter Training")
        for i in tqdm(range(self.config.EPOCHS)):
            self.model_module.model.train()
            p_bar=tqdm(self.train_loader, desc=f'Epoch {i+1}/{self.config.EPOCHS}: ')
            for imgs, labels in p_bar:
                pred=self.model_module.model(imgs)
                loss=self.criterion(pred,labels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

                self.metrics.current.loss.append(loss.item())
                # self.metrics.current.accuracy.append(accuracy.item())
                # p_bar.set_postfix(loss=loss.item(), accuracy=accuracy.item())
                p_bar.set_postfix(loss=loss.item())
                if(p_bar.n==len(self.train_loader)-1):
                    self.metrics.update()

            self.model_module.model.eval()
            p_bar=tqdm(self.val_loader, desc=f'Epoch {i+1}/{self.config.EPOCHS}: ')
            for imgs, labels in p_bar:
                with torch.no_grad():
                    pred=self.model_module.model(imgs)
                loss=self.criterion(pred,labels)

                self.metrics.current.loss.append(loss.item())
                # self.metrics.current.accuracy.append(accuracy.item())
                p_bar.set_postfix(loss=loss.item())
                if(p_bar.n==len(self.val_loader)-1):
                    self.metrics.update('val')
                    # loss, accuracy = self.metrics.final('val')
                    # p_bar.set_postfix(loss=loss, accuracy=accuracy)
        
        logger.info("Finish Training")
        self.model_module.save()
    
    def evaluate(self):
        pass

    def visualize(self):
        self.metrics.visualize()

    def callback(self):
        pass