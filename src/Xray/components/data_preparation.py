import os
import joblib

from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from src.Xray.entity.entity import DataModuleConfig

from src.logger import logger


class DataModule:
    def __init__(self, config: DataModuleConfig):
        self.config=config
        self.test_transform=transforms.Compose([
            transforms.Resize(config.TRANSFORM.RESIZE_SIZE),
            transforms.CenterCrop(config.TRANSFORM.CENTERCROP_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.TRANSFORM.NORMALIZE_MEAN, std=config.TRANSFORM.NORMALIZE_STD),
        ])

        self.train_transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(config.TRANSFORM.RANDOM_ROTATION),
            transforms.ColorJitter(hue=config.TRANSFORM.HUE, brightness=config.TRANSFORM.BRIGHTNESS,
                                   contrast=config.TRANSFORM.CONTRAST,saturation=config.TRANSFORM.SATURATION),
            self.test_transform
        ])
    
    def prepare_data(self):
        logger.info('Save transform')
        os.makedirs(self.config.ARTIFACT_DIR, exist_ok=True)
        joblib.dump(self.train_transform, self.config.TRAIN_TRANSFORM_FILE)
        joblib.dump(self.test_transform, self.config.TEST_TRANSFORM_FILE)

    def setup(self, stage=None):
        logger.info('Setting up the data module')
        self.train=ImageFolder(self.config.TRAIN_DATA_PATH,transform=self.train_transform)
        self.val=ImageFolder(self.config.TEST_DATA_PATH,transform=self.test_transform)
        self.test=ImageFolder(self.config.TEST_DATA_PATH,transform=self.test_transform)

    def create_train_dataloader(self):
        logger.info('Create train data loader')
        return DataLoader(self.train, batch_size=self.config.BATCH_SIZE,
                          shuffle=True, drop_last=True)

    def create_val_dataloader(self):
        logger.info('Create val data loader')
        return DataLoader(self.val, batch_size=self.config.BATCH_SIZE,
                          shuffle=False, drop_last=True)

    def create_test_dataloader(self):
        logger.info('Create test data loader')
        return DataLoader(self.test, batch_size=self.config.BATCH_SIZE,
                          shuffle=False, drop_last=True)