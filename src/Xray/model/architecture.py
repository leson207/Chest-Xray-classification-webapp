from torch import nn
from src.Xray.entity.entity import ModelConfig

class Net(nn.Module):
    def __init__(self, args: ModelConfig):
        super().__init__()
        self.flatten=nn.Flatten()
        self.head=nn.Linear(224*224*3,args.NUM_CLASSES)
    
    def forward(self, x):
        x=self.flatten(x)
        x=self.head(x)

        return x

    def predict(self, x):
        x=self.flatten(x)
        x=self.head(x)

        return x