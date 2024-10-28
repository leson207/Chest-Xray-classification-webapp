from box import Box
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

class Metrics():
    def __init__(self):
        self.current = Box({'loss': [],
                          'accuracy': []})
        self.total = Box({'loss': {'train': [],'val': []},
                          'accuracy': {'train': [],'val': []}})

    def update(self,mode='train'):
        self.total.loss[mode].append(self.current.loss)
        self.total.accuracy[mode].append(self.current.accuracy)

        self.current.loss=[]
        self.current.accuracy=[]

        return

    def final(self, mode='train'):
        if mode=='train':
            loss = self.total.loss.train[-1][-1]
            accuracy = self.total.accuracy.train[-1][-1]

        else:
            loss = sum(self.total.loss.val[-1])/len(self.total.loss.val[-1])
            if len(self.total.accuracy.val[-1])!=0:
                accuracy = sum(self.total.accuracy.val[-1])/len(self.total.accuracy.val[-1])
                return loss, accuracy

        return loss

    def visualize(self):
        fix , axes = plt.subplots(1,2,figsize=(15,5))
        for i, metric in enumerate(['loss','accuracy']):
            flat_train = [x for epoch in self.total[metric]['train'] for x in epoch]
            flat_val = [x for epoch in self.total[metric]['val'] for x in epoch]

            target_len = len(flat_train)
            steps=np.linspace(0, target_len - 1, len(flat_val))
            new_steps = np.arange(target_len)
            val_interp = np.interp(new_steps, steps, flat_val)

            df=pd.DataFrame({
                'Step': np.concatenate([new_steps, new_steps]),
                'Loss': np.concatenate([flat_train, val_interp]),
                'Type': ['Train'] * target_len + ['Validation'] * target_len
            })
            sns.lineplot(data=df, x='Step', y='Loss', hue='Type', marker='o', ax=axes[i])

            num_epochs = len(self.total[metric]['train'])
            train_steps = len(self.total[metric]['train'][0])
            boundaries=np.cumsum([train_steps]*num_epochs)
            for boundary in boundaries[:-1]:
                axes[i].axvline(boundary - 1, color='r', linestyle='--', label='Sublist boundary')

            axes[i].set_xlabel('Step')
            axes[i].set_ylabel(metric)
            axes[i].set_title(f'Training and Validation {metric} at Each Step (Interpolated)')

            axes[i].legend()
            axes[i].grid(False)

        plt.show()