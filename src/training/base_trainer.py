from abc import ABC, abstractmethod
from tqdm import tqdm
import torch
import numpy as np
import wandb


class BaseTrainer(ABC):
    """Base trainer with common logic."""
    def __init__(self, model, optimizer, criterion, device):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device


    @abstractmethod
    def predict(self, *args, **kwargs):
        pass

    @abstractmethod
    def compute_loss(self, *args, **kwargs):
        pass


    def train_epoch(self, train_dataloader):
        self.model.train()
        total_loss = 0
        for X, y in train_dataloader:
            X = X.to(self.device)
            y = y.to(self.device)
            self.optimizer.zero_grad()

            outputs = self.predict(X)
            loss = self.compute_loss(outputs, y)

            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(train_dataloader)


    def validate(self, val_dataloader):
        val_loss = 0
        val_outputs_list = []
        y_val_list = []

        for X_val, y_val in val_dataloader:
            X_val = X_val.to(self.device)
            y_val = y_val.to(self.device)
            self.model.eval()
            with torch.no_grad():


                val_outputs = self.predict(X_val)

                loss = self.compute_loss(val_outputs, y_val)


                val_loss += loss.item()
                val_outputs_list.append(val_outputs.cpu().numpy())
                y_val_list.append(y_val.cpu().numpy())


        val_outputs_list = np.concatenate(val_outputs_list, axis=0)
        y_val_list = np.concatenate(y_val_list, axis=0)

        avg_std = val_outputs_list.std(axis=0).mean()

        return val_loss / len(val_dataloader), val_outputs_list, y_val_list, avg_std
