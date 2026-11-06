from .base_trainer import BaseTrainer
import torch
import numpy as np
from tqdm import tqdm
import random

class TransformerCrossAttentionTrainer(BaseTrainer):
    def predict(self, X, drug_id):
        return self.model(X, drug_id)

    def compute_loss(self, outputs, targets):
        return self.criterion(outputs, targets)

    def train_epoch(self, train_dataloader):
        self.model.train()

        total_loss = 0
        for X, y, drug_id in tqdm(train_dataloader):
            X = X.to(self.device)
            y = y.to(self.device)
            drug_id = drug_id.to(self.device)

            self.optimizer.zero_grad()

            outputs = self.predict(X, drug_id)
            loss = self.compute_loss(outputs, y)

            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        # Second return value mirrors RectifiedFlowTrainer.train_epoch's (loss, diff)
        # signature so callers can treat every trainer's train_epoch uniformly.
        return total_loss / len(train_dataloader), None

    def validate(self, val_dataloader):
        self.model.eval()
        val_loss = 0
        val_outputs_list = []
        X_val_list = []
        y_val_list = []
        drug_id_list = []

        for X_val, y_val, val_drug_id in val_dataloader:
            X_val = X_val.to(self.device)
            y_val = y_val.to(self.device)
            val_drug_id = val_drug_id.to(self.device)

            with torch.no_grad():

                val_outputs = self.predict(X_val, val_drug_id)

                loss = self.compute_loss(val_outputs, y_val)

                val_loss += loss.item()
                val_outputs_list.append(val_outputs.cpu().numpy())
                X_val_list.append(X_val.cpu().numpy())
                y_val_list.append(y_val.cpu().numpy())
                drug_id_list.append(val_drug_id.cpu().numpy())


        val_outputs_list = np.concatenate(val_outputs_list, axis=0)
        X_val_list = np.concatenate(X_val_list, axis=0)
        y_val_list = np.concatenate(y_val_list, axis=0)
        drug_id_list = np.concatenate(drug_id_list, axis=0)

        avg_std = val_outputs_list.std(axis=0).mean()

        return val_loss / len(val_dataloader), val_outputs_list, X_val_list, y_val_list, drug_id_list, avg_std

