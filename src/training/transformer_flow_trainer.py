from .base_trainer import BaseTrainer
import torch
import torch.nn.functional as F
import numpy as np
from torch.distributions.beta import Beta


class TransformerFlowTrainer(BaseTrainer):
    """Trainer for rectified flow model with MSE loss."""
    def __init__(self, model, optimizer, scheduler, criterion, device, gate_loss_weight=1.0):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.device = device
        self.gate_loss_weight = gate_loss_weight


    def predict(self, z0, z1, context):
        # Sample t from uniform distribution
        t = torch.full((z0.size(0), 1), torch.rand(1).item(), device=self.device) 
        z_t = z0 * (1 - t) + z1 * t  # Linear interpolation between z0 and z1

        velocity, gate_logits = self.model(z_t, t, context)
        return velocity, gate_logits

    def predict_full(self, z0, context, n_steps=100):
        z0 = z0.to(self.device)
        context = context.to(self.device)
        dt = 1.0 / n_steps
        z_t = z0.clone().to(self.device)

        with torch.no_grad():
            for i in range(n_steps):
                t = torch.full((z0.size(0), 1), i * dt, device=z0.device)
                velocity, gate_logits = self.model(z_t, t, context)
                z_t = z_t + velocity * dt

            # Zero-mask via the learned gate (predicted from the final step, t close to 1)
            # instead of a hard magnitude threshold on the integrated state.
            z_t = torch.relu(z_t)
            zero_mask = torch.sigmoid(gate_logits) > 0.5
            z_t = z_t.masked_fill(zero_mask, 0.0)
        return z_t


    def compute_loss(self, velocity_pred, velocity_target, gate_logits, gate_target):
        velocity_loss = self.criterion(velocity_pred, velocity_target)
        gate_loss = F.binary_cross_entropy_with_logits(gate_logits, gate_target)
        total_loss = velocity_loss + self.gate_loss_weight * gate_loss
        return total_loss, velocity_loss, gate_loss


    def train_epoch(self, train_dataloader):
        self.model.train()
        total_loss = 0
        differences = []
        for z0, z1, context in train_dataloader:
            z0 = z0.to(self.device)
            z1 = z1.to(self.device)
            context = context.to(self.device)
            velocity_target = z1 - z0
            gate_target = (z1 == 0).float()

            self.optimizer.zero_grad()

            velocity_pred, gate_logits = self.predict(z0, z1, context)
            loss, velocity_loss, gate_loss = self.compute_loss(velocity_pred, velocity_target, gate_logits, gate_target)

            diff = torch.mean(velocity_pred - velocity_target).detach().cpu().numpy()
            differences.append(diff)

            loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.scheduler.step(loss.item())
            total_loss += loss.item()
        return total_loss / len(train_dataloader), np.array(differences)
    

    def validate(self, val_dataloader, n_steps=100):
        self.model.eval()
        val_loss = 0
        z0_list = []
        z1_pred_list = []
        z1_list = []
        context_list = []

        for z0, z1, context in val_dataloader:
            z0 = z0.to(self.device)
            z1 = z1.to(self.device)
            context = context.to(self.device)

            z1_pred = self.predict_full(z0, context, n_steps)

            loss = self.criterion(z1_pred, z1)

            val_loss += loss.item()
            z0_list.append(z0.cpu().numpy())
            z1_pred_list.append(z1_pred.cpu().numpy())
            z1_list.append(z1.cpu().numpy())
            context_list.append(context.cpu().numpy())


        z0_list = np.concatenate(z0_list, axis=0)
        z1_pred_list = np.concatenate(z1_pred_list, axis=0)
        z1_list = np.concatenate(z1_list, axis=0)
        context_list = np.concatenate(context_list, axis=0)

        avg_std = z1_pred_list.std(axis=0).mean()

        return val_loss / len(val_dataloader), z1_pred_list, z0_list, z1_list, context_list, avg_std
    