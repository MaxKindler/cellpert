# Registry of the three architectures that can be selected with `python run.py --architecture <name>`.
# Each entry defines the model class and its arguments, the trainer and its loss, how mini-batches
# are built ("per_perturbation": every batch contains a single drug, needed by the Transformers whose
# attention runs across the cells of a batch) and the default config.

import torch

from models import MLP, TransformerCrossAttentionMultiModel, TransformerVelocityModel
from training import RectifiedFlowTrainer, TransformerCrossAttentionTrainer, TransformerFlowTrainer
from utils import CombinedLoss, compute_morgan_fingerprints
from .rectified_flow_config import rectified_flow_config_defaults
from .transformer_config import transformer_config_defaults
from .transformer_flow_config import transformer_flow_config_defaults


def _flow_trainer_kwargs(model, optimizer, cfg, device):
    return dict(
        scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.99, patience=1000),
        criterion=torch.nn.MSELoss(),
        gate_loss_weight=cfg.gate_loss_weight,
    )


def _perturbation_fingerprints(cfg):
    # Order matches the drug-ID convention in process_adata_unseen_pert: train
    # perturbations get IDs 0..len(perturbations)-1, unseen ones are offset
    # right after them, so row i of this tensor must line up with drug ID i.
    drug_names = list(cfg.perturbations) + list(cfg.unseen_perturbations)
    return compute_morgan_fingerprints(drug_names, radius=cfg.fingerprint_radius, n_bits=cfg.fingerprint_bits)


ARCHITECTURES = {
    "rectified_flow": {
        "model_cls": MLP,
        "model_kwargs": lambda cfg, input_dim: dict(
            in_dim=input_dim,
            out_dim=input_dim,
            time_dim=64,
            context_dim=cfg.embedding_dim,
            h=cfg.hidden_size,
            dropout=cfg.dropout,
            fingerprints=_perturbation_fingerprints(cfg),
        ),
        "trainer_cls": RectifiedFlowTrainer,
        "trainer_kwargs": _flow_trainer_kwargs,
        "dataset_batching": "standard",
        "config_defaults": rectified_flow_config_defaults,
    },
    "transformer_cellattention": {
        "model_cls": TransformerCrossAttentionMultiModel,
        "model_kwargs": lambda cfg, input_dim: dict(
            n_genes=input_dim,
            hidden_size=cfg.hidden_size,
            embed_dim=cfg.embedding_dim,
            num_heads=cfg.num_heads,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout,
            fingerprints=_perturbation_fingerprints(cfg),
        ),
        "trainer_cls": TransformerCrossAttentionTrainer,
        # accepts (model, optimizer, criterion, device).
        "trainer_kwargs": lambda model, optimizer, cfg, device: dict(
            criterion=CombinedLoss(sinkhorn_weight=0.0, energy_weight=1.0, mse_weight=0.0, blur=0.05),
        ),
        "dataset_batching": "per_perturbation",
        "config_defaults": transformer_config_defaults,
    },
    "transformer_flow": {
        "model_cls": TransformerVelocityModel,
        "model_kwargs": lambda cfg, input_dim: dict(
            n_genes=input_dim,
            hidden_size=cfg.hidden_size,
            embed_dim=cfg.embedding_dim,
            time_dim=64,
            num_heads=cfg.num_heads,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout,
            fingerprints=_perturbation_fingerprints(cfg),
        ),
        "trainer_cls": TransformerFlowTrainer,
        "trainer_kwargs": _flow_trainer_kwargs,
        "dataset_batching": "per_perturbation", # "standard"
        "config_defaults": transformer_flow_config_defaults,
    },
}
