# Final hyperparameters of the Transformer Flow model (rectified flow with a cell-attention
# Transformer velocity network), selected as the best run (highest validation DE-F1) of the
# wandb sweep in transformer_flow_sweep_config.py.
# TASK can be overridden on the command line: python run.py ... --task SEEN
# GPU is only logged.

DATASET_NAME = "tahoe100M_SW480_filtered_1024"
HVG_GENE_NUMBER = 1024
EPOCHS = 100
BATCH_SIZE = 256
LEARNING_RATE = 0.000002936887509824933
HIDDEN_SIZE = 8192
LOSS_FN = "MSELoss" 
ARCHITECTURE = "TransformerFlow"
PERTURBATIONS = ["BI-3406", "Capmatinib", "Paclitaxel", "Oxaliplatin", "Gemcitabine", "Vemurafenib", "Anastrozole", "Minodronic acid", "Everolimus", "Adagrasib"] 
UNSEEN_PERTURBATIONS = ["RMC-6236", "4EGI-1", "AT7519"]
NUM_HEADS = 2 
NUM_LAYERS = 4 
DROPOUT = 0.12262553527396296
WEIGHT_DECAY = 0.00003702941055095859
REFLOW = 0
N_STEPS = 100
EMBEDDING_DIM = 512 
SEED = 123
GPU = 2
GATE_LOSS_WEIGHT = 2.2158082226972393
EARLY_STOPPING_PATIENCE = 100
TASK = "UNSEEN-PERT"
FINGERPRINT_RADIUS = 2
FINGERPRINT_BITS = 1024

transformer_flow_config_defaults = {
    "learning_rate": LEARNING_RATE,
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    "architecture": ARCHITECTURE,
    "model_family": "TransformerFlow",
    "loss_function": LOSS_FN,
    "hidden_size": HIDDEN_SIZE,
    "dataset": DATASET_NAME,
    "HVG_GENE_NUMBER": HVG_GENE_NUMBER,
    "perturbations": PERTURBATIONS,
    "unseen_perturbations": UNSEEN_PERTURBATIONS,
    "num_heads": NUM_HEADS,
    "num_layers": NUM_LAYERS,
    "dropout": DROPOUT,
    "weight_decay": WEIGHT_DECAY,
    "reflow": REFLOW,
    "n_steps": N_STEPS,
    "embedding_dim": EMBEDDING_DIM,
    "seed": SEED,
    "gpu": GPU,  # Set the GPU to use (0, 1, 2, etc.)
    "gate_loss_weight": GATE_LOSS_WEIGHT,
    "early_stopping_patience": EARLY_STOPPING_PATIENCE,
    "task": TASK,
    "fingerprint_radius": FINGERPRINT_RADIUS,
    "fingerprint_bits": FINGERPRINT_BITS,
}