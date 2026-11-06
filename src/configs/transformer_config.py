# Final hyperparameters of the cell-attention Transformer (direct prediction, energy-distance loss),
# selected as the best run (highest validation DE-F1) of the wandb sweep in transformer_sweep_config.py.
# TASK can be overridden on the command line: python run.py ... --task SEEN

DATASET_NAME = "tahoe100M_SW480_filtered_1024"
HVG_GENE_NUMBER = 1024 
EPOCHS = 100
BATCH_SIZE = 64
LEARNING_RATE = 0.0000011397150113560763 
NUM_HEADS = 16
NUM_LAYERS = 4
HIDDEN_SIZE = 4096 
LOSS_FN = "EnergyLoss" 
ARCHITECTURE = "Transformer"
PERTURBATIONS = ["BI-3406", "Capmatinib", "Paclitaxel", "Oxaliplatin", "Gemcitabine", "Vemurafenib", "Anastrozole", "Minodronic acid", "Everolimus", "Adagrasib"] 
UNSEEN_PERTURBATIONS = ["RMC-6236", "4EGI-1", "AT7519"]
DROPOUT = 0.009508747409396444
WEIGHT_DECAY = 0.0026052440958904363
SEED = 123
EMBEDDING_DIM = 64
EARLY_STOPPING_PATIENCE = 100
TASK = "UNSEEN-PERT"
FINGERPRINT_RADIUS = 2
FINGERPRINT_BITS = 1024

transformer_config_defaults = {
    "learning_rate": LEARNING_RATE,
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    "architecture": ARCHITECTURE,
    "model_family": "Transformer",
    "loss_function": LOSS_FN,
    "hidden_size": HIDDEN_SIZE,
    "num_heads": NUM_HEADS,
    "num_layers": NUM_LAYERS,
    "dataset": DATASET_NAME,
    "HVG_GENE_NUMBER": HVG_GENE_NUMBER,
    "perturbations": PERTURBATIONS,
    "unseen_perturbations": UNSEEN_PERTURBATIONS,
    "dropout": DROPOUT,
    "weight_decay": WEIGHT_DECAY,
    "seed": SEED,
    "embedding_dim": EMBEDDING_DIM,
    # transformer_cellattention never reflows; kept here so every config dict
    # shares the same schema and `config.reflow` is always safe to read.
    "reflow": 0,
    "n_steps": 100,
    "early_stopping_patience": EARLY_STOPPING_PATIENCE,
    "task": TASK,
    "fingerprint_radius": FINGERPRINT_RADIUS,
    "fingerprint_bits": FINGERPRINT_BITS,
}