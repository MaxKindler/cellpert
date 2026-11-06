# Final hyperparameters of the Rectified Flow model (MLP velocity network), selected as the
# best run (highest validation DE-F1) of the wandb sweep in rectified_flow_sweep_config.py.
# TASK can be overridden on the command line: python run.py ... --task SEEN

DATASET_NAME = "tahoe100M_SW480_filtered_1024"
HVG_GENE_NUMBER = 1024
EPOCHS = 100
BATCH_SIZE = 256 
LEARNING_RATE = 0.00022108622561383163
HIDDEN_SIZE = 512 
LOSS_FN = "MSELoss"
ARCHITECTURE = "RectifiedFlow"
PERTURBATIONS = ["BI-3406", "Capmatinib", "Paclitaxel", "Oxaliplatin", "Gemcitabine", "Vemurafenib", "Anastrozole", "Minodronic acid", "Everolimus", "Adagrasib"] # [p]"Paclitaxel", "BI-3406", "Everolimus"
UNSEEN_PERTURBATIONS = ["RMC-6236", "4EGI-1", "AT7519"]
DROPOUT = 0.011317246749076693
WEIGHT_DECAY = 0.02903540874594708 
REFLOW = 0
N_STEPS = 100
EMBEDDING_DIM = 128 
SEED = 123
GATE_LOSS_WEIGHT = 2.3355180699800773 
EARLY_STOPPING_PATIENCE = 100
TASK = "UNSEEN-PERT" 
FINGERPRINT_RADIUS = 2
FINGERPRINT_BITS = 1024

rectified_flow_config_defaults = {
    "learning_rate": LEARNING_RATE,
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    "architecture": ARCHITECTURE,
    "model_family": "RectifiedFlow",
    "loss_function": LOSS_FN,
    "hidden_size": HIDDEN_SIZE,
    "dataset": DATASET_NAME,
    "HVG_GENE_NUMBER": HVG_GENE_NUMBER,
    "perturbations": PERTURBATIONS,
    "unseen_perturbations": UNSEEN_PERTURBATIONS,
    "dropout": DROPOUT,
    "weight_decay": WEIGHT_DECAY,
    "reflow": REFLOW,
    "n_steps": N_STEPS,
    "embedding_dim": EMBEDDING_DIM,
    "seed": SEED,
    "gate_loss_weight": GATE_LOSS_WEIGHT,
    "early_stopping_patience": EARLY_STOPPING_PATIENCE,
    "task": TASK,
    "fingerprint_radius": FINGERPRINT_RADIUS,
    "fingerprint_bits": FINGERPRINT_BITS,
}