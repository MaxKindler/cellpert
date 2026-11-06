# wandb Bayesian hyperparameter sweep for the Transformer Flow model (python run.py --architecture transformer_flow --task SEEN --run_sweep).
# Parameters not listed here are taken from transformer_flow_config.py. Optimised metric: validation DE-F1.

transformer_flow_sweep_config = {
    'method': 'bayes', 
    'metric': {
        'name': 'de_val_score',
        'goal': 'maximize'
    },
    'early_terminate':{
        'type': 'hyperband',
        'min_iter': 70,
        'eta': 3
    },
    'parameters': {
        'learning_rate':{
            'distribution': 'log_uniform_values',
            'min': 1e-6,
            'max': 1e-3
        },
        'epochs': {
            'distribution': 'constant',
            'value': 100
        },
        'batch_size': {
            'values': [32, 64, 128, 256, 512] # No batch_size < 32 because of perperturbation sampler
        },
        'loss_function': {
            'values': ['MSELoss']
        },
        'architecture': {
            'values': ['TransformerFlow']
        },
        'perturbations': {
            'values': [["BI-3406", "Capmatinib", "Paclitaxel", "Oxaliplatin", "Gemcitabine", "Vemurafenib", "Anastrozole", "Minodronic acid", "Everolimus", "Adagrasib"]]
        },
        'hidden_size': {
            'values': [256, 512, 1024, 2048, 4096, 8192]
        },
        'dataset': {
            'values': ['tahoe100M_SW480_filtered_1024']
        },
        'HVG_GENE_NUMBER': {
            'values': [1024]
        },
        'num_heads': {
            'values': [2, 4, 8, 16]
        },
        'num_layers': {
            'values': [1, 2, 4]
        },
        'weight_decay': {
            'distribution': 'log_uniform_values',
            'min': 1e-6,
            'max': 5e-2
        },
        'dropout': {
            'distribution': 'uniform',
            'min': 0.0,
            'max': 0.5
        },
        'reflow': {
            'distribution': 'constant',
            'value': 0
        },
        "n_steps": {
            'distribution': 'constant',
            'value': 100
        },
        "embedding_dim": {
            'values': [128, 256, 512]
        },
        "gate_loss_weight": {
            'distribution': 'uniform',
            'min': 1.0,
            'max': 3.0
        },
        "seed": {
            'distribution': 'constant',
            'value': 123
        }
    }
}