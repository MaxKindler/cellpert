# wandb Bayesian hyperparameter sweep for the Rectified Flow model (run with: python run.py --architecture rectified_flow --task SEEN --run_sweep).
# Parameters not listed here are taken from rectified_flow_config.py. Optimised metric: validation DE-F1.

rectified_flow_sweep_config = {
    'method': 'bayes', 
    'metric': {
        'name': 'de_val_score',
        'goal': 'maximize'
    },
    'early_terminate':{
        'type': 'hyperband',
        'min_iter': 80, # minimum number of iterations per configuration 
        'eta': 3 
    },
    'parameters': {
        'learning_rate':{
            'distribution': 'log_uniform_values',
            'min': 1e-6,
            'max': 1e-2
        },
        'epochs': {
            'distribution': 'constant',
            'value': 100
        },
        'batch_size': {
            'values': [32, 64, 128, 256, 512]
        },
        'loss_function': {
            'values': ['MSELoss']
        },
        'architecture': {
            'values': ['RectifiedFlow']
        },
        'perturbations': {
            'values': [["BI-3406", "Capmatinib", "Paclitaxel", "Oxaliplatin", "Gemcitabine", "Vemurafenib", "Anastrozole", "Minodronic acid", "Everolimus", "Adagrasib"]]
        },
        'hidden_size': {
            'values': [256, 512, 1024, 2048, 4096, 8192]
        },
        'embedding_dim': {
            'values': [128, 256, 512]
        },
        'dataset': {
            'values': ['tahoe100M_SW480_filtered_1024']
        },
        'HVG_GENE_NUMBER': {
            'values': [1024]
        },
        'weight_decay': {
            'distribution': 'log_uniform_values',
            'min': 1e-6,
            'max': 5e-2
        },
        "gate_loss_weight": {
            'distribution': 'uniform',
            'min': 1.0,
            'max': 3.0
        },
        'dropout': {
            'distribution': 'uniform',
            'min': 0.0,
            'max': 0.7
        },
    }
}