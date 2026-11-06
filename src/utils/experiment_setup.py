import os
import random
import numpy as np
import torch
import wandb

# wandb project / entity (user or team) that runs are logged to. Override with the
# environment variables WANDB_PROJECT / WANDB_ENTITY; if WANDB_ENTITY is not set,
# wandb uses the default entity of the logged-in account.
WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "cellpert")
WANDB_ENTITY = os.environ.get("WANDB_ENTITY", None)

def set_seed(seed):
    """
    Set the random seed for reproducibility.

    Args:
        seed (int): The seed value to set.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def init_wandb_run(config_defaults=None):
    """
    Initialize a Weights & Biases run.

    Args:
        project_name (str): The name of the wandb project.
        run_name (str, optional): The name of the specific run. Defaults to None.
        config (dict, optional): Configuration parameters for the run. Defaults to None.
    """
    run = wandb.init(config=config_defaults, project=WANDB_PROJECT, entity=WANDB_ENTITY)#, name=f"TRAIN{config_defaults["task"]}_{config_defaults['architecture']}_seed{config_defaults['seed']}")
    run.name = f"TRAIN{config_defaults['task']}-{config_defaults['model_family']}-seed{config_defaults['seed']}-{run.name}"
    run_name = run.name
    config = wandb.config
    return run_name, config


def log_and_finish_wandb_run(test_loss, model_results, mean_baseline_results, mean_shift_baseline_results):
    """
    Log the test metrics of the model and of the baselines to Weights & Biases.

    Note: the key "de_precition_test_score_all" is misspelled on purpose; the result
    tables exported from wandb (plots/data/results) and the notebooks use this name.
    The run is finished by the caller (run.py), so that reflow rounds can still log to it.

    Args:
        test_loss (float): The test loss value.
        model_results (dict): Evaluation metrics for the model.
        mean_baseline_results (dict): Evaluation metrics for the mean baseline.
        mean_shift_baseline_results (dict): Evaluation metrics for the mean shift baseline.
    """
    wandb.log({
            "test_loss": test_loss, 
            "de_test_score": np.mean(model_results["DE_F1"]), 
            "de_precision_test_score": np.mean(model_results["DE_Precision"]), 
            "de_recall_test_score": np.mean(model_results["DE_Recall"]), 
            "energy_test_score": np.mean(model_results["Energy_Loss"]), 
            "mse_test_loss": np.mean(model_results["MSE"]),
            "mean_baseline_de_score": np.mean(mean_baseline_results["DE_F1"]),
            "mean_baseline_precision_score": np.mean(mean_baseline_results["DE_Precision"]),
            "mean_baseline_recall_score": np.mean(mean_baseline_results["DE_Recall"]),
            "mean_baseline_energy_score": np.mean(mean_baseline_results["Energy_Loss"]),
            "mean_baseline_mse_score": np.mean(mean_baseline_results["MSE"]),
            "mean_shift_baseline_de_score": np.mean(mean_shift_baseline_results["DE_F1"]),
            "mean_shift_baseline_precision_score": np.mean(mean_shift_baseline_results["DE_Precision"]),
            "mean_shift_baseline_recall_score": np.mean(mean_shift_baseline_results["DE_Recall"]),
            "mean_shift_baseline_energy_score": np.mean(mean_shift_baseline_results["Energy_Loss"]),
            "mean_shift_baseline_mse_score": np.mean(mean_shift_baseline_results["MSE"]),
            "de_test_score_all": model_results["DE_F1"],
            "de_precition_test_score_all": model_results["DE_Precision"],
            "de_recall_test_score_all": model_results["DE_Recall"],
            "energy_test_score_all": model_results["Energy_Loss"],
            "mse_test_loss_all": model_results["MSE"],
            "mean_baseline_de_score_all": mean_baseline_results["DE_F1"],
            "mean_baseline_precision_score_all": mean_baseline_results["DE_Precision"],
            "mean_baseline_recall_score_all": mean_baseline_results["DE_Recall"],
            "mean_baseline_energy_score_all": mean_baseline_results["Energy_Loss"],
            "mean_baseline_mse_score_all": mean_baseline_results["MSE"],
            "mean_shift_baseline_de_score_all": mean_shift_baseline_results["DE_F1"],
            "mean_shift_baseline_precision_score_all": mean_shift_baseline_results["DE_Precision"],
            "mean_shift_baseline_recall_score_all": mean_shift_baseline_results["DE_Recall"],
            "mean_shift_baseline_energy_score_all": mean_shift_baseline_results["Energy_Loss"],
            "mean_shift_baseline_mse_score_all": mean_shift_baseline_results["MSE"]
    })