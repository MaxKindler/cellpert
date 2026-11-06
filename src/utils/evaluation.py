import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from utils import des_loss, energy_loss

def evaluate_per_perturbation(pred, y, x, drug_id_list, perturbations, var):
    """
    Evaluate the model's performance on a set of perturbations.

    Args:
        pred: The model's predictions.
        y: The true labels.
        x: The input data.
        drug_id_list: A list of drug IDs corresponding to the input data.
        perturbations: A list of perturbation functions to apply to the input data.
        var: Additional variables or parameters needed for evaluation.    
    
    Returns:
        A dictionary containing the evaluation metrics for each perturbation.
    """

    results = {
        "DE_F1": [],
        "DE_Precision": [],
        "DE_Recall": [],
        "Energy_Loss": [],
        "MSE": []
    }

    for c in np.unique(drug_id_list):
        idx = drug_id_list == c
        idx = idx.squeeze()
        perturbation = perturbations[int(c.item())]

        # METRICS
        de_f1_score, de_precision, de_recall = des_loss(pred[idx], y[idx], x[idx], var, testing=True)

        # Compute energy loss on a subset of the data to avoid online backend in geomloss (we want tensorized backend)
        if len(pred[idx]) > 5000:
            energy_test_losses = []
            for i in range(0, len(pred[idx]), 5000):
                energy_test_losses.append(energy_loss(pred[idx][i:i+5000], y[idx][i:i+5000]))
            energy_test_loss = np.mean(energy_test_losses)
        else:
            energy_test_loss = energy_loss(pred[idx], y[idx])

        mse_test_loss = F.mse_loss(torch.from_numpy(pred[idx]), torch.from_numpy(y[idx])).item()
        print(f"Perturbation {perturbation}: DE F1 Score: {de_f1_score:.4f}, Precision: {de_precision:.4f}, Recall: {de_recall:.4f}, Energy Loss: {energy_test_loss:.4f}, MSE Loss: {mse_test_loss:.4f}")

        results["DE_F1"].append(de_f1_score)
        results["DE_Precision"].append(de_precision)
        results["DE_Recall"].append(de_recall)
        results["Energy_Loss"].append(energy_test_loss)
        results["MSE"].append(mse_test_loss)
        
    return results