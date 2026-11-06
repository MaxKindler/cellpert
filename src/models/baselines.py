import numpy as np
import torch
import torch.nn.functional as F
from utils import des_loss, energy_loss

def compute_mean_baselines(train_dataset, X_test, y_test, drug_id_list, var, perturbations, pool_train_perturbations=False):
    """
    Evaluate the two non-learned baselines on the test cells, separately for each perturbation:
      * mean baseline:       every test cell is predicted as the mean perturbed training cell
      * mean-shift baseline: test control cell + (mean perturbed - mean control) of the training cells

    Args:
        train_dataset: The training dataset (TahoeMultiDataset with .x, .y, .drug_ids).
        X_test: Control cells paired with the test cells, shape (n_cells, n_genes).
        y_test: True perturbed test cells, shape (n_cells, n_genes).
        drug_id_list: Index into `perturbations` for every test cell, shape (n_cells, 1).
        var: adata.var (gene annotations), needed for the DE test.
        perturbations: Names of the test perturbations.
        pool_train_perturbations: If True, the baseline for every test perturbation is
            computed from all training perturbations pooled together, rather than from
            the training samples matching that perturbation's drug ID. Use this for
            unseen-perturbation evaluation, where the test drug IDs have no matching
            entries in train_dataset (there's no drug-specific training mean to fall
            back on since the perturbation was never trained on).

    Returns:
        Two dicts (mean baseline, mean-shift baseline) mapping each metric to a list with one
        value per test perturbation.
    """
    # mean_baselines = {}
    # mean_shift_baselines = {}
    results_mean = {"DE_F1": [], "DE_Precision": [], "DE_Recall": [], "Energy_Loss": [], "MSE": [], "DE_F1_NUM_True": [], "DE_F1_NUM_Pred": []}
    results_mean_shift = {"DE_F1": [], "DE_Precision": [], "DE_Recall": [], "Energy_Loss": [], "MSE": [], "DE_F1_NUM_True": [], "DE_F1_NUM_Pred": []}

    if pool_train_perturbations:
        pooled_mean_baseline = train_dataset.y.mean(axis=0)
        pooled_mean_shift = pooled_mean_baseline - train_dataset.x.mean(axis=0)

    for c in np.unique(drug_id_list):
        idx = drug_id_list == c
        idx = idx.squeeze()
        perturbation = perturbations[int(c.item())]

        if pool_train_perturbations:
            mean_baseline = pooled_mean_baseline
            mean_shift = pooled_mean_shift
        else:
            train_perturbed_idx = train_dataset.drug_ids == c
            train_perturbed_idx = train_perturbed_idx.squeeze()

            # Compute the mean baseline prediction for the current perturbation
            mean_baseline = train_dataset.y[train_perturbed_idx].mean(axis=0)
            # Compute the mean shift baseline prediction for the current perturbation
            mean_shift = mean_baseline - train_dataset.x.mean(axis=0)

        n_test = X_test[idx].shape[0]
        mean_baseline_pred = np.tile(mean_baseline, (n_test, 1))
        # mean_baselines[perturbation] = mean_baseline_pred

        mean_shift_baseline_pred = X_test[idx] + mean_shift
        # mean_shift_baselines[perturbation] = mean_shift_baseline_pred

        # DE-F1
        # des_loss returns (F1, precision, recall). (The numbers of true/predicted DE genes, used for an
        # additional analysis, can be re-enabled at the end of des_loss in utils/metrics_utils.py.)
        mean_de_f1_score, mean_de_precision, mean_de_recall = des_loss(mean_baseline_pred, y_test[idx], X_test[idx], var, testing=False)
        mean_shift_de_f1_score, mean_shift_de_precision, mean_shift_de_recall = des_loss(mean_shift_baseline_pred, y_test[idx], X_test[idx], var, testing=False)

        # E-distance
        # Compute energy loss on a subset of the data to avoid online backend in geomloss (we want tensorized backend)
        if len(mean_baseline_pred) > 5000:
            mean_baseline_energy_losses = []
            mean_shift_baseline_energy_losses = []
            for i in range(0, len(mean_baseline_pred), 5000):
                mean_baseline_energy_losses.append(energy_loss(mean_baseline_pred[i:i+5000], y_test[idx][i:i+5000]))
                mean_shift_baseline_energy_losses.append(energy_loss(mean_shift_baseline_pred[i:i+5000], y_test[idx][i:i+5000]))
            mean_baseline_energy = np.mean(mean_baseline_energy_losses)
            mean_shift_baseline_energy = np.mean(mean_shift_baseline_energy_losses)
        else:
            mean_baseline_energy = energy_loss(mean_baseline_pred, y_test[idx])
            mean_shift_baseline_energy = energy_loss(mean_shift_baseline_pred, y_test[idx])

        # MSE
        mean_baseline_mse = F.mse_loss(torch.tensor(mean_baseline_pred), torch.tensor(y_test[idx])).item()
        mean_shift_baseline_mse = F.mse_loss(torch.tensor(mean_shift_baseline_pred), torch.tensor(y_test[idx])).item()
        
        results_mean["DE_F1"].append(mean_de_f1_score)
        results_mean["DE_Precision"].append(mean_de_precision)
        results_mean["DE_Recall"].append(mean_de_recall)
        results_mean["Energy_Loss"].append(mean_baseline_energy)
        results_mean["MSE"].append(mean_baseline_mse)

        ###############################
        # results_mean["DE_F1_NUM_True"].append(mean_num_true_de_genes)
        # results_mean["DE_F1_NUM_Pred"].append(mean_num_pred_de_genes)

        # results_mean_shift["DE_F1_NUM_True"].append(mean_shift_num_true_de_genes)
        # results_mean_shift["DE_F1_NUM_Pred"].append(mean_shift_num_pred_de_genes)
        ###############################

        results_mean_shift["DE_F1"].append(mean_shift_de_f1_score)
        results_mean_shift["DE_Precision"].append(mean_shift_de_precision)
        results_mean_shift["DE_Recall"].append(mean_shift_de_recall)
        results_mean_shift["Energy_Loss"].append(mean_shift_baseline_energy)
        results_mean_shift["MSE"].append(mean_shift_baseline_mse)

        # results_mean[perturbation] = {
        #     "DE_F1": mean_de_f1_score,
        #     "DE_Precision": mean_de_precision,
        #     "DE_Recall": mean_de_recall,
        #     "Energy_Loss": mean_baseline_energy,
        #     "MSE": mean_baseline_mse
        # }

        # results_mean_shift[perturbation] = {
        #     "DE_F1": mean_shift_de_f1_score,
        #     "DE_Precision": mean_shift_de_precision,
        #     "DE_Recall": mean_shift_de_recall,
        #     "Energy_Loss": mean_shift_baseline_energy,
        #     "MSE": mean_shift_baseline_mse
        # }

    return results_mean, results_mean_shift