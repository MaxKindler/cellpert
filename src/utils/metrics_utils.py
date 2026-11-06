import anndata as ad
import scanpy as sc
import pandas as pd
import torch
import torch.nn.functional as F
from .helper_utils import combine_to_adata
from geomloss import SamplesLoss

def get_top_genes(adata, drugs, top_n=10):
    # filter adata for relevant groups
    mask = (adata.obs['drug'] == 'DMSO_TF')
    for drug in drugs:
        mask = mask | (adata.obs['drug'] == drug)

    adata = adata[mask]
    sc.tl.rank_genes_groups(adata, groupby="drug", reference="DMSO_TF", method="wilcoxon")
    result = sc.get.rank_genes_groups_df(adata, group=drugs[0])
    top_genes = result.sort_values('pvals_adj').head(top_n)['names'].tolist()
    return top_genes


# Calculate DES loss
# We want to measure the overlap between the top DE genes in the predicted vs. true data compared to the control cells
def des_loss(predicted, true_perturbed, controls, adata_var, groupby='type', group1='perturbed', group2='control', pval_threshold=0.05, lfc_threshold=0.25, testing=False):
    # First calculate ground truth DE 
    adata_true = combine_to_adata(true_perturbed, controls, adata_var)
    sc.tl.rank_genes_groups(adata_true, groupby=groupby, groups=[group1], reference=group2, method='wilcoxon')
    true_de = sc.get.rank_genes_groups_df(adata_true, group='perturbed')
    true_de_genes = set(true_de[
        (true_de['pvals_adj'] < pval_threshold) & 
        (true_de['logfoldchanges'].abs() > lfc_threshold)
    ]['names'])

    # Next calculate DE for predicted vs. control
    adata_pred = combine_to_adata(predicted, controls, adata_var)
    sc.tl.rank_genes_groups(adata_pred, groupby=groupby, groups=[group1], reference=group2, method='wilcoxon')
    pred_de = sc.get.rank_genes_groups_df(adata_pred, group='perturbed')
    pred_de_genes = set(pred_de[
        (pred_de['pvals_adj'] < pval_threshold) & 
        (pred_de['logfoldchanges'].abs() > lfc_threshold)
    ]['names'])

    # Compuate overlap
    if len(true_de_genes) == 0 and len(pred_de_genes) == 0:
        return 1.0, 1.0, 1.0 # Perfect score if no DE genes in either
    
    # Compare fold changes (Not used)
    comparison = true_de.merge(pred_de, on='names', suffixes=('_true', '_pred'))
    correlation = comparison['logfoldchanges_true'].corr(comparison['logfoldchanges_pred'])

    # print(f"Variance of true perturbed: {true_perturbed.var(axis=0).mean():.4f}")
    # print(f"Variance of predicted: {predicted.var(axis=0).mean():.4f}")
    # print(f"Variance of control: {controls.var(axis=0).mean():.4f}")

    tp = len(true_de_genes & pred_de_genes)
    fp = len(pred_de_genes - true_de_genes)
    fn = len(true_de_genes - pred_de_genes)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return f1_score, precision, recall


def energy_loss(predicted_list, true_list):
    loss_fn = SamplesLoss(loss="energy")
    predicted_tensor = torch.tensor(predicted_list, dtype=torch.float32)
    true_tensor = torch.tensor(true_list, dtype=torch.float32)
    loss = loss_fn(predicted_tensor, true_tensor)
    return loss.item()

def combined_loss(predicted, true_perturbed, alpha=0.5):
    mse = F.mse_loss(predicted, true_perturbed)
    energy = SamplesLoss(loss="energy")(predicted, true_perturbed)
    return alpha * mse + (1 - alpha) * energy

class CombinedLoss(torch.nn.Module):
    """Combined Sinkhorn + Energy loss."""

    def __init__(self, sinkhorn_weight=0.001, energy_weight=1.0, mse_weight=1.0, blur=0.05):
        super().__init__()
        self.sinkhorn_weight = sinkhorn_weight
        self.energy_weight = energy_weight
        self.mse_weight = mse_weight

        self.sinkhorn_loss = SamplesLoss(loss="sinkhorn", blur=blur)
        self.energy_loss = SamplesLoss(loss="energy", blur=blur)

    def forward(self, pred, target):
        sinkhorn_val = self.sinkhorn_loss(pred, target)
        energy_val = self.energy_loss(pred, target)
        mse_val = F.mse_loss(pred, target)
        return self.sinkhorn_weight * sinkhorn_val + self.energy_weight * energy_val + self.mse_weight * mse_val