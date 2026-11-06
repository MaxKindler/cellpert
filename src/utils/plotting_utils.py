import os
import scanpy as sc
import anndata as ad
import numpy as np
import matplotlib.pyplot as plt
from .helper_utils import combine_to_adata


def plot_violin_plots(test_outputs_list, y_test_list, adata_var, genes_to_show, perturbation_name, run_name, architecture="RectifiedFlow"):
    adata_plotting = combine_to_adata(test_outputs_list, y_test_list, adata_var)
    # Rename obs 'type' values to 'predicted' and 'true'
    adata_plotting.obs['type'] = adata_plotting.obs['type'].replace({'perturbed': 'Predicted', 'control': 'True'})
    axes = sc.pl.violin(adata_plotting, keys=genes_to_show, groupby='type', rotation=None, show=False)
    for ax in axes:
        ax.tick_params(axis='x', labelsize=24)
        ax.tick_params(axis='y', labelsize=22)
        ax.set_ylabel(ax.get_ylabel(), fontsize=24)
        ax.set_xlabel(None)
    # v.set_title(f"Violin Plots of top genes for {perturbation_name}", fontsize=24)
    os.makedirs(f'../plots/violin/{run_name}', exist_ok=True)
    plt.savefig(f'../plots/violin/{run_name}/{architecture}_{perturbation_name}_violin_plot.png', bbox_inches='tight', dpi=300)
    plt.close()


def plot_mean_bars(test_outputs_list, y_test_list, adata_var, top_genes):
    adata_plotting = combine_to_adata(test_outputs_list, y_test_list, adata_var)

    adata_plotting_means = adata_plotting.to_df().groupby(adata_plotting.obs['type']).mean()

    # filter for top genes
    adata_plotting_means = adata_plotting_means[top_genes]
    adata_plotting_means.transpose().plot(kind='bar', figsize=(20, 10))
    plt.title("Mean Gene Expression of top 10 genes for Predicted vs True")
    plt.xlabel("Genes")
    plt.ylabel("Mean Expression")
    plt.xticks(rotation=90)
    plt.show()

    # sc.pl.bar(adata_plotting, keys=top_genes, groupby='type', rotation=90, show=True)
