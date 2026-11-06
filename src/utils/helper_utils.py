import anndata as ad
import numpy as np

def combine_to_adata(perturbed, control, adata_var):
    perturbed_adata = ad.AnnData(X=perturbed)
    perturbed_adata.obs['type'] = 'perturbed'
    control_adata = ad.AnnData(X=control)
    control_adata.obs['type'] = 'control'

    obs_output = np.arange(len(perturbed_adata))
    obs_y = np.arange(len(perturbed_adata), len(perturbed_adata)+len(control_adata))

    perturbed_adata.obs_names = obs_output
    control_adata.obs_names = obs_y

    adata_combined = ad.concat([perturbed_adata, control_adata], axis=0)
    adata_combined.var = adata_var

    return adata_combined