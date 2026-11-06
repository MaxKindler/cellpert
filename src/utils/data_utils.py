import scanpy as sc
import scipy.sparse as sp
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from datasets import TahoeMultiDataset


# Load dataset and filter for highly variable genes
def load_adata(dataset_name, num_genes):
    adata = sc.read_h5ad(f"../data/{dataset_name}.h5ad")
    sc.pp.highly_variable_genes(adata, n_top_genes=num_genes)
    adata = adata[:, adata.var['highly_variable']].copy()
    
    # # Add pseudocount to avoid log(0) warnings in rank_genes_groups
    # if sp.issparse(adata.X):
    #     adata.X = sp.csr_matrix(adata.X.toarray() + 1e-10)
    # else:
    #     adata.X = adata.X + 1e-10

    # Save the filtered AnnData object for future use
    adata.write_h5ad(f"../data/{dataset_name}_filtered_{num_genes}.h5ad")
    
    return adata

def process_adata(adata, perturbations, seed):
    """
    Filter the AnnData object to include only the specified perturbations.

    Args:
        adata: The AnnData object containing the dataset.
        perturbations: A list of perturbation names to filter by.

    Returns:
        A filtered AnnData object containing only the specified perturbations.
    """

    x_adata = adata[adata.obs['drug'] == 'DMSO_TF'].X.toarray()
    y_adata = []
    drug_ids_all = []
    for i, drug in enumerate(perturbations):
        y_mask = adata.obs['drug'] == drug
        y_drug = adata[y_mask].X.toarray()
        
        y_adata.append(y_drug)
        drug_ids_all.append(torch.ones(len(y_drug), 1, dtype=torch.int) * i)

    y_adata = np.concatenate(y_adata, axis=0)
    drug_ids_all = torch.cat(drug_ids_all, dim=0)

    # 2. Split X (DMSO) and Y (Drugs) independently to prevent leakage
    # Note: X and Y aren't paired 1:1, so we split their pools separately.
    x_train, x_temp = train_test_split(x_adata, test_size=0.2, random_state=seed)
    x_val, x_test = train_test_split(x_temp, test_size=0.5, random_state=seed)

    y_train, y_temp, id_train, id_temp = train_test_split(y_adata, drug_ids_all, test_size=0.2, random_state=seed)
    y_val, y_test, id_val, id_test = train_test_split(y_temp, id_temp, test_size=0.5, random_state=seed)

    # 3. Instantiate the isolated datasets
    train_dataset = TahoeMultiDataset(x_train, y_train, id_train, seed)
    val_dataset = TahoeMultiDataset(x_val, y_val, id_val, seed)
    test_dataset = TahoeMultiDataset(x_test, y_test, id_test, seed)

    return train_dataset, val_dataset, test_dataset


def process_adata_unseen_pert(adata, train_perturbations, test_perturbations, seed):
    """
    Filter the AnnData object to include only the specified perturbations.

    Args:
        adata: The AnnData object containing the dataset.
        train_perturbations: A list of perturbation names to filter for training.
        test_perturbations: A list of perturbation names to filter for testing.
        seed: Random seed for reproducibility.

    Returns:
        A filtered AnnData object containing only the specified perturbations.
    """

    x_adata = adata[adata.obs['drug'] == 'DMSO_TF'].X.toarray()

    # Train perturbations
    y_adata = []
    drug_ids_all = []
    for i, drug in enumerate(train_perturbations):
        y_mask = adata.obs['drug'] == drug
        y_drug = adata[y_mask].X.toarray()
        
        y_adata.append(y_drug)
        drug_ids_all.append(torch.ones(len(y_drug), 1, dtype=torch.int) * i)

    y_adata = np.concatenate(y_adata, axis=0)
    drug_ids_all = torch.cat(drug_ids_all, dim=0)

    # Test perturbations
    y_test_adata = []
    test_drug_ids_all = []
    for i, drug in enumerate(test_perturbations):
        y_mask = adata.obs['drug'] == drug
        y_drug = adata[y_mask].X.toarray()
        
        y_test_adata.append(y_drug)
        drug_id = i + len(train_perturbations)  # Offset test drug IDs to avoid overlap with train IDs
        test_drug_ids_all.append(torch.ones(len(y_drug), 1, dtype=torch.int) * drug_id)

    y_test_adata = np.concatenate(y_test_adata, axis=0)
    test_drug_ids_all = torch.cat(test_drug_ids_all, dim=0)

    # 2. Split X (DMSO) and Y (Drugs) independently to prevent leakage
    # Note: X and Y aren't paired 1:1, so we split their pools separately.
    x_train, x_temp = train_test_split(x_adata, test_size=0.2, random_state=seed)
    x_val, x_test = train_test_split(x_temp, test_size=0.5, random_state=seed)

    y_train, y_val, id_train, id_val = train_test_split(y_adata, drug_ids_all, test_size=0.2, random_state=seed)
    y_test, id_test = y_test_adata, test_drug_ids_all  # Using all test perturbations

    # 3. Instantiate the isolated datasets
    train_dataset = TahoeMultiDataset(x_train, y_train, id_train, seed)
    val_dataset = TahoeMultiDataset(x_val, y_val, id_val, seed)
    test_dataset = TahoeMultiDataset(x_test, y_test, id_test, seed)

    # print(f"Unique drug IDs in training set: {torch.unique(drug_ids_all)}")
    # print(f"Unique drug IDs in validation set: {torch.unique(id_val)}")
    # print(f"Unique drug IDs in test set: {torch.unique(id_test)}")

    return train_dataset, val_dataset, test_dataset