# PyTorch datasets and sampler.
#
# Control (DMSO) and perturbed cells are different cells (scRNA-seq is destructive), so there is no
# true pairing: every perturbed cell is paired with a control cell drawn at random on each access
# (TahoeMultiDataset.__getitem__). The datasets re-seed Python's and NumPy's global RNGs when they
# are created, so the pairing is reproducible for a given seed.
#
# Note: PerPerturbationSampler reads the drug IDs via dataset[idx], which also draws random control
# cells. This is kept as is because changing it would change the random numbers used later and
# therefore the results.

import torch
from torch.utils.data import Dataset
import random
import numpy as np



class TahoeMultiDataset(Dataset):
    def __init__(self, x, y, drug_ids, seed):
        self.x = x
        self.y = y
        self.drug_ids = drug_ids
        np.random.seed(seed)
        random.seed(seed)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        random_idx = random.randint(0, len(self.x) - 1)
        input_gene_expression = self.x[random_idx]
        output_gene_expression = self.y[idx]
        drug_id = self.drug_ids[idx]
        return torch.tensor(input_gene_expression, dtype=torch.float32), output_gene_expression, drug_id



class TahoeReflowDataset(Dataset):
    def __init__(self, x, y, context, seed):
        self.x = x
        self.y = y
        self.context = context
        np.random.seed(seed)
        random.seed(seed)

    def __len__(self):
        return min(len(self.x), len(self.y))

    def __getitem__(self, idx):
        input_gene_expression = self.x[idx]
        output_gene_expression = self.y[idx]
        context = self.context[idx]
        return torch.tensor(input_gene_expression, dtype=torch.float32), torch.tensor(output_gene_expression, dtype=torch.float32), torch.tensor(context, dtype=torch.float32)
    

class PerPerturbationSampler:
    def __init__(self, dataset, batch_size):
        self.batch_size = batch_size
        self.perturbation_indices = {}
        
        for idx in range(len(dataset)):
            _, _, drug_id = dataset[idx]
            drug_id = drug_id.item()
            if drug_id not in self.perturbation_indices:
                self.perturbation_indices[drug_id] = []
            self.perturbation_indices[drug_id].append(idx)
    
    def get_batches(self):
        """Yields batches where all cells share the same perturbation"""
        all_batches = []
        for drug_id, indices in self.perturbation_indices.items():
            random.shuffle(indices)
            for i in range(0, len(indices), self.batch_size):
                batch_indices = indices[i:i + self.batch_size]
                if len(batch_indices) >= 32:  # minimum batch size
                    all_batches.append(batch_indices)
        
        random.shuffle(all_batches)  # shuffle batch order
        return all_batches
