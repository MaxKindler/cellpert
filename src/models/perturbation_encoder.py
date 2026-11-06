import torch.nn as nn


class PerturbationEncoder(nn.Module):
    """
    Maps a perturbation ID to an embedding via a fixed Morgan-fingerprint lookup
    followed by a learned projection
    """

    def __init__(self, fingerprints, embed_dim):
        super().__init__()
        # Not a Parameter: fingerprints are fixed, structure-derived features,
        # not something gradient descent should be adjusting.
        self.register_buffer("fingerprints", fingerprints)
        self.proj = nn.Linear(fingerprints.shape[1], embed_dim)

    def forward(self, drug_id):
        drug_id = drug_id.long().view(-1)
        fps = self.fingerprints[drug_id]
        return self.proj(fps)
