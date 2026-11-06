import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator


def compute_morgan_fingerprints(drug_names, radius=2, n_bits=1024):
    """
    Build a (len(drug_names), n_bits) tensor of Morgan (ECFP) fingerprint bits,
    one row per drug name, in the given order.

    Unlike a learned nn.Embedding row, a fingerprint is a fixed, structure-derived
    representation: it is meaningful even for a drug that never appeared in
    training, since it's computed from that drug's own molecular graph rather
    than fit from label data. That's what lets a projection trained only on the
    training perturbations' fingerprints generalize to unseen perturbations.
    """
    # Imported lazily: configs -> architectures -> models -> utils (this package)
    # would otherwise be a circular import at module load time.
    from configs.drug_smiles import DRUG_SMILES

    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)

    fps = torch.zeros(len(drug_names), n_bits, dtype=torch.float32)
    for i, name in enumerate(drug_names):
        smiles = DRUG_SMILES[name]
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Could not parse SMILES for drug '{name}': {smiles}")
        bit_vect = generator.GetFingerprint(mol)
        fps[i] = torch.from_numpy(np.array(bit_vect, dtype=np.float32))
    return fps
