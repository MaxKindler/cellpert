# Canonical SMILES for every perturbation used in PERTURBATIONS / UNSEEN_PERTURBATIONS
# across the config files. Sourced from the Tahoe-100M drug_metadata table
# (https://huggingface.co/datasets/tahoebio/Tahoe-100M, metadata/drug_metadata.parquet)
# and cached here so fingerprint computation doesn't need network access at train time.
DRUG_SMILES = {
    "BI-3406": "CC1=NC2=CC(=C(C=C2C(=N1)NC(C)C3=CC(=CC(=C3)N)C(F)(F)F)OC4CCOC4)OC",
    "Capmatinib": "CNC(=O)C1=C(C=C(C=C1)C2=NN3C(=CN=C3N=C2)CC4=CC5=C(C=C4)N=CC=C5)F",
    "Paclitaxel": "CC1=C2C(C(=O)C3(C(CC4C(C3C(C(C2(C)C)(CC1OC(=O)C(C(C5=CC=CC=C5)NC(=O)C6=CC=CC=C6)O)O)OC(=O)C7=CC=CC=C7)(CO4)OC(=O)C)O)C)OC(=O)C",
    "Oxaliplatin": "C1CCC(C(C1)[NH-])[NH-].C(=O)(C(=O)O)O.[Pt+2]",
    "Gemcitabine": "C1=CN(C(=O)N=C1N)C2C(C(C(O2)CO)O)(F)F",
    "Vemurafenib": "CCCS(=O)(=O)NC1=C(C(=C(C=C1)F)C(=O)C2=CNC3=C2C=C(C=N3)C4=CC=C(C=C4)Cl)F",
    "Anastrozole": "CC(C)(C#N)C1=CC(=CC(=C1)CN2C=NC=N2)C(C)(C)C#N",
    "Minodronic acid": "C1=CC2=NC=C(N2C=C1)CC(O)(P(=O)(O)O)P(=O)(O)O",
    "Everolimus": "CC1CCC2CC(C(=CC=CC=CC(CC(C(=O)C(C(C(=CC(C(=O)CC(OC(=O)C3CCCCN3C(=O)C(=O)C1(O2)O)C(C)CC4CCC(C(C4)OC)OCCO)C)C)O)OC)C)C)C)OC",
    "Adagrasib": "CN1CCCC1COC2=NC3=C(CCN(C3)C4=CC=CC5=C4C(=CC=C5)Cl)C(=N2)N6CCN(C(C6)CC#N)C(=O)C(=C)F",
    "RMC-6236": "CCN1C2=C3C=C(C=C2)C4=CSC(=N4)CC(C(=O)N5CCCC(N5)C(=O)OCC(CC3=C1C6=C(N=CC(=C6)N7CCN(CC7)C)C(C)OC)(C)C)NC(=O)C8CC8C",
    "4EGI-1": "C1=CC=C(C(=C1)CC(=NNC2=NC(=CS2)C3=CC(=C(C=C3)Cl)Cl)C(=O)O)[N+](=O)[O-]",
    "AT7519": "C1CNCCC1NC(=O)C2=C(C=NN2)NC(=O)C3=C(C=CC=C3Cl)Cl",
}
