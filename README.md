# Predicting Single-Cell Perturbation Responses: A Comparison of Transformer and Flow-Based Approaches

Code, data instructions and results of the master's thesis: Predicting Single-Cell Perturbation Responses: A Comparison of Transformer and Flow-Based Approaches (Max Kindler).

The task is to predict how cancer cells change their gene expression after drug treatment.
Given unperturbed (control) cells and a drug, the models generate the transcriptomes of the
treated cells. Three models are compared with two non-learned baselines:

| Name in code (`--architecture`) | Model | Idea |
|---|---|---|
| `rectified_flow` | **Rectified Flow** | Learns an ODE (an MLP velocity field) that transports control cells to perturbed cells |
| `transformer_cellattention` | **Transformer** | Predicts perturbed cells directly; self-attention runs across the cells of a batch, trained with the energy distance |
| `transformer_flow` | **Transformer Flow** | Rectified flow whose velocity field is the cell-attention Transformer |
| (`report_baselines_results.py`) | **Mean / mean-shift baselines** | Mean perturbed training cell / control cell + mean perturbation effect |

Drugs are represented by Morgan fingerprints of their molecular structure. This lets the models also make predictions for drugs never seen during training.

Two evaluation tasks are used:

* **SEEN**: the 10 training drugs are tested on held-out cells (80/10/10 split by cell).
* **UNSEEN-PERT**: the models are tested on 3 drugs (`RMC-6236`, `4EGI-1`, `AT7519`) that are completely excluded from training and validation.
Metrics, computed per drug and then averaged: **DE-F1 / precision / recall** (overlap of the differentially expressed genes of predicted vs. true cells, each compared to control cells),
**energy distance** (distance between the predicted and true cell distributions) and **MSE**. Every result is the mean ± std over **20 seeds**.

---

## Contents

1. [Repository structure](#1-repository-structure)
2. [Setup](#2-setup)
3. [Data](#3-data)
4. [Reproducing the results](#4-reproducing-the-results)

---

## 1. Repository structure

```
cellpert/
├── README.md
├── requirements.txt          # exact pinned versions of all Python packages
├── environment.yml           # conda environment (Python 3.11 + requirements.txt)
├── data/                     # data sets (not in git, created by src/create_dataset.py)
├── models/                   # model checkpoints (not in git, created during training)
├── src/                      # all source code - run scripts from this directory
│   ├── create_dataset.py     # download + preprocess Tahoe-100M  -> data/
│   ├── run.py                # train + evaluate a model (20 seeds, sweeps, reflow)
│   ├── report_baselines_results.py   # evaluate the mean / mean-shift baselines
│   ├── configs/              # final hyperparameters, sweep definitions, drug SMILES
│   ├── datasets/             # PyTorch datasets (control/perturbed pairing) and batch sampler
│   ├── models/               # model architectures and baselines
│   ├── training/             # training / validation loops
│   ├── utils/                # data splits, metrics, evaluation, plotting, wandb helpers
│   └── analysis/dataset_analysis.ipynb   # exploratory analysis of the data set
└── plots/                    # notebooks producing the thesis figures and statistics
    ├── data/results/final/   # result tables of all thesis runs (exported from wandb)
    ├── data/sweep/           # results of the hyperparameter sweeps
    └── thesis_plots/ ...     # the generated figures
```

## 2. Setup

**Requirements**

* Linux with an NVIDIA GPU. The experiments were run on a server with NVIDIA RTX 2080 Ti (11 GB)
  and Quadro GV100 (32 GB) GPUs. The flow models also run on the CPU, but very slowly.
* About 45 GB of free disk space if you build the data set from scratch, or 0.5 GB for the final
  data set alone.
* At least 64 GB of RAM for the data preprocessing. Training needs about 16 GB.
* Python 3.11 (the thesis used 3.11.14).

**conda**

```bash
git clone <repository-url> cellpert
cd cellpert
conda env create -f environment.yml
conda activate cellpert
```

`requirements.txt` pins the exact versions of the environment used for the thesis, including
PyTorch 2.10.0 with CUDA 12.8 (installed automatically as pip wheels). 

Check the installation:

```bash
cd src
python -c "import torch, scanpy, geomloss, rdkit, wandb; print(torch.__version__, torch.cuda.is_available())"
```

**Weights & Biases (wandb)**

All training metrics are logged to [wandb](https://wandb.ai):

```bash
wandb login                       # once; uses your own account
export WANDB_PROJECT=cellpert        # optional (default: cellpert)
export WANDB_ENTITY=<your-team>      # optional (default: your personal account)
```

If you don't want to use a wandb account, set `export WANDB_MODE=offline`. Runs are then stored
locally in `src/wandb/` and can be uploaded later with `wandb sync`. To switch logging off
completely, use `export WANDB_MODE=disabled`.

## 3. Data

The thesis uses the **Tahoe-100M** single-cell drug perturbation atlas
([Zhang et al., 2025](https://doi.org/10.1016/j.cell.2026.08.035)), plate 2, restricted to the cell line **SW480**. The data are publicly available from the
[Arc Institute Virtual Cell Atlas](https://github.com/ArcInstitute/arc-virtual-cell-atlas/tree/main/tahoe-100M).
No account is needed.

### 3.1 Create the data set (one command)

```bash
cd src
python create_dataset.py --download
```

The script runs these steps, skipping any step whose output already exists:

| Step | Output (in `data/`) | Size | What happens |
|---|---|---|---|
| 0. download | `tahoe100M_plate2.h5ad` | 29 GB | raw counts of Tahoe-100M plate 2 |
| 1. subset + normalise | `tahoe100M_SW480.h5ad` | 12 GB | keep SW480 cells |
| 2. HVG selection | **`tahoe100M_SW480_filtered_1024.h5ad`** | 0.5 GB | 1024 most highly variable genes |

Only the last file is used for training and evaluation. You can delete the two larger files
afterwards.

### 3.2 What is used from the data

* Control cells: `adata.obs["drug"] == "DMSO_TF"`
* Training drugs (IDs 0–9): BI-3406, Capmatinib, Paclitaxel, Oxaliplatin, Gemcitabine,
  Vemurafenib, Anastrozole, Minodronic acid, Everolimus, Adagrasib
* Unseen drugs (IDs 10–12, UNSEEN-PERT task only): RMC-6236, 4EGI-1, AT7519
* Drug structures (SMILES) for the fingerprints are stored in `src/configs/drug_smiles.py`,
  so no network access is needed during training.
* Splits (`src/utils/data_utils.py`): control and perturbed cells are split independently
  with `sklearn.model_selection.train_test_split(random_state=seed)`.
  * SEEN: 80/10/10 train/val/test
  * UNSEEN-PERT: training drugs 80/20 train/val; all cells of the unseen drugs form the test
    set; control cells 80/10/10
* Each perturbed cell is paired with a randomly drawn control cell from the same split.

## 4. Reproducing the results

All commands are run from `src/`. Each call trains the model for the **20 seeds** of the thesis
one after another. For each seed it:

1. trains for 100 epochs and keeps the checkpoint with the best validation DE-F1;
2. evaluates that checkpoint on the test set, together with both baselines on the same test
   cells;
3. logs everything to wandb (`test_loss`, `de_test_score`, `de_precision_test_score`,
   `de_recall_test_score`, `energy_test_score`, `mse_test_loss`, the per-drug lists `*_all`,
   and `mean_baseline_*` / `mean_shift_baseline_*`);
4. saves violin plots of the top DE genes to `plots/violin/<run name>/`.

Use `--cuda_device <i>` to select a GPU and `--seeds 123` to run a single seed.

### 4.1 Main results (SEEN and UNSEEN-PERT)

| Model | Command | Time / seed |
|---|---|---|
| Rectified Flow | `python run.py --architecture rectified_flow --task SEEN` | ~2 h |
| | `python run.py --architecture rectified_flow --task UNSEEN-PERT` | ~1.75 h |
| Transformer | `python run.py --architecture transformer_cellattention --task SEEN` | ~1.5 h |
| | `python run.py --architecture transformer_cellattention --task UNSEEN-PERT` | ~1.5 h |
| Transformer Flow | `python run.py --architecture transformer_flow --task SEEN` | ~2.5 h |
| | `python run.py --architecture transformer_flow --task UNSEEN-PERT` | ~3.3 h |

Times are median wandb run times of the thesis runs. The thesis results of these runs are in
`plots/data/results/final/wandb_{rectified_flow,transformer,transformerflow}_seen_unseen.csv`.

### 4.2 Reflow experiment (Rectified Flow, UNSEEN-PERT)

```bash
python run.py --architecture rectified_flow --task UNSEEN-PERT --reflow 2
```

After the normal training and test evaluation (logged as above), the model generates
(control → predicted) pairs for the training cells. A new model is then trained on these fixed
pairs ("reflow"), and this is repeated twice. Each round is tested and logged with the prefix
`reflow-1_` / `reflow-2_`, e.g. `reflow-2_de_test_score`. Thesis results:
`plots/data/results/final/wandb_rectifiedflow_reflow.csv`. Time: about 1.8 h per seed.

### 4.3 Baselines

Mean and mean-shift baselines (no GPU or wandb needed, a few minutes per seed and task):

```bash
python report_baselines_results.py --task SEEN
python report_baselines_results.py --task UNSEEN-PERT
```

Both commands append to `plots/data/baseline_results_reproduced.csv` (change it with
`--output`). Thesis results: `plots/data/results/final/baseline_results.csv`.

### 4.4 Hyperparameter sweeps

The final hyperparameters in `src/configs/*_config.py` were selected with wandb Bayesian
sweeps on the SEEN task (metric: validation DE-F1):

```bash
python run.py --architecture rectified_flow --task SEEN --run_sweep
python run.py --architecture transformer_cellattention --task SEEN --run_sweep
python run.py --architecture transformer_flow --task SEEN --run_sweep
# add more agents to an existing sweep:
python run.py --architecture rectified_flow --task SEEN --run_sweep --sweep_id <sweep id>
```

The search spaces are in `src/configs/*_sweep_config.py`. The sweep results of the thesis are in
`plots/data/sweep/`. Because the Bayesian search itself is random, a new sweep will not repeat
exactly the same trials.

### 4.5 From wandb to the result tables

The notebooks read the wandb run tables as CSV files. To build them from your own runs, export
the runs table in the wandb web UI (*Runs* table → *Export as CSV*)