"""
Create the processed data set used in the thesis from the public Tahoe-100M data.

Run from the `src/` directory:

    cd src
    python create_dataset.py --download

Steps (each is skipped if its output file already exists):
  0. (--download) Download Tahoe-100M plate 2 (~29 GB) to ../data/tahoe100M_plate2.h5ad
  1. Keep only SW480 cells, store raw counts in layers["counts"], normalise to 1e4 counts
     per cell and log1p-transform           -> ../data/tahoe100M_SW480.h5ad          (~12 GB)
  2. Keep the 1024 most highly variable genes (scanpy, flavor "seurat")
                                            -> ../data/tahoe100M_SW480_filtered_1024.h5ad (~0.5 GB)
The file from step 2 is the only one needed for training and evaluation.
"""
import os
import urllib.request

import torch
import numpy as np
import scanpy as sc
import argparse
import pathlib
from pathlib import Path
import os
import sys


# Public download link of Tahoe-100M plate 2 (Arc Institute Virtual Cell Atlas, Google Cloud bucket
# gs://arc-institute-virtual-cell-atlas). Size: 29,037,152,127 bytes, MD5 (base64): usxviuqGbuw0RYnECCVCWw==
PLATE2_URL = "https://storage.googleapis.com/arc-institute-virtual-cell-atlas/tahoe100M/2025-02-25/h5ad/plate2_filt_Vevo_Tahoe100M_WServicesFrom_ParseGigalab.h5ad"


def download_plate(target_path, url=PLATE2_URL):
    """Download the raw Tahoe-100M plate file (skipped if it already exists)."""
    if os.path.exists(target_path):
        print(f"{target_path} already exists, skipping download.")
        return
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    print(f"Downloading {url}\n  -> {target_path} (~29 GB, this can take a while)")
    urllib.request.urlretrieve(url, target_path + ".part")
    os.rename(target_path + ".part", target_path)


def create_dataset(adata_train_path:str, target_path:str, num_cells_per_perturbation:int=10_000, cell_line:str="SW480", highly_variable:bool=False):
    """
    Subset the raw plate to one cell line, normalise and log-transform it.

    adata_train_path: str: Path of the raw Tahoe-100M plate file
    target_path: str: Path where the processed data set is saved
    num_cells_per_perturbation: int: Not used (all cells of the cell line are kept)
    cell_line: str: Cell line to keep (obs["cell_name"]); the thesis uses SW480
    highly_variable: bool: Directly select 10,000 HVGs (not used in the thesis; the 1024 HVGs
        are selected in a second step, see main())
    """
    print("Read training data...")
    adata = sc.read_h5ad(adata_train_path, backed="r")


    if cell_line is not None:
        print(f"Subset to cell line {cell_line}...")
        adata = adata[adata.obs['cell_name'] == cell_line, :]

    adata = adata.to_memory()
    adata.layers["counts"] = adata.X.copy()

    # normalize total so that cells are comparable
    print("Normalize total counts...")
    sc.pp.normalize_total(adata, target_sum=1e4)


    # if log_normalized:
    print("Log normalize training data...")
    sc.pp.log1p(adata)

    if highly_variable:
        print("Select highly variable genes...")
        sc.pp.highly_variable_genes(adata, n_top_genes=10_000)#, batch_key="batch", ) # 4096 is too much for the toy dataset, because the decoder is not trained yet and thus cannot handle a large number of genes. We need to find a good balance between number of genes and number of cells for the toy dataset.
        adata = adata[:, adata.var.highly_variable].copy()

    print(f"Write training data to {target_path}")
    adata.write_h5ad(target_path)


def main():
    parser = argparse.ArgumentParser(description="Create the processed SW480 data set from Tahoe-100M plate 2")
    parser.add_argument("--download", action="store_true", help="Download Tahoe-100M plate 2 if it is not present yet")
    parser.add_argument("--adata_train_path", type=str, default="../data/tahoe100M_plate2.h5ad", help="Path of the raw plate file")
    parser.add_argument("--target_path", type=str, default="../data/tahoe100M_SW480.h5ad", help="Path where the normalized cell-line data set is saved")
    parser.add_argument("--num_cells", type=int, default=100_000, help="Not used (all cells are kept)")
    parser.add_argument("--cell_line", type=str, default="SW480", help="Cell line to keep")
    parser.add_argument("--highly_variable", action="store_true", help="Select 10,000 HVGs already in step 1 (not used in the thesis)")
    parser.add_argument("--hvg_genes", type=int, default=1024, help="Number of HVGs kept in step 2 (thesis: 1024)")
    args = parser.parse_args()

    if args.download:
        download_plate(args.adata_train_path)

    # Step 1: cell-line subset, normalisation, log1p
    if os.path.exists(args.target_path):
        print(f"{args.target_path} already exists, skipping step 1.")
    else:
        create_dataset(args.adata_train_path, args.target_path, args.num_cells, cell_line=args.cell_line, highly_variable=args.highly_variable)

    # Step 2: HVG selection with the same function that produced the thesis data set
    # (utils/data_utils.load_adata writes ../data/<name>_filtered_<n>.h5ad)
    dataset_name = Path(args.target_path).stem
    hvg_path = Path(args.target_path).with_name(f"{dataset_name}_filtered_{args.hvg_genes}.h5ad")
    if hvg_path.exists():
        print(f"{hvg_path} already exists, skipping step 2.")
    else:
        from utils import load_adata
        print(f"Select {args.hvg_genes} highly variable genes -> {hvg_path}")
        load_adata(dataset_name, args.hvg_genes)


if __name__ == "__main__":
    main()
