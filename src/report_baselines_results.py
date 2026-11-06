"""
Evaluate the mean and mean-shift baselines for all 20 seeds and write them to a CSV file.

Run from the `src/` directory. No GPU is needed. The data are split exactly as in run.py,
so the baselines are evaluated on the same test cells as the models.

    cd src
    python report_baselines_results.py --task SEEN
    python report_baselines_results.py --task UNSEEN-PERT

Both calls append to the same CSV (default: ../plots/data/baseline_results_reproduced.csv).
The thesis results are in ../plots/data/results/final/baseline_results.csv.
"""
import argparse
import json
import os 

import scanpy as sc
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from configs import rectified_flow_config_defaults
from models import compute_mean_baselines
from utils import load_adata, process_adata, process_adata_unseen_pert, set_seed
from datasets import TahoeMultiDataset


def _collect_test_arrays(test_dataset):
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    x_test_list = []
    y_test_list = []
    drug_id_list = []
    for X_test, y_test, drug_id in test_loader:
        x_test_list.append(X_test.numpy())
        y_test_list.append(y_test.numpy())
        drug_id_list.append(drug_id.numpy())

    return (
        np.concatenate(x_test_list, axis=0),
        np.concatenate(y_test_list, axis=0),
        np.concatenate(drug_id_list, axis=0),
    )


def _compute_for_adata(adata, config, use_unseen_pooled_baseline):
    if config["task"] == "SEEN":
        train_dataset, val_dataset, test_dataset = process_adata(
            adata,
            config["perturbations"],
            config["seed"],
        )
        test_perturbations = config["perturbations"]
        drug_id_offset = 0
    elif config["task"] == "UNSEEN-PERT":
        train_dataset, val_dataset, test_dataset = process_adata_unseen_pert(
            adata,
            config["perturbations"],
            config["unseen_perturbations"],
            config["seed"],
        )
        test_perturbations = config["unseen_perturbations"]
        drug_id_offset = len(config["perturbations"])

        # Subsample each unseen perturbation to ~the per-perturbation test size of the SEEN task (~900 cells),
        # keeping the full control pool (x) untouched
        
        # n_cells_per_pert = 937
        # rng = np.random.default_rng(config["seed"])
        # drug_ids = test_dataset.drug_ids.numpy().flatten()
        # idx = np.concatenate([
        #     rng.choice(np.where(drug_ids == d)[0], size=min(n_cells_per_pert, (drug_ids == d).sum()), replace=False)
        #     for d in np.unique(drug_ids)
        # ])
        # idx.sort()

        # test_dataset = TahoeMultiDataset(test_dataset.x, test_dataset.y[idx], test_dataset.drug_ids[idx], seed=config["seed"])
    else:
        raise ValueError(f"Unsupported task: {config['task']!r}")

    print(f"TASK: {config['task']} - Train/Val/Test sizes: {len(train_dataset)}/{len(val_dataset)}/{len(test_dataset)}")
    print(np.unique_counts(test_dataset.drug_ids.numpy().flatten()).counts)
    print(np.mean(np.unique_counts(test_dataset.drug_ids.numpy().flatten()).counts))
    
    x_test, y_test, drug_id_list = _collect_test_arrays(test_dataset)
    drug_id_list = drug_id_list - drug_id_offset

    mean_baseline_results, mean_shift_baseline_results = compute_mean_baselines(
        train_dataset,
        x_test,
        y_test,
        drug_id_list,
        adata.var,
        test_perturbations,
        pool_train_perturbations=use_unseen_pooled_baseline,
    )

    return {
        "train_size": len(train_dataset),
        "val_size": len(val_dataset),
        "test_size": len(test_dataset),
        "test_perturbations": test_perturbations,
        "mean_baseline": mean_baseline_results,
        "mean_shift_baseline": mean_shift_baseline_results,
    }


def get_baseline_results(task, seed):
    config = dict(rectified_flow_config_defaults)
    config["seed"] = seed
    config["task"] = task

    set_seed(config["seed"])
    raw_adata = sc.read_h5ad(f"../data/{config['dataset']}.h5ad")
    # Not needed for the reported results (the data set is already HVG-filtered); this call
    # would only recompute the HVGs and write an extra copy of the data set to ../data/.
    # filtered_adata = load_adata(config["dataset"], config["HVG_GENE_NUMBER"])

    raw_results = _compute_for_adata(raw_adata, config, use_unseen_pooled_baseline=(config["task"] == "UNSEEN-PERT"))
    # filtered_results = _compute_for_adata(filtered_adata, config, use_unseen_pooled_baseline=(config["task"] == "UNSEEN-PERT"))

    return {
        "seed": config["seed"],
        "dataset": config["dataset"],
        "task": config["task"],
        "perturbations": config["perturbations"],
        "unseen_perturbations": config["unseen_perturbations"],
        "raw_adata_reference": raw_results,
        # "filtered_adata_reference": filtered_results,
    }


# def _print_summary(label, results):
#     mean_baseline_results = results["mean_baseline"]
#     mean_shift_baseline_results = results["mean_shift_baseline"]

#     mean_baseline_de_f1 = np.mean(mean_baseline_results["DE_F1"])
#     mean_baseline_precision = np.mean(mean_baseline_results["DE_Precision"])
#     mean_baseline_recall = np.mean(mean_baseline_results["DE_Recall"])
#     mean_baseline_energy = np.mean(mean_baseline_results["Energy_Loss"])
#     mean_baseline_mse = np.mean(mean_baseline_results["MSE"])

#     mean_shift_baseline_de_f1 = np.mean(mean_shift_baseline_results["DE_F1"])
#     mean_shift_baseline_precision = np.mean(mean_shift_baseline_results["DE_Precision"])
#     mean_shift_baseline_recall = np.mean(mean_shift_baseline_results["DE_Recall"])
#     mean_shift_baseline_energy = np.mean(mean_shift_baseline_results["Energy_Loss"])
#     mean_shift_baseline_mse = np.mean(mean_shift_baseline_results["MSE"])

#     print(f"\n{label}")
#     print(f"Train/Val/Test sizes: {results['train_size']}/{results['val_size']}/{results['test_size']}")
#     print(f"Test perturbations: {results['test_perturbations']}")
#     print("Mean Baseline Results:")
#     print(f"DE-F1: {mean_baseline_de_f1:.4f}")
#     print(f"Precision: {mean_baseline_precision:.4f}")
#     print(f"Recall: {mean_baseline_recall:.4f}")
#     print(f"Energy Loss: {mean_baseline_energy:.4f}")
#     print(f"MSE: {mean_baseline_mse:.4f}")

#     print("Mean Shift Baseline Results:")
#     print(f"DE-F1: {mean_shift_baseline_de_f1:.4f}")
#     print(f"Precision: {mean_shift_baseline_precision:.4f}")
#     print(f"Recall: {mean_shift_baseline_recall:.4f}")
#     print(f"Energy Loss: {mean_shift_baseline_energy:.4f}")
#     print(f"MSE: {mean_shift_baseline_mse:.4f}")


def _rows_from_results(results):
    rows = []
    # for reference_name, reference_key in [("raw", "raw_adata_reference"), ("filtered_hvg", "filtered_adata_reference")]:
    #     reference_results = results[reference_key]
    reference_results = results["raw_adata_reference"]
    for baseline_name, baseline_key in [("mean", "mean_baseline"), ("mean_shift", "mean_shift_baseline")]:
        baseline_results = reference_results[baseline_key]
        rows.append({
            "seed": results["seed"],
            "task": results["task"],
            # "reference": reference_name,
            "baseline": baseline_name,
            "DE_F1": np.mean(baseline_results["DE_F1"]),
            "DE_Precision": np.mean(baseline_results["DE_Precision"]),
            "DE_Recall": np.mean(baseline_results["DE_Recall"]),
            "Energy_Loss": np.mean(baseline_results["Energy_Loss"]),
            "MSE": np.mean(baseline_results["MSE"]),
            # "DE-F1_NUM_True": np.mean(baseline_results["DE_F1_NUM_True"]),
            # "DE-F1_NUM_Pred": np.mean(baseline_results["DE_F1_NUM_Pred"]),
            "DE-F1_all": baseline_results["DE_F1"],
            "DE-Precision_all": baseline_results["DE_Precision"],
            "DE-Recall_all": baseline_results["DE_Recall"],
            "Energy_Loss_all": baseline_results["Energy_Loss"],
            "MSE_all": baseline_results["MSE"],
        })
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the mean and mean-shift baselines.")
    parser.add_argument("--task", choices=["SEEN", "UNSEEN-PERT"], required=True)
    parser.add_argument("--seeds", type=int, nargs="+",
                        default=[123, 456, 789, 101112, 131415, 161718, 192021, 222324, 252627, 282930, 313233, 343536, 373839, 404142, 434445, 464748, 495051, 525354, 555657, 585960])
    parser.add_argument("--output", default="../plots/data/baseline_results_reproduced.csv",
                        help="CSV file to append the results to (created with a header if it does not exist)")
    args = parser.parse_args()

    all_rows = []
    for seed in args.seeds:
        set_seed(seed)
        results = get_baseline_results(task=args.task, seed=seed)
        all_rows.extend(_rows_from_results(results))

    # Append to an existing file (e.g. SEEN first, then UNSEEN-PERT) without repeating the header
    output_path = args.output
    write_header = not os.path.exists(output_path)
    pd.DataFrame(all_rows).to_csv(output_path, index=False, mode="a", header=write_header)
    print(f"\nSaved baseline results for all seeds to {output_path}")
