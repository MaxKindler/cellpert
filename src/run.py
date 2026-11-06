"""
Train and evaluate one model architecture (Rectified Flow, Transformer or Transformer Flow).

Run this script from the `src/` directory (all data/model paths are relative to it):

    cd src
    python run.py --architecture rectified_flow --task SEEN           # 20 seeds, thesis setup
    python run.py --architecture rectified_flow --task UNSEEN-PERT --reflow 2   # reflow experiment
    python run.py --architecture transformer_flow --task SEEN --run_sweep        # hyperparameter sweep

For every seed the script splits the data (see utils/data_utils.py), trains the model,
keeps the checkpoint with the best validation DE-F1, evaluates it on the test set together
with the mean / mean-shift baselines and logs everything to Weights & Biases (wandb).
See README.md for details.
"""
import argparse
import gc
import os
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

import numpy as np
import torch
from torch.utils.data import DataLoader
import scanpy as sc

from configs import ARCHITECTURES
from configs import rectified_flow_sweep_config, transformer_sweep_config, transformer_flow_sweep_config
from datasets import PerPerturbationSampler, TahoeReflowDataset
from models import compute_mean_baselines
from utils import load_adata, process_adata, process_adata_unseen_pert, get_top_genes, plot_violin_plots
from utils import save_model, load_model, delete_model
from utils import set_seed, init_wandb_run, evaluate_per_perturbation, log_and_finish_wandb_run

import wandb
from utils.experiment_setup import WANDB_PROJECT, WANDB_ENTITY

# Seeds used for all results reported in the thesis (mean +- std over these 20 runs).
SEEDS = [123, 456, 789, 101112, 131415, 161718, 192021, 222324, 252627, 282930, 313233, 343536, 373839, 404142, 434445, 464748, 495051, 525354, 555657, 585960]

# Log in to wandb unless logging is switched off (WANDB_MODE=offline or disabled).
if os.environ.get("WANDB_MODE") not in ("offline", "disabled", "dryrun"):
    wandb.login()


def train(spec, config_defaults=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    run_name, config = init_wandb_run(config_defaults=config_defaults)
    model_family = config.model_family
    dataset_batching = spec["dataset_batching"]

    # adata = load_adata(config.dataset, config.HVG_GENE_NUMBER)
    adata = sc.read_h5ad(f"../data/{config.dataset}.h5ad")

    # SEEN: train/val/test share the same perturbations, split by cell.
    # UNSEEN-PERT: test perturbations are held out entirely from train/val, so
    # their drug IDs are offset to avoid colliding with train drug IDs.
    if config.task == "SEEN":
        train_dataset, val_dataset, test_dataset = process_adata(adata, config.perturbations, config.seed)
        test_perturbations = config.perturbations
        drug_id_offset = 0
    elif config.task == "UNSEEN-PERT":
        train_dataset, val_dataset, test_dataset = process_adata_unseen_pert(adata, config.perturbations, config.unseen_perturbations, config.seed)
        test_perturbations = config.unseen_perturbations
        drug_id_offset = len(config.perturbations)
    else:
        raise ValueError(f"Unknown task: {config.task!r}. Expected 'SEEN' or 'UNSEEN-PERT' (UNSEEN-CL not yet implemented).")

    if dataset_batching == "per_perturbation":
        per_perturbation_sampler = PerPerturbationSampler(train_dataset, batch_size=config.batch_size)
        per_perturbation_sampler_val = PerPerturbationSampler(val_dataset, batch_size=256)
        per_perturbation_sampler_test = PerPerturbationSampler(test_dataset, batch_size=256)
    else:
        train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=256, shuffle=False)
        test_dataloader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    input_dim = adata.n_vars  # Number of genes

    model = spec["model_cls"](**spec["model_kwargs"](config, input_dim)).to(device)
    # Number of parameters distorted due to extra unused layer in Transformer and Transformer Flow implementation
    # print(f"Model has {sum(p.numel() for p in model.parameters())} parameters")

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    trainer_extra_kwargs = spec["trainer_kwargs"](model, optimizer, config, device)
    criterion = trainer_extra_kwargs["criterion"]

    trainer = spec["trainer_cls"](model=model, optimizer=optimizer, device=device, **trainer_extra_kwargs)

    # Everything below is wrapped in try/finally so that a mid-run exception
    # (OOM, a bad batch, whatever) still frees the GPU memory and finishes the
    # wandb run. Without this, a crashed trial leaks its model/optimizer and a
    # zombie wandb run, so a long sweep gets progressively more likely to OOM
    # on later trials regardless of their own hyperparameters.
    try:
        # Best-checkpoint tracking + early stopping: save whenever val_loss improves,
        # and stop once it hasn't improved for `early_stopping_patience` epochs, so we
        # never report/test a model that has drifted past its best validation point.
        best_val_loss = 0 #float("inf")
        epochs_without_improvement = 0
        patience = config.early_stopping_patience

        for epoch in range(config.epochs):
            if dataset_batching == "per_perturbation":
                train_dataloader = DataLoader(train_dataset, batch_sampler=per_perturbation_sampler.get_batches())

            train_loss, _ = trainer.train_epoch(train_dataloader)

            if dataset_batching == "per_perturbation":
                val_dataloader = DataLoader(val_dataset, batch_sampler=per_perturbation_sampler_val.get_batches())

            val_loss, pred_val_list, x_val_list, y_val_list, drug_id_list, avg_std = trainer.validate(val_dataloader)

            # drug_id_list = drug_id_list - drug_id_offset
            metrics_results = evaluate_per_perturbation(pred_val_list, y_val_list, x_val_list, drug_id_list, config.perturbations, adata.var)
            avg_de_f1_score = np.mean(metrics_results["DE_F1"])
            avg_precision_score = np.mean(metrics_results["DE_Precision"])
            avg_recall_score = np.mean(metrics_results["DE_Recall"])
            avg_energy_score = np.mean(metrics_results["Energy_Loss"])
            avg_mse_loss = np.mean(metrics_results["MSE"])

            print(f"Avg DE F1: {avg_de_f1_score:.4f}, Avg DE Precision: {avg_precision_score:.4f}, Avg DE Recall: {avg_recall_score:.4f}, Avg Energy Loss: {avg_energy_score:.4f}, Avg MSE Loss: {avg_mse_loss:.4f}")
            print(f"Epoch {epoch+1}/{config.epochs}, Val {config.loss_function}: {val_loss:.4f}, Avg Std: {avg_std:.4f}")

            if avg_de_f1_score > best_val_loss:
                best_val_loss = avg_de_f1_score
                epochs_without_improvement = 0
                save_model(model, config.architecture, run_name, model_family)
            else:
                epochs_without_improvement += 1

            wandb.log({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "best_val_loss": best_val_loss,
                "de_val_score": avg_de_f1_score,
                "de_precision_val_score": avg_precision_score,
                "de_recall_val_score": avg_recall_score,
                "energy_val_score": avg_energy_score,
                "mse_val_loss": avg_mse_loss,
            })

            if patience > 0 and epochs_without_improvement >= patience:
                print(f"Early stopping at epoch {epoch+1}: val_loss has not improved for {patience} epochs (best: {best_val_loss:.4f})")
                break

        # Reload the best validation checkpoint (not just whatever epoch training
        # stopped/ended on) before any reflow refinement or final testing.
        model = load_model(model, config.architecture, run_name, model_family, device)


        # TEST: evaluate the best validation checkpoint on the test set, together with
        # the mean / mean-shift baselines evaluated on the same test cells, and write
        # violin plots of the top DE genes. These are the results reported in the thesis.
        if dataset_batching == "per_perturbation":
            test_dataloader = DataLoader(test_dataset, batch_sampler=per_perturbation_sampler_test.get_batches())

        test_loss, pred_test_list, X_test_list, y_test_list, drug_id_list, avg_std_test = trainer.validate(test_dataloader)

        print(f"Test min/mean/max {pred_test_list[0].min():.4f}/{pred_test_list[0].mean():.4f}/{pred_test_list[0].max():.4f}")
        print(f"True min/mean/max {y_test_list[0].min():.4f}/{y_test_list[0].mean():.4f}/{y_test_list[0].max():.4f}")

        # Map drug IDs to indices into test_perturbations (unseen drugs have IDs offset by len(perturbations))
        drug_id_list = drug_id_list - drug_id_offset
        mean_baseline_results, mean_shift_baseline_results = compute_mean_baselines(
            train_dataset, X_test_list, y_test_list, drug_id_list, adata.var, test_perturbations,
            pool_train_perturbations=(config.task == "UNSEEN-PERT"),
        )

        model_results = evaluate_per_perturbation(pred_test_list, y_test_list, X_test_list, drug_id_list, test_perturbations, adata.var)

        # Create violin plots
        # top_genes = get_top_genes(adata, test_perturbations, top_n=20)
        # for i, perturbation in enumerate(test_perturbations):
        #     idx = (drug_id_list == i).squeeze()
        #     plot_violin_plots(pred_test_list[idx], y_test_list[idx], adata.var, top_genes, perturbation, run_name, architecture=config.architecture)

        # Logs test_loss, de_test_score, ..., and the baseline metrics to wandb
        log_and_finish_wandb_run(test_loss, model_results, mean_baseline_results, mean_shift_baseline_results)

        # REFLOW: only rectified-flow-style architectures set config.reflow > 0;
        # everything else (e.g. transformer_cellattention) defaults it to 0, so this
        # loop is a no-op for them. Model/trainer are rebuilt via the spec so reflow
        # never silently swaps in a different architecture than the one selected.
        for reflow_epoch in range(config.reflow):
            _, pred_list, X_list, _, drug_id_list, _ = trainer.validate(train_dataloader, n_steps=config.n_steps)
            reflow_dataset = TahoeReflowDataset(X_list, pred_list, drug_id_list, seed=config.seed)
            reflow_dataloader = DataLoader(reflow_dataset, batch_size=config.batch_size, shuffle=True)

            model = spec["model_cls"](**spec["model_kwargs"](config, input_dim)).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
            trainer_extra_kwargs = spec["trainer_kwargs"](model, optimizer, config, device)
            criterion = trainer_extra_kwargs["criterion"]
            trainer = spec["trainer_cls"](model=model, optimizer=optimizer, device=device, **trainer_extra_kwargs)

            for epoch in range(config.epochs):
                train_loss, _ = trainer.train_epoch(reflow_dataloader)
                val_loss, pred_val_list, X_val_list, y_val_list, drug_id_list, avg_std = trainer.validate(val_dataloader, n_steps=config.n_steps)

                # Validation cells only contain training perturbations, so no drug-ID offset is applied
                # (subtracting it only mislabelled the printed perturbation names; metrics are unaffected).
                metrics_results = evaluate_per_perturbation(pred_val_list, y_val_list, X_val_list, drug_id_list, config.perturbations, adata.var)
                avg_de_f1_score = np.mean(metrics_results["DE_F1"])
                avg_precision_score = np.mean(metrics_results["DE_Precision"])
                avg_recall_score = np.mean(metrics_results["DE_Recall"])
                avg_energy_score = np.mean(metrics_results["Energy_Loss"])
                avg_mse_loss = np.mean(metrics_results["MSE"])

                print(f"Reflow-{reflow_epoch+1} Epoch {epoch+1}/{config.epochs}, Val {config.loss_function}: {val_loss:.4f}, Avg Std: {avg_std:.4f}")
                wandb.log({
                    "epoch": epoch + 1,
                    f"reflow-{reflow_epoch+1}_loss": train_loss,
                    f"reflow-{reflow_epoch+1}_val_loss": val_loss,
                    f"reflow-{reflow_epoch+1}_de_val_score": avg_de_f1_score,
                    f"reflow-{reflow_epoch+1}_de_precision_val_score": avg_precision_score,
                    f"reflow-{reflow_epoch+1}_de_recall_val_score": avg_recall_score,
                    f"reflow-{reflow_epoch+1}_energy_val_score": avg_energy_score,
                    f"reflow-{reflow_epoch+1}_mse_val_loss": avg_mse_loss,
                })

            # No-op when reflow is disabled (model is already the best main-loop checkpoint,
            # reloaded above); persists reflow's result otherwise, since the reflow loop
            # itself doesn't do best-checkpoint tracking.
            save_model(model, config.architecture, run_name, model_family)

            # Test loop: reload the just-saved checkpoint so test metrics reflect the
            # saved artifact, not just whatever is left in memory.
            model = load_model(model, config.architecture, run_name, model_family, device)

            if dataset_batching == "per_perturbation":
                test_dataloader = DataLoader(test_dataset, batch_sampler=per_perturbation_sampler_test.get_batches())

            test_loss, pred_test_list, X_test_list, y_test_list, drug_id_list, avg_std_test = trainer.validate(test_dataloader)

            print(f"Test min/mean/max {pred_test_list[0].min():.4f}/{pred_test_list[0].mean():.4f}/{pred_test_list[0].max():.4f}")
            print(f"True min/mean/max {y_test_list[0].min():.4f}/{y_test_list[0].mean():.4f}/{y_test_list[0].max():.4f}")

            drug_id_list = drug_id_list - drug_id_offset
            mean_baseline_results, mean_shift_baseline_results = compute_mean_baselines(
                train_dataset, X_test_list, y_test_list, drug_id_list, adata.var, test_perturbations,
                pool_train_perturbations=(config.task == "UNSEEN-PERT"),
            )

            model_results = evaluate_per_perturbation(pred_test_list, y_test_list, X_test_list, drug_id_list, test_perturbations, adata.var)

            top_genes = get_top_genes(adata, test_perturbations, top_n=20)
            for i, perturbation in enumerate(test_perturbations):
                idx = (drug_id_list == i).squeeze()
                plot_violin_plots(pred_test_list[idx], y_test_list[idx], adata.var, top_genes, perturbation, run_name, architecture=config.architecture)

            # log_and_finish_wandb_run(test_loss, model_results, mean_baseline_results, mean_shift_baseline_results)
            wandb.log({
                f"reflow-{reflow_epoch+1}_test_loss": test_loss, 
                f"reflow-{reflow_epoch+1}_de_test_score": np.mean(model_results["DE_F1"]), 
                f"reflow-{reflow_epoch+1}_de_precision_test_score": np.mean(model_results["DE_Precision"]), 
                f"reflow-{reflow_epoch+1}_de_recall_test_score": np.mean(model_results["DE_Recall"]), 
                f"reflow-{reflow_epoch+1}_energy_test_score": np.mean(model_results["Energy_Loss"]), 
                f"reflow-{reflow_epoch+1}_mse_test_loss": np.mean(model_results["MSE"]),
                f"reflow-{reflow_epoch+1}_de_test_score_all": model_results["DE_F1"],
                f"reflow-{reflow_epoch+1}_de_precision_test_score_all": model_results["DE_Precision"],
                f"reflow-{reflow_epoch+1}_de_recall_test_score_all": model_results["DE_Recall"],
                f"reflow-{reflow_epoch+1}_energy_test_score_all": model_results["Energy_Loss"],
                f"reflow-{reflow_epoch+1}_mse_test_loss_all": model_results["MSE"],
            })

        wandb.finish()
    except Exception as e:
        print(f"Exception occurred during training: {e}")
        import traceback
        print(f"{type(e).__module__}.{type(e).__name__}: {e!r}")
        traceback.print_exc()
    finally:
        # because of memory issues, we need to clear the GPU memory after each run
        # (runs on success AND on a mid-run exception - see comment above)
        #
        # `trainer` (and trainer_extra_kwargs, which for rectified_flow/transformer_flow
        # holds the LR scheduler) keep their own references to model/optimizer, so
        # deleting only the outer `model`/`optimizer` names doesn't actually drop their
        # refcount to zero - the CUDA tensors stay alive, `empty_cache()` has nothing to
        # free, and this trial's GPU memory leaks into the next one for the rest of the
        # sweep's process lifetime. Deleting `trainer`/`trainer_extra_kwargs` too closes
        # that gap.
        del trainer
        del trainer_extra_kwargs
        del model
        del optimizer
        del criterion
        # Remove saved model to free up disk space, since we log it to wandb anyway
        if config.seed != 123:  # keep the first seed's model for debugging purposes
            delete_model(config.architecture, run_name, model_family)
        # delete_model(config.architecture, run_name, model_family)
        gc.collect()
        torch.cuda.empty_cache()
        if wandb.run is not None:
            wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the model training and evaluation.")
    parser.add_argument("--architecture", choices=ARCHITECTURES.keys(), required=True,
                        help="rectified_flow | transformer_cellattention | transformer_flow")
    parser.add_argument("--task", choices=["SEEN", "UNSEEN-PERT"], default=None,
                        help="Overrides TASK in the config file. SEEN: test on held-out cells of the training perturbations; "
                             "UNSEEN-PERT: test on perturbations never seen during training.")
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS, help="Seeds to run (default: the 20 thesis seeds)")
    parser.add_argument("--reflow", type=int, default=None,
                        help="Number of reflow rounds (rectified flow only; the thesis reflow experiment uses 2). Overrides the config.")
    parser.add_argument("--sweep_id", default=None, help="WandB sweep ID to resume a sweep (optional)")
    parser.add_argument("--cuda_device", default="0", help="GPU index (sets CUDA_VISIBLE_DEVICES)")
    parser.add_argument("--run_sweep", action="store_true", default=False, help="Run a hyperparameter sweep instead of a training run")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device

    spec = ARCHITECTURES[args.architecture]
    config_defaults = spec["config_defaults"]
    if args.task is not None:
        config_defaults["task"] = args.task
    if args.reflow is not None:
        config_defaults["reflow"] = args.reflow

    if not args.run_sweep:
        # Run a training with 20 seeds for robustness
        for seed in args.seeds:
            config_defaults["seed"] = seed
            set_seed(config_defaults["seed"])
            train(spec, config_defaults)
    else:
        # Run a wandb hyperparameter sweep instead of training runs (the thesis sweeps used the SEEN task, seed 123)
        if args.sweep_id is not None:
            sweep_id = args.sweep_id if WANDB_ENTITY is None else f"{WANDB_ENTITY}/{WANDB_PROJECT}/{args.sweep_id}"
        else:
            if args.architecture == "rectified_flow":
                sweep_id = wandb.sweep(rectified_flow_sweep_config, project=WANDB_PROJECT, entity=WANDB_ENTITY)
            elif args.architecture == "transformer_cellattention":
                sweep_id = wandb.sweep(transformer_sweep_config, project=WANDB_PROJECT, entity=WANDB_ENTITY)
            elif args.architecture == "transformer_flow":
                sweep_id = wandb.sweep(transformer_flow_sweep_config, project=WANDB_PROJECT, entity=WANDB_ENTITY)
            else:
                raise ValueError(f"Unknown architecture: {args.architecture}")

        print(f"Starting sweep with ID: {sweep_id}")
        wandb.agent(sweep_id=sweep_id, function=lambda: train(spec, config_defaults), count=100, project=WANDB_PROJECT, entity=WANDB_ENTITY)
