import os
import torch

def save_model(model, architecture, run_name, model_family):
    if model_family not in ["Transformer", "RectifiedFlow", "TransformerFlow"]:
        raise ValueError(f"Unknown model family: {model_family}")
    # Checkpoints are written to <repo>/models/<family>/runs/ (relative to src/)
    os.makedirs(f"../models/{model_family}/runs", exist_ok=True)
    torch.save(model.state_dict(), f"../models/{model_family}/runs/{architecture}_{run_name}.pth")

def load_model(model, architecture, run_name, model_family, device):
    if model_family not in ["Transformer", "RectifiedFlow", "TransformerFlow"]:
        raise ValueError(f"Unknown model family: {model_family}")
    model.load_state_dict(torch.load(f"../models/{model_family}/runs/{architecture}_{run_name}.pth", map_location=device))
    model.to(device)
    return model

def delete_model(architecture, run_name, model_family):
    if model_family not in ["Transformer", "RectifiedFlow", "TransformerFlow"]:
        raise ValueError(f"Unknown model family: {model_family}")
    model_path = f"../models/{model_family}/runs/{architecture}_{run_name}.pth"
    if os.path.exists(model_path):
        os.remove(model_path)

    
