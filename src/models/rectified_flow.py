import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .perturbation_encoder import PerturbationEncoder

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim, scale=1000.0):
        super().__init__()
        self.dim = dim
        self.scale = scale  # stretches t in [0,1] so sin/cos bands aren't all near-identical

    def forward(self, t):
        # t: (batch, 1)
        half_dim = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half_dim, device=t.device, dtype=t.dtype) / half_dim
        )
        args = t * self.scale * freqs 
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb

# MLP architecture for the Rectified Flow model
class MLP(nn.Module):

    def __init__(self, in_dim, time_dim, context_dim, h, out_dim, dropout, fingerprints):
        super(MLP, self).__init__()
        self.backbone = nn.Sequential(nn.Linear(in_dim + time_dim + context_dim, h),
                                     nn.Tanh(),
                                     nn.Dropout(dropout),
                                     nn.Linear(h, h),
                                     nn.Tanh(),
                                     nn.Dropout(dropout),)
        # Velocity head predicts dz/dt; gate head predicts logit(P(gene is zero in z1)),
        # used at inference to zero-mask genes instead of a hard magnitude threshold.
        self.time_embed = SinusoidalTimeEmbedding(time_dim)
        
        self.velocity_head = nn.Linear(h, out_dim)
        self.gate_head = nn.Linear(h, out_dim)

        self.embedding = PerturbationEncoder(fingerprints, context_dim)

    def forward(self, x, t, context):
        context_emb = self.embedding(context.int())  
        time_emb = self.time_embed(t)  
        features = self.backbone(torch.cat((x, time_emb, context_emb), dim=1))
        return self.velocity_head(features), self.gate_head(features)
