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
        args = t * self.scale * freqs  # (batch, half_dim)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class TransformerVelocityModel(nn.Module):
    """
    Cross-attention via concatenation and self-attention, following the
    MHNfs CrossAttentionModule pattern:
    1) Encode perturbation 
    2) Concatenate cell expression values with the perturbation 
    3) Residual Connection
    4) Discard perturbation embedding to output only the predicted gene expression values
    """

    def __init__(self, n_genes, hidden_size, embed_dim=128, time_dim=128, num_heads=8,
                 num_layers=2, dropout=0.1, fingerprints=None):
        super().__init__()
        self.n_genes = n_genes
        self.embed_dim = embed_dim - time_dim
        self.time_dim = time_dim
        self.total_dim = n_genes + time_dim + self.embed_dim

        # Perturbation embedding
        self.perturbation_embeddings = PerturbationEncoder(fingerprints, self.embed_dim)
        self.time_embed = SinusoidalTimeEmbedding(self.time_dim)

        # Transformer block operates on the concatenated sequence
        # Using the nanoGPT-style block like the reference
        self.transformer_blocks = nn.ModuleList([
            self.TransformerBlock(self.total_dim, hidden_size, num_heads, dropout) 
            for _ in range(num_layers)
        ])

        # Gate head predicts logit(P(gene is zero in z1)) from the full post-attention
        # hidden state, used at inference to zero-mask genes instead of a hard threshold.
        self.gate_head = nn.Linear(self.total_dim, n_genes)

    def encode_perturbation(self, drug_id):
        pert_emb = self.perturbation_embeddings(drug_id.int()) 
        return  pert_emb

    def forward(self, x, t, drug_id):
        # x: (batch_size, n_genes)
        # drug_id: (batch_size, 1)

        # Step 1: Encode perturbation
        perturbation_token = self.encode_perturbation(drug_id) 

        # Encode time to fit dimensions
        time_emb = self.time_embed(t) 


        # Concatenate: input genes first, then the perturbation
        combined = torch.cat([x, time_emb, perturbation_token], dim=1).unsqueeze(0) 

        # Step 2: Self-attention over batches of joint sequence of genes and perturbation embedding
        h = combined
        for block in self.transformer_blocks:
            h = block(h)

        h = h.squeeze(0) 

        # Step 3: Extract only the input cell tokens (discard perturbation embedding)
        velocity = h[:, :self.n_genes] - x 
        # Subtract x back out to get the predicted delta

        # Gate logits use the full hidden state (genes + time + perturbation channels)
        gate_logits = self.gate_head(h)

        return velocity, gate_logits

    # -------------------------------------------------------------------------
    # Sub-modules (following nanoGPT style from reference)
    class TransformerBlock(nn.Module):
        def __init__(self, n_embd, hidden_size, n_head, dropout):
            super().__init__()
            self.ln_1 = nn.LayerNorm(n_embd)
            self.attn = self.SelfAttention(n_embd, n_head, dropout)
            self.ln_2 = nn.LayerNorm(n_embd)
            self.mlp = self.MLP(n_embd, hidden_size, dropout)

        def forward(self, x):
            x = x + self.attn(self.ln_1(x))
            x = x + self.mlp(self.ln_2(x))
            return x

        class SelfAttention(nn.Module):
            def __init__(self, n_embd, n_head, dropout):
                super().__init__()
                assert n_embd % n_head == 0
                self.n_head = n_head
                self.head_dim = n_embd // n_head

                self.q_proj = nn.Linear(n_embd, n_embd)
                self.k_proj = nn.Linear(n_embd, n_embd)
                self.v_proj = nn.Linear(n_embd, n_embd)
                self.c_proj = nn.Linear(n_embd, n_embd)

                self.attn_dropout = nn.Dropout(dropout)
                self.resid_dropout = nn.Dropout(dropout)

            def forward(self, x):
                B, T, C = x.size()

                q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
                k = self.k_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
                v = self.v_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)

                att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
                att = F.softmax(att, dim=-1)
                att = self.attn_dropout(att)

                y = att @ v
                y = y.transpose(1, 2).contiguous().view(B, T, C)
                y = self.resid_dropout(self.c_proj(y))
                return y

        class MLP(nn.Module):
            def __init__(self, n_embd, hidden_size, dropout):
                super().__init__()
                self.c_fc = nn.Linear(n_embd, hidden_size)
                self.tanh = nn.Tanh()
                # NOTE: c_proj1 is not used in forward(). It is kept on purpose because it existed
                # when the thesis experiments were run: creating it draws random numbers, so removing
                # it changes the initialisation of all following layers (and the checkpoint keys).
                # Unfortunately it distorts the number of parameters in the model.
                self.c_proj1 = nn.Linear(hidden_size, hidden_size)
                self.c_proj = nn.Linear(hidden_size, n_embd)
                self.dropout = nn.Dropout(dropout)

            def forward(self, x):
                x = self.c_fc(x)
                x = self.tanh(x)
                x = self.c_proj(x)
                x = self.dropout(x)
                return x
