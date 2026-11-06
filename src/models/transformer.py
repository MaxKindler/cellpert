import torch
import torch.nn as nn
import math
import torch.nn.functional as F

from .perturbation_encoder import PerturbationEncoder

class TransformerCrossAttentionMultiModel(nn.Module):
    """
    Cross-attention via concatenation and self-attention, following the
    MHNfs CrossAttentionModule pattern:
    1) Encode perturbation 
    2) Concatenate cell expression values with the perturbation 
    3) Residual Connection
    4) Discard perturbation embedding to output only the predicted gene expression values
    """

    def __init__(self, n_genes, hidden_size, embed_dim=128, num_heads=8,
                 num_layers=2, dropout=0.1, fingerprints=None):
        super().__init__()
        self.n_genes = n_genes
        self.embed_dim = embed_dim
        self.total_dim = n_genes + embed_dim

        # Perturbation embedding
        self.perturbation_embeddings = PerturbationEncoder(fingerprints, embed_dim)

        # Transformer block operates on the concatenated sequence
        # Using the nanoGPT-style block like the reference
        self.transformer_blocks = nn.ModuleList([
            self.TransformerBlock(self.total_dim, hidden_size, num_heads, dropout) # self.total_dim
            for _ in range(num_layers)
        ])


    def encode_perturbation(self, drug_id):
        pert_emb = self.perturbation_embeddings(drug_id.int()) 
        return  pert_emb

    def forward(self, x, drug_id):
        # x: (batch_size, n_genes)
        # drug_id: (batch_size, 1)

        # Step 1: Encode perturbation
        perturbation_token = self.encode_perturbation(drug_id)  

        # Concatenate: input genes first, then the perturbation
        combined = torch.cat([x, perturbation_token], dim=1).unsqueeze(0) 

        # Step 2: Self-attention over batches of joint sequence of genes and perturbation embedding
        h = combined
        for block in self.transformer_blocks:
            h = block(h)

        # Step 3: Residual connection 
        h = combined + h # This residual connection should be removed, but is kept for compatibility with the original thesis experiments
        h = h.squeeze(0) # shape: (batch_size, n_genes + emb_dim)

        # Step 4: Extract only the input cell tokens (discard perturbation embedding)
        output = h[:, :self.n_genes] 

        return torch.relu(output + x) # This residual connection should be removed, but is kept for compatibility with the original thesis experiments

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
                self.relu = nn.ReLU()
                # NOTE: c_proj1 is not used in forward(). It is kept on purpose because it existed
                # when the thesis experiments were run: creating it draws random numbers, so removing
                # it changes the initialisation of all following layers (and the checkpoint keys).
                # Unfortunately it distorts the number of parameters in the model.
                self.c_proj1 = nn.Linear(hidden_size, hidden_size)
                self.c_proj = nn.Linear(hidden_size, n_embd)
                self.dropout = nn.Dropout(dropout)

            def forward(self, x):
                x = self.c_fc(x)
                x = self.relu(x)
                x = self.c_proj(x)
                x = self.dropout(x)
                return x