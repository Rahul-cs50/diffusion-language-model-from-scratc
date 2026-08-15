import torch
import torch.nn as nn

from .config import DiffusionLMConfig


class DiffusionTransformerLM(nn.Module):
    """Minimal bidirectional Transformer denoiser for discrete text."""

    def __init__(self, cfg: DiffusionLMConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.seq_len, cfg.d_model)
        self.time_emb = nn.Embedding(cfg.diffusion_steps + 1, cfg.d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.d_ff,
            dropout=cfg.dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=cfg.n_layers)
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, input_ids, timesteps, attention_mask=None):
        _, length = input_ids.shape
        if length > self.cfg.seq_len:
            raise ValueError(f"Sequence length {length} > cfg.seq_len {self.cfg.seq_len}")
        positions = torch.arange(length, device=input_ids.device).unsqueeze(0)
        x = self.tok_emb(input_ids) + self.pos_emb(positions)
        x = x + self.time_emb(timesteps).unsqueeze(1)
        x = self.drop(x)
        padding_mask = None if attention_mask is None else ~attention_mask
        x = self.encoder(x, src_key_padding_mask=padding_mask)
        x = self.ln_f(x)
        return self.lm_head(x)

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())
