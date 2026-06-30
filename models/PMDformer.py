import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from math import sqrt


class _SelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        d_keys = d_model // n_heads
        d_values = d_model // n_heads

        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, L, _ = x.shape
        _, S, _ = x.shape
        H = self.n_heads

        queries = self.query_projection(x).view(B, L, H, -1)
        keys = self.key_projection(x).view(B, S, H, -1)
        values = self.value_projection(x).view(B, S, H, -1)

        scale = 1.0 / sqrt(queries.size(-1))
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        attn = self.dropout(torch.softmax(scale * scores, dim=-1))
        values = torch.einsum("bhls,bshd->blhd", attn, values)

        out = rearrange(values, "b l h d -> b l (h d)")
        return self.out_projection(out), attn


class _TrendRestorationAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        d_keys = d_model // n_heads
        d_values = d_model // n_heads

        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, means):
        B, L, _ = x.shape
        _, S, _ = x.shape
        H = self.n_heads

        queries = self.query_projection(x).view(B, L, H, -1)
        keys = self.key_projection(x).view(B, S, H, -1)
        values = self.value_projection(x).view(B, S, H, -1) + means.unsqueeze(2)

        scale = 1.0 / sqrt(queries.size(-1))
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        attn = self.dropout(torch.softmax(scale * scores, dim=-1))
        values = torch.einsum("bhls,bshd->blhd", attn, values)

        out = rearrange(values, "b l h d -> b l (h d)")
        return self.out_projection(out), attn


class _EncoderLayerPVA(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        new_x, attn = self.attention(x)
        x = x + self.dropout(new_x)

        y = self.norm1(x)
        y = self.dropout(F.gelu(self.linear1(y)))
        y = self.dropout(self.linear2(y))
        return self.norm2(x + y), attn


class _EncoderPVA(nn.Module):
    def __init__(self, attn_layers, norm_layer=None):
        super().__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(self, x):
        attns = []
        for attn_layer in self.attn_layers:
            x, attn = attn_layer(x)
            attns.append(attn)
        if self.norm is not None:
            x = self.norm(x)
        return x, attns


class _EncoderLayerTRA(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, means):
        new_x, attn = self.attention(x, means)
        x = x + self.dropout(new_x)

        y = self.norm1(x)
        y = self.dropout(F.gelu(self.linear1(y)))
        y = self.dropout(self.linear2(y))
        return self.norm2(x + y), attn


class _EncoderTRA(nn.Module):
    def __init__(self, attn_layers, norm_layer=None):
        super().__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(self, x, means):
        attns = []
        for attn_layer in self.attn_layers:
            x, attn = attn_layer(x, means)
            attns.append(attn)
        if self.norm is not None:
            x = self.norm(x)
        return x, attns


class Model(nn.Module):
    """PMDformer, adapted from github.com/aohu1105/PMDformer."""

    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.d_model = configs.d_model
        self.enc_in = configs.enc_in
        self.e_layers = configs.e_layers
        self.use_norm = bool(configs.use_norm)
        self.patch_size = configs.patch_size
        self.stride = configs.patch_size

        self.patch_num = (configs.seq_len - self.patch_size) // self.stride + 1
        if self.patch_num <= 0:
            raise ValueError("PMDformer requires patch_size <= seq_len")

        self.Embedding_layer = nn.Linear(self.patch_size, configs.d_model)
        self.Predicting_layer = nn.Linear(configs.d_model * self.patch_num, configs.pred_len)

        self.Encoder_TRA = _EncoderTRA(
            [
                _EncoderLayerTRA(
                    _TrendRestorationAttention(
                        d_model=configs.d_model,
                        n_heads=configs.n_heads,
                        dropout=configs.dropout,
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                )
                for _ in range(configs.e_layers)
            ],
            norm_layer=nn.LayerNorm(configs.d_model),
        )

        self.Encoder_PVA = _EncoderPVA(
            [
                _EncoderLayerPVA(
                    _SelfAttention(
                        d_model=configs.d_model,
                        n_heads=configs.n_heads,
                        dropout=configs.dropout,
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                )
                for _ in range(configs.v_layers)
            ],
            norm_layer=nn.LayerNorm(configs.d_model),
        )

        self.patch_pos = nn.Parameter(torch.zeros(self.patch_num, configs.d_model))

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        x = x_enc
        B, _, N = x.shape

        x = rearrange(x, "b l n -> b n l")
        if self.use_norm:
            means = x.mean(-1, keepdim=True)
            stdev = torch.sqrt(torch.var(x, dim=-1, keepdim=True, unbiased=False) + 1e-5)
            x = (x - means) / stdev

        x = x.unfold(dimension=-1, size=self.patch_size, step=self.stride)
        x = rearrange(x, "b n p s -> (b n) p s")

        patch_means = x.mean(-1, keepdim=True)
        x = x - patch_means

        x = self.Embedding_layer(x)
        x = x + self.patch_pos.unsqueeze(0)

        x_main = x[:, :-1, :]
        x_last = x[:, -1:, :]
        x_last = rearrange(x_last, "(b n) p d -> (b p) n d", b=B, n=N)
        x_last, _ = self.Encoder_PVA(x_last)
        x_last = rearrange(x_last, "(b p) n d -> (b n) p d", b=B, n=N)
        x = torch.cat([x_main, x_last], dim=1)

        x, _ = self.Encoder_TRA(x, patch_means)
        x = x + patch_means
        x = rearrange(x, "(b n) p d -> b n (p d)", b=B, n=N)
        x = self.Predicting_layer(x)

        if self.use_norm:
            x = x * stdev + means

        return rearrange(x, "b n t -> b t n")
