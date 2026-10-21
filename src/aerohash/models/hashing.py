"""Window-to-hash model with global average pooling and an MLP head."""
import torch
from torch import nn
from .encoders import TSEncoder, FreqEncoder
from .fusion import GatedFusion
from .validation import validate_model_settings

class MyModelTS(nn.Module):
    """Weather window → continuous hash code.

    x → TSEncoder / FreqEncoder → GatedFusion → Transformer.encoder(pos)
      → temporal global average pooling → MLP → quantize → pred

    quant_mode: 'tanh' | 'anneal' | 'ste'
    """

    def __init__(self, in_dim, transformer, seq_len, code_length,
                 dropout=0.0, use_tf=False, quant_mode='tanh',
                 use_deriv=False, use_ch_attn=False):
        super().__init__()
        validate_model_settings(
            quant_mode=quant_mode, seq_len=seq_len, in_dim=in_dim,
            hidden_dim=transformer.d_model, nheads=transformer.nhead,
            code_length=code_length, use_tf=use_tf)
        self.in_dim = in_dim
        self.seq_len = seq_len
        self.transformer = transformer
        self.quant_mode = quant_mode
        self.use_tf = use_tf
        self.hash_scale = 1.0  # Used for annealing; updated by the training loop
        hidden_dim = transformer.d_model

        self.ts_encoder = TSEncoder(
            in_dim, hidden_dim, dropout=dropout,
            use_deriv=use_deriv, use_ch_attn=use_ch_attn,
        )
        if use_tf:
            self.freq_encoder = FreqEncoder(
                in_dim, hidden_dim, n_freq=seq_len // 2 + 1)
            self.gated_fusion = GatedFusion(hidden_dim)

        self.pos_embed = nn.Embedding(seq_len, hidden_dim)
        nn.init.uniform_(self.pos_embed.weight)

        self.mlp_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, code_length),
        )

    def __setstate__(self, state):
        """Restore input metadata absent from historical full-model checkpoints."""
        super().__setstate__(state)
        if not hasattr(self, 'seq_len'):
            self.seq_len = self.pos_embed.num_embeddings
        if not hasattr(self, 'in_dim'):
            channels = self.ts_encoder.proj.in_channels
            self.in_dim = channels // 2 if self.ts_encoder.use_deriv else channels

    def _encode_tokens(self, x):
        """Time/frequency encoding + optional gated fusion → (B, C, W)."""
        H_t = self.ts_encoder(x)
        if self.use_tf:
            return self.gated_fusion(H_t, self.freq_encoder(x))
        return H_t

    def _quantize(self, z):
        if self.quant_mode == 'anneal':
            return torch.tanh(self.hash_scale * z)
        if self.quant_mode == 'ste':
            t = torch.tanh(z)
            return t + (t.sign() - t).detach()
        if self.quant_mode == 'tanh':
            return torch.tanh(z)
        raise ValueError(f'Unknown quant_mode: {self.quant_mode!r}')

    def _pool_tokens(self, feat):
        """(B, C, W) → positional self-attention → window feature (B, C).

        Position is added to attention queries/keys in every encoder layer;
        values retain the encoded features. Pool only over the time axis.
        """
        bs, _, length = feat.shape
        src = feat.flatten(2).permute(2, 0, 1)  # (W, B, C)
        pos = self.pos_embed.weight[:length].unsqueeze(1).repeat(1, bs, 1)
        memory = self.transformer.encoder(src, src_key_padding_mask=None, pos=pos)
        return memory.mean(dim=0)

    def forward(self, x):
        """Fixed windows (B, W, D) → codes (B, M) and pooled features (B, C)."""
        if x.ndim != 3 or x.shape[1:] != (self.seq_len, self.in_dim):
            raise ValueError(
                f'Expected input (N, {self.seq_len}, {self.in_dim}), got {tuple(x.shape)}')
        pooled = self._pool_tokens(self._encode_tokens(x))
        pred = self._quantize(self.mlp_head(pooled))
        # Historical consumers expect both feature aliases and the attention placeholder.
        return {'pred': pred, 'attn_weights': None, 'hs': pooled, 'hs_ori': pooled}
