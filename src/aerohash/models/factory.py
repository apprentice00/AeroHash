"""Build the GAP hashing model from resolved runtime settings."""
from .hashing import MyModelTS
from .transformer import build_transformer
from .validation import validate_model_settings
from .dimensions import resolve_dimensions


def build_ts(args, code_length, *, dimensions=None):
    resolved = resolve_dimensions(code_length)
    if dimensions is not None and dimensions != resolved:
        raise ValueError('Supplied dimensions do not match model settings')
    validate_model_settings(
        quant_mode=getattr(args, 'quant_mode', 'tanh'),
        seq_len=args.seq_len, in_dim=args.in_dim, hidden_dim=args.hidden_dim,
        nheads=args.nheads, code_length=code_length,
        use_tf=getattr(args, 'use_tf', False))
    return MyModelTS(
        in_dim=args.in_dim,
        transformer=build_transformer(args),
        seq_len=args.seq_len,
        code_length=code_length,
        dropout=args.dropout,
        use_tf=getattr(args, 'use_tf', False),
        quant_mode=getattr(args, 'quant_mode', 'tanh'),
        use_deriv=getattr(args, 'use_deriv', False),
        use_ch_attn=getattr(args, 'use_ch_attn', False),
    )
