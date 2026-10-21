"""Early validation of the GAP model's input and architecture settings."""


def positive_integer(name, value):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f'{name} must be a positive integer, got {value!r}')


def validate_model_settings(*, quant_mode, seq_len, in_dim,
                            hidden_dim, nheads, code_length, use_tf):
    if quant_mode not in ('tanh', 'anneal', 'ste'):
        raise ValueError(f'Unknown quant_mode: {quant_mode!r}')
    for name, value in (('seq_len', seq_len), ('in_dim', in_dim),
                        ('hidden_dim', hidden_dim), ('nheads', nheads),
                        ('code_length', code_length)):
        positive_integer(name, value)
    if hidden_dim % nheads:
        raise ValueError('hidden_dim must be divisible by nheads')
    if use_tf and seq_len < 2:
        raise ValueError('seq_len must be at least 2 when use_tf is enabled (FFT removes DC)')
