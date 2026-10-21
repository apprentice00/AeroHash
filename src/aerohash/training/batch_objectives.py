"""Compose sampled-batch objectives in the historical arithmetic order."""
from aerohash.losses.tfc import tfc_relational_loss
from aerohash.losses.ordinal import ordinal_relational_loss
from aerohash.losses.bit_regularization import bit_balance_loss, bit_decorrelation_loss


def compute_batch_objectives(codes, data, targets, criterion, B, S, omega,
                             center_loss, severity, sim_sigma, iteration, args):
    """Return differentiable total plus raw and weighted scalar diagnostics.

    Names ending in _weighted denote the actual additive contribution. ADSH
    already includes its quantization weighting; it is never added twice.
    """
    cnn_loss, hash_loss, quan_loss = criterion(codes, B, S, omega)
    total = cnn_loss
    terms = {'cnn_loss': cnn_loss, 'hash_loss': hash_loss, 'quan_loss': quan_loss}

    def add(name, value, weight):
        nonlocal total
        weighted = weight * value
        terms[f'{name}_loss'] = value
        terms[f'{name}_weighted'] = weighted
        total = total + weighted

    if center_loss is not None:
        add('center', center_loss(codes, targets), args.center_weight)
    if getattr(args, 'use_tfc', False):
        add('tfc', tfc_relational_loss(codes, data), args.tfc_weight)
    if getattr(args, 'use_ord', False) and iteration >= getattr(args, 'ord_warmup', 0):
        add('ord', ordinal_relational_loss(
            codes, targets, severity, sim_sigma, args.ord_margin, args.ord_smargin,
            args.ord_neg_boost), args.ord_weight)
    if getattr(args, 'use_bitreg', False):
        add('bal', bit_balance_loss(codes), args.bit_balance_weight)
        add('dec', bit_decorrelation_loss(codes), args.bit_decorr_weight)
    terms['total_loss'] = total
    return total, terms
