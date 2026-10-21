"""Dataset-specific objective setup and active model diagnostics."""
import os
import json
import numpy as np
import torch
from loguru import logger
from aerohash.losses.center import SeverityCenterLoss


def build_objectives(model, retrieval_targets, args, dimensions):
    # v2: Severity weights are shared across the four stations; use the first station. RBF bandwidth sigma varies slightly by station → use the mean for combined training.
    with open(os.path.join(args.root, 'config.json'), encoding='utf-8') as f:
        data_cfg = json.load(f)
    severity = torch.tensor(data_cfg['severity_weight'], dtype=torch.float32,
                            device=args.device)
    roots = getattr(args, 'roots', [args.root])
    _sigmas = []
    for _r in roots:
        with open(os.path.join(_r, 'config.json'), encoding='utf-8') as f:
            _sigmas.append(float(json.load(f)['sigma']))
    sim_sigma = float(np.mean(_sigmas))

    # Contribution ①: Severity-aware hyperspherical center loss (multi-label soft proxies).
    # class_freq = number of training/gallery windows N_c containing each class, used for the long-tail frequency margin.
    center_loss = None
    if getattr(args, 'use_center', False):
        class_freq = (retrieval_targets > 0).float().sum(dim=0)
        center_loss = SeverityCenterLoss(
            num_classes=args.num_classes,
            code_length=dimensions.training_dim,
            severity=severity,
            class_freq=class_freq,
            m0=args.center_m0,
            lam=args.center_lambda,
            gamma=args.center_gamma,
            s=args.center_s,
            margin_mode=args.center_margin_mode,
        ).to(args.device)
        logger.info('[center] mode={} weight={} m_c={}'.format(
            args.center_margin_mode, args.center_weight,
            [round(float(v), 4) for v in center_loss.m_c.tolist()]))

    # Contribution ②: Status of time-frequency gated fusion (encoder branches) and TFC physics-guided supervision (loss term)
    logger.info('[tf] use_tf(TSEncoder||FreqEncoder→GatedFusion)={} use_tfc(supervision)={} tfc_weight={}'.format(
        getattr(args, 'use_tf', False), getattr(args, 'use_tfc', False),
        getattr(args, 'tfc_weight', 0.0)))

    # Contribution ③: Status of ordinal (relative ranking) soft supervision (weather anchors + weather-negative weighting + warmup)
    logger.info('[ord] use_ord={} weight={} margin={} s_margin={} neg_boost={} warmup={}'.format(
        getattr(args, 'use_ord', False), getattr(args, 'ord_weight', 0.0),
        getattr(args, 'ord_margin', 0.0), getattr(args, 'ord_smargin', 0.0),
        getattr(args, 'ord_neg_boost', 0.0), getattr(args, 'ord_warmup', 0)))

    # Quantization path and bit-regularization status
    logger.info('[quantization] quant_mode={} scale=[{}->{}] use_bitreg={} bal_w={} dec_w={}'.format(
        getattr(args, 'quant_mode', 'tanh'),
        getattr(args, 'quant_scale_min', 1.0), getattr(args, 'quant_scale_max', 1.0),
        getattr(args, 'use_bitreg', False), getattr(args, 'bit_balance_weight', 0.0),
        getattr(args, 'bit_decorr_weight', 0.0)))

    # P1: Status of first-order difference input features and ECA channel attention in TSEncoder
    logger.info('[p1] use_deriv={} use_ch_attn={}'.format(
        getattr(args, 'use_deriv', False), getattr(args, 'use_ch_attn', False)))

    logger.info('[GAP] n_params={}', sum(p.numel() for p in model.parameters() if p.requires_grad))

    return severity, sim_sigma, center_loss
