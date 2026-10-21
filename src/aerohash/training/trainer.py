"""ADSH training orchestration: sample, optimize, update codes, select and report."""
import os
import time
import torch
import torch.optim as optim
from loguru import logger
from aerohash.models.factory import build_ts
from aerohash.models.dimensions import resolve_dimensions
from aerohash.losses.adsh import ADSH_Loss
from aerohash.losses.similarity import weighted_rbf_similarity
from aerohash.data.sampling import sample_dataloader
from aerohash.evaluation.stations import load_stations, selection_scores, final_evaluation
from aerohash.checkpoints import save_checkpoint
from aerohash.config import write_resolved_config
from .dcc import solve_dcc, calc_loss
from .epoch import train_epoch
from .optimizer import build_optimizer
from .objectives import build_objectives
from .schedule import evaluation_step, EarlyStopping

def train(train_loader, retrieval_dataloader, code_length, args):
    """Alternate sampled neural optimization and discrete gallery updates.

    Select the best model by station-macro weather mAP and return that score.
    """
    print('train start')

    # Explicit run names isolate parallel experiments; otherwise use a timestamp.
    ckpt_root = args.ckpt_root
    ckpt_name = args.ckpt_name or time.strftime('%Y-%m-%d-%H-%M-%S')
    ret_path = os.path.join(ckpt_root, ckpt_name)
    os.makedirs(ret_path, exist_ok=True)
    print('checkpoint dir:', ret_path)

    dimensions = resolve_dimensions(code_length)
    args.training_dim = dimensions.training_dim
    args.retrieval_dim = dimensions.retrieval_dim
    step = evaluation_step(args)
    write_resolved_config(args, ret_path)
    logger.info('[model] head=GAP code_length={} enc_layers={}', code_length, args.enc_layers)
    if args.net == "ts":
        model = build_ts(args, code_length, dimensions=dimensions)
    else:
        raise ValueError('Invalid net: {} (only "ts" is supported)'.format(args.net))

    model.to(args.device)

    num_retrieval = len(retrieval_dataloader.dataset)

    criterion = ADSH_Loss(dimensions.training_dim, args.gamma)
    # Sampling is capped by the dataset; U uses the same local index space.
    num_sampled = min(args.num_samples, len(train_loader.dataset))
    U = torch.zeros(num_sampled, dimensions.training_dim).to(args.device)
    B = torch.randn(num_retrieval, dimensions.training_dim).to(args.device)

    retrieval_targets = retrieval_dataloader.dataset.get_onehot_targets().to(args.device)
    severity, sim_sigma, center_loss = build_objectives(model, retrieval_targets, args, dimensions)
    eval_stations, select_on_val = load_stations(args, args.roots)

    # Optimizer: model parameters + optional center parameters as a separate parameter group (supports center_lr)
    optimizer = build_optimizer(model, center_loss, args)

    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, args.lr_step,gamma=args.lr_gamma)

    start = time.time()
    best_mAP = 0
    es_patience = int(getattr(args, 'early_stop_patience', 10) or 0)
    es_min_iter = int(getattr(args, 'early_stop_min_iter', 40) or 0)
    early_stop = EarlyStopping(es_patience, es_min_iter)
    for it in range(args.max_iter):
        iter_start = time.time()

        train_dataloader, sample_index = sample_dataloader(train_loader, args.num_samples, args.batch_size)
        sample_index = sample_index.to(args.device)

        # Create Similarity matrix (v2: severity-weighted RBF soft supervision, with CLEAR downweighted)
        train_targets = train_dataloader.dataset.get_onehot_targets().to(args.device)
        S = weighted_rbf_similarity(train_targets, retrieval_targets,
                                    severity, sim_sigma)
        # v2 P0: Center to [-1,1] so dissimilar pairs receive negative targets (repulsion).
        # With S∈[0,1], the ADSH target code_length*S is always ≥0, causing global attraction → constant codes become nearly optimal
        # → representation collapse (identical query codes). After centering, constant codes cannot satisfy both positive and negative targets, breaking collapse.
        S = 2.0 * S - 1.0

        # P0-②: Quantization-scale annealing: increase the tanh slope linearly with training progress to approach sign.
        if getattr(args, 'quant_mode', 'tanh') == 'anneal':
            progress = it / max(1, args.max_iter - 1)
            model.hash_scale = (args.quant_scale_min
                                + (args.quant_scale_max - args.quant_scale_min) * progress)

        # Optimize the neural hash function on the sampled windows.
        for epoch in range(args.max_epoch):
            train_epoch(model, train_dataloader, optimizer, criterion, U, B, S,
                        sample_index, center_loss, severity, sim_sigma, it, epoch, args)

        scheduler.step()

        expand_U = torch.zeros(B.shape).to(args.device)
        expand_U[sample_index, :] = U
        B = solve_dcc(B, U, expand_U, S, dimensions.training_dim, args.gamma)

        iter_loss = calc_loss(U, B, S, dimensions.training_dim, sample_index, args.gamma)
      
        logger.info('[iter:{}/{}][loss:{:.6f}][iter_time:{:.2f}]'.format(it+1, args.max_iter, iter_loss, time.time()-iter_start))

        scheduled_evaluation = (it < step and (it + 1) % args.val_freq == 0) or it >= step
        if scheduled_evaluation or it + 1 == args.max_iter:
            # Leakage-free model selection: prefer independent val (X_val_query×X_val_db, ground truth S_val); fall back to
            # test if unavailable. Use macro-averaged weather mAP across stations. Encoding b=sign(pred) matches aerohash evaluate.
            overall_mAP, weather_mAP, clear_mAP, n_uniq, n_q = selection_scores(
                model, eval_stations, select_on_val, args.device)
            mAP = weather_mAP
            if early_stop.observe(mAP):
                save_checkpoint(model, os.path.join(ret_path, 'model.t'), args.device)
            best_mAP, no_improve = early_stop.best, early_stop.no_improve
            tag = 'val' if select_on_val else 'test(LEAK)'
            logger.info('[iter:{}/{}][code_length:{}][{}][overall:{:.5f}][weather:{:.5f}][clear:{:.5f}][best_weather:{:.5f}][q_uniq:{}/{}][no_improve:{}]'.format(
                it+1, args.max_iter, code_length, tag, overall_mAP, weather_mAP, clear_mAP, best_mAP, n_uniq, n_q, no_improve))
            # Early stopping: minimum iterations reached and consecutive non-improving iterations exceed patience
            if early_stop.should_stop(it + 1):
                logger.info('[early_stop][iter:{}/{}][no_improve:{}>=patience:{}][best_weather:{:.5f}]'.format(
                    it + 1, args.max_iter, no_improve, es_patience, best_mAP))
                break
    logger.info('[Training time:{:.2f}]'.format(time.time()-start))

    # After training: run per-station out-of-sample test evaluation on the best checkpoint from this run;
    # if val is missing, model selection also uses test (compatibility fallback), which is not independent validation.
    # Write metrics_weather_<station>.json + macro-averaged metrics_weather.json;
    # concatenate station codes into codes_query/codes_db.npy for collect_multiseed q_uniq statistics.
    if not early_stop.has_score:
        raise RuntimeError('Training produced no finite selection score; '
                           'cannot evaluate a checkpoint from this run')
    final_evaluation(ret_path, best_mAP, eval_stations, select_on_val, args.device)

    return best_mAP
