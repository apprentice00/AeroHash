"""Train a model with resolved experiment configuration."""
import os
import random
from pathlib import Path
import numpy as np
from aerohash.config import parse_config, training_namespace


def seed_everything(seed):
    import torch
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


def main(argv=None):
    config = parse_config('train', argv)
    os.environ['CUDA_VISIBLE_DEVICES'] = config.training.cuda_visible_devices
    from loguru import logger
    from aerohash.data.loaders import load_data_multi
    from aerohash.training.trainer import train
    args = training_namespace(config)
    seed_everything(args.seed)
    log_path = Path(args.log_root)/f'{args.arch}-{args.net}'/args.info
    log_path.mkdir(parents=True, exist_ok=True)
    sink = logger.add(str(log_path/'-{time:YYYY-MM-DD-HH}.log'), rotation='500 MB', level='INFO')
    try:
        logger.success(config)
        _, train_loader, retrieval = load_data_multi(args.dataset, args.roots, args.batch_size, args.num_workers)
        if args.arch != 'query_opt':
            raise ValueError(f'Invalid architecture name: {args.arch}')
        for code_length in args.code_length:
            mAP = train(train_loader, retrieval, code_length, args)
            logger.info('[code_length:{}][map:{:.4f}]', code_length, mAP)
    finally:
        logger.remove(sink)
