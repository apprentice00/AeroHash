"""One sampled epoch: forward, cache, objective, backward and reporting."""
from tqdm import tqdm
from loguru import logger
from .meters import AverageMeter
from .batch_objectives import compute_batch_objectives


def train_epoch(model, train_dataloader, optimizer, criterion, U, B, S,
                sample_index, center_loss, severity, sim_sigma, it, epoch, args):
    """Optimize sampled windows; local indices address U and rows of S.

    sample_index maps those local indices to gallery rows in B. Only U is
    detached: differentiable codes feed all active objectives before backward.
    """
    model.train()
    if center_loss is not None:
        center_loss.train()
    names = ('cnn_loss', 'hash_loss', 'quan_loss', 'center_loss', 'tfc_loss',
             'ord_loss', 'bal_loss', 'dec_loss', 'total_loss', 'center_weighted',
             'tfc_weighted', 'ord_weighted', 'bal_weighted', 'dec_weighted')
    meters = {name: AverageMeter() for name in names}
    pbar = tqdm(train_dataloader, total=len(train_dataloader), ncols=50, leave=False)
    for data, targets, index in pbar:
        data, targets, index = data.to(args.device), targets.to(args.device), index.to(args.device)
        optimizer.zero_grad()
        codes = model(data)['pred']
        U[index, :] = codes.detach()
        total_loss, terms = compute_batch_objectives(
            codes, data, targets, criterion, B, S[index, :], sample_index[index],
            center_loss, severity, sim_sigma, it, args)
        for name, value in terms.items():
            meters[name].update(value.detach().item(), n=data.size(0))
        total_loss.backward()
        optimizer.step()
    diagnostics = ''.join(f'[{name}:{meter.avg:.6f}]' for name, meter in meters.items())
    logger.info('[epoch:{}/{}]{}', epoch + 1, args.max_epoch, diagnostics)
