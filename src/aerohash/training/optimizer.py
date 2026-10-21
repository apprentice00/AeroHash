"""Optimizer construction with a separate learnable-center parameter group."""
import torch.optim as optim

def build_optimizer(model, center_loss, args):
    param_groups = [{'params': model.parameters(), 'lr': args.lr}]
    if center_loss is not None:
        param_groups.append({'params': center_loss.parameters(),
                             'lr': getattr(args, 'center_lr', args.lr)})
    if args.optim == 'SGD':
        optimizer = optim.SGD(param_groups, lr=args.lr, weight_decay=args.wd, momentum=args.momen, nesterov=args.nesterov)
    elif args.optim == 'AdamW':
        # v2 P0: AdamW + weight decay to curb hash-head weight growth that causes tanh saturation/vanishing gradients
        optimizer = optim.AdamW(param_groups, lr=args.lr, weight_decay=args.wd)
    elif args.optim == 'Adam':
        optimizer = optim.Adam(param_groups, lr=args.lr, weight_decay=args.wd)
    else:
        raise ValueError('Invalid optim: {}'.format(args.optim))

    return optimizer
