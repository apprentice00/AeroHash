"""Encode model predictions to {-1,+1} int8 hash codes."""
import numpy as np
import torch

@torch.no_grad()
def encode_array(model, X, device, batch_size=512):
    """Use the model as a hash function: b=sign(model(x)['pred'])∈{-1,+1}, int8.

    Model selection during training and standalone evaluation share sign encoding; zero predictions map to +1.
    """
    if len(X) == 0:
        raise ValueError('Cannot encode an empty evaluation array')
    if batch_size < 1:
        raise ValueError('batch_size must be positive')
    was_training = model.training
    model.eval()
    codes = []
    try:
        for s in range(0, len(X), batch_size):
            xb = torch.from_numpy(X[s:s + batch_size]).float().to(device)
            pred = model(xb)['pred']
            codes.append(pred.sign().cpu().numpy())
    finally:
        model.train(was_training)
    out = np.concatenate(codes, axis=0)
    out[out == 0] = 1
    return out.astype(np.int8)
