"""Weather/clear balanced sampling without replacement."""
import torch
from torch.utils.data import DataLoader
from .weather import WeatherWindows

def sample_dataloader(dataloader, num_samples, batch_size):
    """
    Sample data from dataloder.

    Args
        dataloader (torch.utils.data.DataLoader): Dataloader.
        num_samples (int): Number of samples.
        batch_size (int): Batch size.

    Returns
        sample_dataloader (torch.utils.data.DataLoader): Sample dataloader.
    """
    data = dataloader.dataset.data
    targets = dataloader.dataset.targets

    # v2 P0: Balanced weather/clear sampling. Uniform sampling yields ~80% clear windows per iteration,
    # providing a weak weather contrast signal; the model only learns to distinguish clear conditions → representation collapse.
    # Here, weather and clear windows each make up ~50% of the sample, providing sufficient gradients for weather retrieval. (The last dimension is CLEAR.)
    N = data.shape[0]
    t = torch.as_tensor(targets, dtype=torch.float32)
    is_weather = t[:, :-1].sum(dim=1) > 1e-6
    n_w = int(is_weather.sum())
    n_c = N - n_w
    num = min(num_samples, N)
    if 0 < n_w < N:
        w = torch.empty(N, dtype=torch.float32)
        w[is_weather] = 0.5 / n_w
        w[~is_weather] = 0.5 / n_c
        sample_index = torch.multinomial(w, num, replacement=False)
    else:
        sample_index = torch.randperm(N)[:num]
    data = data[sample_index]
    targets = targets[sample_index]
    sample = DataLoader(WeatherWindows(data, targets), batch_size=batch_size,
                        pin_memory=True, shuffle=True)

    return sample, sample_index

