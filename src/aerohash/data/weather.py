"""Window datasets retain float32 soft labels and shared training/gallery indices."""
import os

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader


class WeatherWindows(Dataset):
    """Window collection for a single split (train/database/query)."""

    def __init__(self, data, targets):
        # data: (N, W, D) float32; targets: (N, K) float32 multi-hot
        self.data = data
        self.targets = targets

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        x = torch.from_numpy(self.data[index]).float()  # (W, D)
        y = self.targets[index]
        return x, y, index

    def get_onehot_targets(self):
        return torch.from_numpy(self.targets).float()


def _load_split(root, split):
    X = np.load(os.path.join(root, f"X_{split}.npy")).astype(np.float32)
    Y = np.load(os.path.join(root, f"Y_{split}.npy")).astype(np.float32)
    # v2: Preserve soft labels (mean-pooled frequency × intensity) for severity-weighted similarity;
    # do not binarize here, as binarization would discard intensity information.
    targets = Y.astype(np.float32)
    return X, targets


def _load_train_core(root):
    """Prefer X_train_core produced by aerohash build-val (validation pool removed to prevent leakage);
    otherwise, warn and fall back to the full X_train (validation may overlap with training; for legacy data compatibility only)."""
    if os.path.isfile(os.path.join(root, "X_train_core.npy")):
        return _load_split(root, "train_core")
    print(f"[warn] {root} has no X_train_core.npy; falling back to X_train (run aerohash build-val first)")
    return _load_split(root, "train")


def load_data_multi(roots, batch_size, num_workers):
    """Scheme ①: Combined multi-station training. Each station is normalized using its own norm_stats;
    concatenate train_core as the unified training set and retrieval database B (train == retrieval).

    Query evaluation uses per-station out-of-sample evaluation in training (independent val/test per station).
    No query_dataloader is returned here (None preserves compatibility with the three-element return signature).
    """
    Xs, Ys = [], []
    for root in roots:
        Xt, Yt = _load_train_core(root)
        Xs.append(Xt)
        Ys.append(Yt)
    X = np.concatenate(Xs, axis=0)
    Y = np.concatenate(Ys, axis=0)

    train_dataset = WeatherWindows(X, Y)
    retrieval_dataset = WeatherWindows(X, Y)
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        pin_memory=True, num_workers=num_workers,
    )
    retrieval_dataloader = DataLoader(
        retrieval_dataset, batch_size=batch_size,
        pin_memory=True, num_workers=num_workers,
    )
    return None, train_dataloader, retrieval_dataloader


def load_data(root, batch_size, num_workers, sampler=None):
    """Return query/train/retrieval DataLoaders.

    ADSH/QueryOpt protocol (consistent with the original CUB code): the retrieval database is the training set itself.
    ADSH learns a binary code matrix B for the retrieval database and samples training points from it each iteration.
    The quantization term indexes B by training samples' database indices, so train and retrieval must share the same index space.
      train == retrieval -> X_train   (learn the hash function + serve as the retrieval database)
      query             -> X_query    (independent queries outside the database)

    The supplied X_database split supports arbitrary-set encoding for the DPSH baseline and is not used in
    ADSH code optimization; using it as a gallery requires rewriting ADSH pairwise/quantization terms (outside the current scope).
    """
    Xq, Yq = _load_split(root, "query")
    Xt, Yt = _load_split(root, "train")

    query_dataset = WeatherWindows(Xq, Yq)
    train_dataset = WeatherWindows(Xt, Yt)
    retrieval_dataset = WeatherWindows(Xt, Yt)

    query_dataloader = DataLoader(
        query_dataset,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=num_workers,
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers,
    )
    retrieval_dataloader = DataLoader(
        retrieval_dataset,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=num_workers,
    )

    return query_dataloader, train_dataloader, retrieval_dataloader
