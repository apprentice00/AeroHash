"""Single-station and concatenated multi-station loading."""
from . import weather

def load_data(dataset, root, num_query, num_train, batch_size, num_workers):
    """
    Load dataset.

    Args
        dataset(str): Dataset name.
        root(str): Path of dataset.
        num_query(int): Number of query data points.
        num_train(int): Number of training data points.
        num_workers(int): Number of loading data threads.

    Returns
        query_dataloader, train_dataloader, retrieval_dataloader(torch.utils.data.DataLoader): Data loader.
    """
    if dataset == 'weather':
        query_dataloader, train_dataloader, retrieval_dataloader = weather.load_data(root,
                                      batch_size,
                                      num_workers,
                                      )
    else:
        raise ValueError("Invalid dataset name!")

    return query_dataloader, train_dataloader, retrieval_dataloader

def load_data_multi(dataset, roots, batch_size, num_workers):
    """Load combined multi-station data (scheme ①: concatenate train_core). Return (None, train, retrieval)."""
    if dataset == 'weather':
        return weather.load_data_multi(roots, batch_size, num_workers)
    raise ValueError("Invalid dataset name!")
