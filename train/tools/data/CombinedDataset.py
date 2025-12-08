import torch
from torch.utils.data import Dataset
import random
import bisect

class CombinedDataset(Dataset):
    def __init__(self, datasets, weights=None, seed=42):
        """
        Args:
            datasets (list of Dataset): List of PyTorch Dataset objects.
            weights (list of float, optional): Sampling probabilities for each dataset.
                                               Should sum to 1. If None, uniform probabilities are used.
            seed (int): Random seed for reproducibility.
        """
        self.datasets = datasets
        self.num_datasets = len(datasets)
        self.seed = seed
        random.seed(seed)

        if weights is None:
            self.weights = [1.0 / self.num_datasets] * self.num_datasets
        else:
            if len(weights) != self.num_datasets:
                raise ValueError("Length of weights must match number of datasets.")
            total = sum(weights)
            self.weights = [w / total for w in weights]  # normalize to sum 1

        # cumulative weights for bisect
        self.cum_weights = []
        cum = 0
        for w in self.weights:
            cum += w
            self.cum_weights.append(cum)

        # calculate approximate total length as sum of lengths
        self.total_len = sum(len(ds) for ds in datasets)

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        # sample a dataset according to weights
        r = random.random()
        dataset_idx = bisect.bisect_right(self.cum_weights, r)
        dataset = self.datasets[dataset_idx]

        # pick a random sample from chosen dataset
        sample_idx = random.randint(0, len(dataset) - 1)
        return dataset[sample_idx]
