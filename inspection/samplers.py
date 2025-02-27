import torch  # Import PyTorch library

# Define a custom Sampler class for random sampling of a subset of indices
class SubsetRandomSampler(torch.utils.data.Sampler):
    """
    A custom Sampler that randomly samples a subset of indices.
    This is useful for creating a DataLoader that iterates over a specific subset of the dataset.
    """

    def __init__(self, indices):
        """
        Initialize the SubsetRandomSampler with a list of indices.

        Args:
            indices (list): List of indices to be sampled.
        """
        self.epoch = 0  # Initialize the epoch counter
        self.indices = indices  # Store the list of indices to be sampled

    def __iter__(self):
        """
        Return an iterator that yields indices in random order.
        The random permutation is different for each epoch.
        """
        # Use torch.randperm to generate a random permutation of the indices
        return (self.indices[i] for i in torch.randperm(len(self.indices)))

    def __len__(self):
        """
        Return the number of indices in the subset.
        """
        return len(self.indices)  # Return the length of the indices list

    def set_epoch(self, epoch):
        """
        Set the epoch number.

        Args:
            epoch (int): The current epoch number.
        """
        self.epoch = epoch  # Update the epoch counter