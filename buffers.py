from collections import defaultdict, deque
from functools import partial

import numpy as np
import torch


class SimpleDictReplayBuffer:
    def __init__(
        self,
        buffer_size,
        device="cpu",
    ):
        """
        Replay Buffer using a dictionary-like interface for storage.

        :param buffer_size: Maximum number of elements in the buffer.
        :param device: Device to store the tensors (e.g., 'cpu' or 'cuda').
        """
        self.buffer_size = buffer_size
        self.device = torch.device(device)

        # Dictionary to store data buffers
        self.buffers = defaultdict(partial(deque, maxlen=self.buffer_size))

    def add(self, data_dict):
        """
        Add a new experience to the buffer.

        :param data_dict: Dictionary containing key-value pairs to store.
        """
        for key, value in data_dict.items():
            # Append the value to the deque
            self.buffers[key].append(value)

    def size(self):
        """
        Get the current size of the buffer.

        :return: Number of elements in the buffer.

        Size is computed from the first key
        """
        key = next(iter(self.buffers.keys()), None)
        return 0 if key is None else len(self.buffers[key])

    def sample(self, batch_size):
        """
        Sample a batch of experiences from the buffer.

        :param batch_size: Number of samples to return.
        :return: A dictionary containing sampled data.
        """
        max_index = len(next(iter(self.buffers.values()), []))

        if batch_size > max_index:
            raise ValueError("Batch size exceeds the number of stored elements.")

        # Randomly sample indices
        indices = np.random.choice(max_index, batch_size, replace=False)

        sampled_data = {}
        for key, buffer in self.buffers.items():
            samples = [buffer[idx] for idx in indices]

            # get collate type
            # if array-like, then stack or make it nested, otherwise keep as list
            example = samples[0]
            if isinstance(example, (float, torch.Tensor, np.ndarray)):
                if isinstance(example, (np.ndarray, float)):
                    samples = [
                        torch.tensor(s, dtype=torch.float32).to(self.device)
                        for s in samples
                    ]
                if all(s.shape == example.shape for s in samples):
                    # stack if all shapes are the same
                    samples = torch.stack(samples).to(self.device)
                elif example.dim() == 2:
                    # use nested tensor
                    samples = torch.nested.nested_tensor(samples).to(self.device)

            sampled_data[key] = samples

        return sampled_data
