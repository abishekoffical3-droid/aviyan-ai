import torch
from torch.utils.data import Dataset


class CausalLanguageModelingDataset(Dataset):
    def __init__(self, token_ids, seq_len):
        self.token_ids = token_ids
        self.seq_len = seq_len

    def __len__(self):
        return max(0, len(self.token_ids) - self.seq_len)

    def __getitem__(self, idx):
        chunk = self.token_ids[idx:idx + self.seq_len + 1]

        input_ids = torch.tensor(
            chunk[:-1],
            dtype=torch.long,
        )

        labels = torch.tensor(
            chunk[1:],
            dtype=torch.long,
        )

        return {
            "input_ids": input_ids,
            "labels": labels,
        }
