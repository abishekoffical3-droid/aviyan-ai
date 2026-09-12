import torch
from torch.utils.data import DataLoader

from training.dataset import CausalLanguageModelingDataset


def create_dataloader(
    token_file="data/cleaned/train_tokens.pt",
    seq_len=32,
    batch_size=2,
    shuffle=True,
):
    tokens = torch.load(token_file, weights_only=True)

    dataset = CausalLanguageModelingDataset(
        tokens.tolist(),
        seq_len=seq_len,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
    )


if __name__ == "__main__":
    loader = create_dataloader()

    batch = next(iter(loader))

    print("DataLoader Test")
    print("===============")
    print("Input IDs:", batch["input_ids"].shape)
    print("Labels:", batch["labels"].shape)
