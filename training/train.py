import torch
import torch.nn.functional as F

from model.aviyan_model import AviyanModel
from model.config import load_config
from training.dataloader import create_dataloader


def train():
    config = load_config("configs/aviyan_test.yaml")

    model = AviyanModel(config)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4,
    )

    loader = create_dataloader(
        seq_len=32,
        batch_size=2,
        shuffle=True,
    )

    model.train()

    epochs = 3

    for epoch in range(epochs):
        total_loss = 0.0

        for step, batch in enumerate(loader, start=1):
            input_ids = batch["input_ids"]
            labels = batch["labels"]

            logits = model(input_ids)

            loss = F.cross_entropy(
                logits.reshape(-1, config.vocab_size),
                labels.reshape(-1),
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            print(
                f"Epoch {epoch + 1}/{epochs} "
                f"Step {step} "
                f"Loss: {loss.item():.6f}"
            )

        average_loss = total_loss / len(loader)

        print(
            f"Epoch {epoch + 1} complete "
            f"Average Loss: {average_loss:.6f}"
        )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config.__dict__,
        },
        "checkpoints/aviyan_test.pt",
    )

    print("\nTraining complete")
    print("Checkpoint: checkpoints/aviyan_test.pt")


if __name__ == "__main__":
    train()
