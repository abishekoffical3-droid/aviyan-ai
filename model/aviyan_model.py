import torch
import torch.nn as nn

from model.config import load_config
from model.embeddings import TokenEmbedding
from model.transformer import AviyanTransformer


class AviyanModel(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.embedding = TokenEmbedding(
            config.vocab_size,
            config.hidden_size,
        )

        self.transformer = AviyanTransformer(
            num_layers=config.num_layers,
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            intermediate_size=config.intermediate_size,
            dropout=config.hidden_dropout,
        )

        self.lm_head = nn.Linear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
        )

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        x = self.transformer(x)
        logits = self.lm_head(x)

        return logits


if __name__ == "__main__":
    config = load_config("configs/aviyan_test.yaml")
    model = AviyanModel(config)

    input_ids = torch.randint(
        0,
        config.vocab_size,
        (1, 8),
    )

    logits = model(input_ids)

    print("AVIYAN Model Test")
    print("=================")
    print("Input:", input_ids.shape)
    print("Output:", logits.shape)

    parameters = sum(p.numel() for p in model.parameters())

    print(f"Parameters: {parameters:,}")
    print(f"Parameters (B): {parameters / 1_000_000_000:.3f}")
