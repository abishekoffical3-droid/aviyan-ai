import torch.nn as nn

from model.blocks import RMSNorm, TransformerBlock


class AviyanTransformer(nn.Module):
    def __init__(
        self,
        num_layers,
        hidden_size,
        num_heads,
        intermediate_size,
        dropout=0.0,
    ):
        super().__init__()

        self.layers = nn.ModuleList([
            TransformerBlock(
                hidden_size=hidden_size,
                num_heads=num_heads,
                intermediate_size=intermediate_size,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        self.norm = RMSNorm(hidden_size)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        return self.norm(x)
