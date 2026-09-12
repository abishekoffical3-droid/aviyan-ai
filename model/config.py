from dataclasses import dataclass
import yaml


@dataclass
class AviyanConfig:
    name: str
    vocab_size: int
    hidden_size: int
    num_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    intermediate_size: int
    max_position_embeddings: int
    activation: str
    normalization: str
    attention_dropout: float
    hidden_dropout: float
    use_bias: bool


def load_config(path: str = "configs/aviyan_1b.yaml") -> AviyanConfig:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    model = data["model"]

    return AviyanConfig(
        name=model["name"],
        vocab_size=model["vocab_size"],
        hidden_size=model["hidden_size"],
        num_layers=model["num_layers"],
        num_attention_heads=model["num_attention_heads"],
        num_key_value_heads=model["num_key_value_heads"],
        intermediate_size=model["intermediate_size"],
        max_position_embeddings=model["max_position_embeddings"],
        activation=model["activation"],
        normalization=model["normalization"],
        attention_dropout=model["attention_dropout"],
        hidden_dropout=model["hidden_dropout"],
        use_bias=model["use_bias"],
    )


if __name__ == "__main__":
    config = load_config()

    print("AVIYAN AI Configuration")
    print("=======================")
    print(f"Model:          {config.name}")
    print(f"Vocabulary:     {config.vocab_size}")
    print(f"Hidden size:    {config.hidden_size}")
    print(f"Layers:         {config.num_layers}")
    print(f"Attention heads:{config.num_attention_heads}")
    print(f"FFN size:       {config.intermediate_size}")
    print(f"Context length: {config.max_position_embeddings}")
