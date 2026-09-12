import yaml


with open("configs/aviyan_1b.yaml", "r") as f:
    config = yaml.safe_load(f)

m = config["model"]

vocab = m["vocab_size"]
hidden = m["hidden_size"]
layers = m["num_layers"]
intermediate = m["intermediate_size"]

# Token embedding + output projection
embedding_params = vocab * hidden
lm_head_params = vocab * hidden

# One Transformer block:
# Q, K, V projections + output projection
attention_params = 4 * hidden * hidden

# SwiGLU feed-forward network
ffn_params = 3 * hidden * intermediate

# Two RMSNorm parameter vectors
norm_params = 2 * hidden

block_params = attention_params + ffn_params + norm_params
transformer_params = layers * block_params

total = embedding_params + lm_head_params + transformer_params

print("AVIYAN-1B Parameter Count")
print("=========================")
print(f"Embedding:      {embedding_params:,}")
print(f"Transformer:    {transformer_params:,}")
print(f"LM Head:        {lm_head_params:,}")
print(f"TOTAL:          {total:,}")
print(f"TOTAL (B):      {total / 1_000_000_000:.3f} B")
