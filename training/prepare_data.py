from pathlib import Path

from tokenizer.load_tokenizer import load_tokenizer


DATA_DIR = Path("data/raw")
OUTPUT_DIR = Path("data/cleaned")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "train_tokens.pt"


def get_text_files():
    return sorted(DATA_DIR.rglob("*.txt"))


def main():
    tokenizer = load_tokenizer()
    files = get_text_files()

    if not files:
        raise RuntimeError("No .txt files found in data/raw")

    all_ids = []

    for path in files:
        text = path.read_text(encoding="utf-8").strip()

        if not text:
            continue

        encoded = tokenizer.encode(text)
        all_ids.extend(encoded.ids)

        print(f"Processed: {path}")
        print(f"Tokens: {len(encoded.ids)}")

    if not all_ids:
        raise RuntimeError("No tokens generated")

    import torch

    tokens = torch.tensor(all_ids, dtype=torch.long)
    torch.save(tokens, OUTPUT_FILE)

    print("\nData preparation complete")
    print("=========================")
    print(f"Total tokens: {len(tokens)}")
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
