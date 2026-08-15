import argparse
import json

from datasets import load_dataset

from diffusion_lm.tokenizer import train_tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/one_hour_4090.json")
    parser.add_argument("--output", default="tokenizer_from_scratch")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    train_ds = load_dataset(
        "roneneldan/TinyStories",
        split=f"train[:{cfg['train_examples']}]",
    )

    tokenizer = train_tokenizer(
        train_ds,
        n_examples=cfg["tokenizer_train_examples"],
        vocab_size=cfg["vocab_size"],
        output_dir=args.output,
    )

    print("Saved tokenizer to:", args.output)
    print("Vocabulary size:", tokenizer.get_vocab_size())


if __name__ == "__main__":
    main()
