import argparse
import json
import os

import matplotlib.pyplot as plt
import torch
from datasets import load_dataset

from diffusion_lm.config import DiffusionLMConfig
from diffusion_lm.data import make_loaders
from diffusion_lm.diffusion import diffusion_loss
from diffusion_lm.model import DiffusionTransformerLM
from diffusion_lm.tokenizer import load_tokenizer


def collect(model, loader, cfg, ids, device, max_batches):
    values = []; model.eval()
    with torch.no_grad():
        for idx, batch in enumerate(loader):
            if idx >= max_batches: break
            batch = {k: v.to(device) for k, v in batch.items()}
            values.append(diffusion_loss(model, batch, cfg.diffusion_steps, ids["mask"], ids["bos"], ids["eos"], ids["pad"]).item())
    return values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", default="tokenizer_from_scratch/tokenizer.json")
    parser.add_argument("--output", default="results/plots/training_validation_loss.png")
    parser.add_argument("--max-batches", type=int, default=100)
    args = parser.parse_args()

    checkpoint_dir = os.path.dirname(args.checkpoint)
    with open(os.path.join(checkpoint_dir, "config.json"), "r", encoding="utf-8") as f: cfg = DiffusionLMConfig.from_dict(json.load(f))
    tokenizer = load_tokenizer(args.tokenizer)
    train_ds = load_dataset("roneneldan/TinyStories", split="train[:1000000]")
    val_ds = load_dataset("roneneldan/TinyStories", split="validation[:10000]")
    train_loader, val_loader = make_loaders(train_ds, val_ds, tokenizer, cfg.seq_len, batch_size=32, pad_id=tokenizer.pad_token_id)
    model = DiffusionTransformerLM(cfg)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))
    device = "cuda" if torch.cuda.is_available() else "cpu"; model.to(device)
    ids = {"pad": tokenizer.pad_token_id, "mask": tokenizer.mask_token_id, "bos": tokenizer.bos_token_id, "eos": tokenizer.eos_token_id}
    train_loss = collect(model, train_loader, cfg, ids, device, args.max_batches)
    val_loss = collect(model, val_loader, cfg, ids, device, args.max_batches)

    plt.figure(figsize=(9, 5)); plt.plot(train_loss, label="Training Loss"); plt.plot(val_loss, label="Validation Loss")
    plt.xlabel("Batch"); plt.ylabel("Diffusion Loss"); plt.title("Training vs Validation Diffusion Loss")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    os.makedirs(os.path.dirname(args.output), exist_ok=True); plt.savefig(args.output, dpi=200, bbox_inches="tight"); plt.close()
    print("Mean training loss:", sum(train_loss) / len(train_loss)); print("Mean validation loss:", sum(val_loss) / len(val_loss))


if __name__ == "__main__":
    main()
