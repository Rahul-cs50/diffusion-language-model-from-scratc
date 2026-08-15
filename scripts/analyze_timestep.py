import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset

from diffusion_lm.config import DiffusionLMConfig
from diffusion_lm.data import make_loaders
from diffusion_lm.diffusion import corrupt_with_mask
from diffusion_lm.model import DiffusionTransformerLM
from diffusion_lm.tokenizer import load_tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", default="tokenizer_from_scratch/tokenizer.json")
    parser.add_argument("--output", default="results/plots/loss_vs_diffusion_timestep.png")
    parser.add_argument("--max-batches", type=int, default=20)
    args = parser.parse_args()

    checkpoint_dir = os.path.dirname(args.checkpoint)
    with open(os.path.join(checkpoint_dir, "config.json"), "r", encoding="utf-8") as f:
        cfg = DiffusionLMConfig.from_dict(json.load(f))

    tokenizer = load_tokenizer(args.tokenizer)
    val_ds = load_dataset("roneneldan/TinyStories", split="validation[:10000]")
    _, val_loader = make_loaders(val_ds, val_ds, tokenizer, cfg.seq_len, batch_size=32, pad_id=tokenizer.pad_token_id)

    model = DiffusionTransformerLM(cfg)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device); model.eval()

    losses, stds = [], []
    with torch.no_grad():
        for t_value in range(1, cfg.diffusion_steps + 1):
            batch_losses = []
            for batch_idx, batch in enumerate(val_loader):
                if batch_idx >= args.max_batches:
                    break
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                t = torch.full((input_ids.size(0),), t_value, device=device, dtype=torch.long)
                noisy, labels, _ = corrupt_with_mask(input_ids, attention_mask, t, tokenizer.mask_token_id, cfg.diffusion_steps, tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id)
                logits = model(noisy, timesteps=t, attention_mask=attention_mask)
                batch_losses.append(F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=-100).item())
            losses.append(np.mean(batch_losses)); stds.append(np.std(batch_losses))

    timesteps = np.arange(1, cfg.diffusion_steps + 1)
    plt.figure(figsize=(10, 5))
    plt.plot(timesteps, losses, marker="o", markersize=3, label="Validation Loss")
    plt.fill_between(timesteps, np.array(losses) - np.array(stds), np.array(losses) + np.array(stds), alpha=0.2)
    plt.xlabel("Diffusion Timestep"); plt.ylabel("Cross-Entropy Loss")
    plt.title("Validation Loss Across Diffusion Timesteps")
    plt.grid(alpha=0.3); plt.legend(); plt.tight_layout()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    plt.savefig(args.output, dpi=200, bbox_inches="tight"); plt.close()

    print("Minimum loss:", min(losses))
    print("Maximum loss:", max(losses))
    print("Best timestep:", int(np.argmin(losses) + 1))
    print("Worst timestep:", int(np.argmax(losses) + 1))


if __name__ == "__main__":
    main()
