import argparse
import json
import os
import time

import numpy as np
import torch
from accelerate import Accelerator
from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import get_cosine_schedule_with_warmup

from diffusion_lm.config import DiffusionLMConfig
from diffusion_lm.data import make_loaders
from diffusion_lm.diffusion import diffusion_loss
from diffusion_lm.model import DiffusionTransformerLM
from diffusion_lm.tokenizer import load_tokenizer


def evaluate(model, loader, cfg, token_ids, n_batches, accelerator):
    model.eval(); losses = []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= n_batches: break
            loss = diffusion_loss(model, batch, cfg.diffusion_steps, token_ids["mask"], token_ids["bos"], token_ids["eos"], token_ids["pad"])
            losses.append(accelerator.gather(loss.detach().float().reshape(1)).cpu())
    model.train()
    return torch.cat(losses).mean().item() if losses else float("nan")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/one_hour_4090.json")
    parser.add_argument("--tokenizer", default="tokenizer_from_scratch/tokenizer.json")
    parser.add_argument("--output", default="checkpoints/final")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f: run_cfg = json.load(f)
    if not os.path.exists(args.tokenizer): raise FileNotFoundError("Tokenizer not found. Run scripts/train_tokenizer.py first.")

    train_ds = load_dataset("roneneldan/TinyStories", split=f"train[:{run_cfg['train_examples']}]")
    val_ds = load_dataset("roneneldan/TinyStories", split=f"validation[:{run_cfg['val_examples']}]")
    tokenizer = load_tokenizer(args.tokenizer)
    token_ids = {"pad": tokenizer.pad_token_id, "mask": tokenizer.mask_token_id, "bos": tokenizer.bos_token_id, "eos": tokenizer.eos_token_id}
    model_cfg = DiffusionLMConfig(vocab_size=run_cfg["vocab_size"], seq_len=run_cfg["seq_len"], d_model=run_cfg["d_model"], n_layers=run_cfg["n_layers"], n_heads=run_cfg["n_heads"], d_ff=run_cfg["d_ff"], dropout=run_cfg["dropout"], diffusion_steps=run_cfg["diffusion_steps"])
    model = DiffusionTransformerLM(model_cfg)
    train_loader, val_loader = make_loaders(train_ds, val_ds, tokenizer, model_cfg.seq_len, run_cfg["batch_size"], token_ids["pad"])
    accelerator = Accelerator(mixed_precision=("bf16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "fp16"))
    optimizer = torch.optim.AdamW(model.parameters(), lr=run_cfg["learning_rate"], weight_decay=run_cfg["weight_decay"])
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=run_cfg["warmup_steps"], num_training_steps=run_cfg["train_steps"])
    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(model, optimizer, train_loader, val_loader, scheduler)

    max_seconds = run_cfg["max_train_seconds"]; grad_accum = run_cfg["grad_accum"]; clip_norm = run_cfg["gradient_clip_norm"]
    model.train(); train_start = time.time(); actual_steps = 0; running = []; train_iter = iter(train_loader)
    progress = tqdm(range(run_cfg["train_steps"]), disable=not accelerator.is_main_process)
    for step in progress:
        if time.time() - train_start >= max_seconds:
            if accelerator.is_main_process: print("Time budget reached; stopping safely.")
            break
        try: batch = next(train_iter)
        except StopIteration: train_iter = iter(train_loader); batch = next(train_iter)
        loss = diffusion_loss(model, batch, model_cfg.diffusion_steps, token_ids["mask"], token_ids["bos"], token_ids["eos"], token_ids["pad"]) / grad_accum
        accelerator.backward(loss)
        if (step + 1) % grad_accum == 0:
            accelerator.clip_grad_norm_(model.parameters(), clip_norm); optimizer.step(); scheduler.step(); optimizer.zero_grad()
        running.append(loss.item() * grad_accum); actual_steps = step + 1
        if (step + 1) % 50 == 0 and accelerator.is_main_process: progress.set_description(f"loss={np.mean(running[-50:]):.4f} lr={scheduler.get_last_lr()[0]:.2e}")
        if (step + 1) % 500 == 0 and accelerator.is_main_process:
            print(f"\nStep {step + 1} | val_loss ~ {evaluate(model, val_loader, model_cfg, token_ids, 10, accelerator):.4f}")

    if accelerator.is_main_process:
        os.makedirs(args.output, exist_ok=True)
        unwrapped = accelerator.unwrap_model(model)
        torch.save(unwrapped.state_dict(), os.path.join(args.output, "model.pt"))
        with open(os.path.join(args.output, "config.json"), "w", encoding="utf-8") as f: json.dump(model_cfg.to_dict(), f, indent=2)
        tokenizer.save_pretrained(os.path.join(args.output, "tokenizer"))
        elapsed = time.time() - train_start
        with open(os.path.join(args.output, "training_summary.json"), "w", encoding="utf-8") as f: json.dump({"steps_completed": actual_steps, "elapsed_seconds": elapsed, "run_mode": run_cfg.get("run_mode")}, f, indent=2)
        print(f"Training complete: {actual_steps} steps in {elapsed / 60:.1f} minutes")


if __name__ == "__main__":
    main()
