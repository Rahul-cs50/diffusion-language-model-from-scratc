import argparse
import json
import os

import matplotlib.pyplot as plt
import torch

from diffusion_lm.config import DiffusionLMConfig
from diffusion_lm.generation import chat_prompt, diffusion_generate
from diffusion_lm.model import DiffusionTransformerLM
from diffusion_lm.tokenizer import load_tokenizer
from diffusion_lm.visualization import make_inference_gif


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", default="tokenizer_from_scratch/tokenizer.json")
    parser.add_argument("--prompt", default="Write a short story about a brave child.")
    parser.add_argument("--output", default="assets/inference.gif")
    args = parser.parse_args()

    checkpoint_dir = os.path.dirname(args.checkpoint)
    with open(os.path.join(checkpoint_dir, "config.json"), "r", encoding="utf-8") as f: cfg = DiffusionLMConfig.from_dict(json.load(f))
    tokenizer = load_tokenizer(args.tokenizer)
    model = DiffusionTransformerLM(cfg)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))
    model.to("cuda" if torch.cuda.is_available() else "cpu")

    output, frames = diffusion_generate(model, tokenizer, chat_prompt(args.prompt), bos_id=tokenizer.bos_token_id, mask_id=tokenizer.mask_token_id, max_new_tokens=128, diffusion_steps=cfg.diffusion_steps, temperature=1.0, top_k=50, record_steps=True)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    make_inference_gif(frames, args.prompt, args.output)

    steps = [step for step, _ in frames]
    mask_ratios = [text.count("█") / max(len(text), 1) for _, text in frames]
    curve_path = os.path.join(os.path.dirname(os.path.dirname(args.output)), "results", "plots", "diffusion_denoising_curve.png")
    os.makedirs(os.path.dirname(curve_path), exist_ok=True)
    plt.figure(figsize=(9, 5)); plt.plot(steps, mask_ratios, marker="o", markersize=3)
    plt.xlabel("Diffusion timestep"); plt.ylabel("Masked token fraction"); plt.title("Progressive Denoising During Generation")
    plt.grid(alpha=0.3); plt.tight_layout(); plt.savefig(curve_path, dpi=200); plt.close()
    print("Final output:"); print(output); print("Saved:", args.output)


if __name__ == "__main__":
    main()
