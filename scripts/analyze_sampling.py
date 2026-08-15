import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd
import torch

from diffusion_lm.generation import chat_prompt, diffusion_generate
from diffusion_lm.metrics import calculate_metrics

PROMPTS = [
    "Tell me about robotics and autonomous drones.",
    "Explain why mathematics is important in machine learning.",
    "Tell me about a person interested in LLMs, mathematics and robotics.",
]
CONFIGS = [(0.7, 20), (0.8, 20), (0.9, 50), (1.0, 50), (1.1, 50), (1.2, 100)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", default="tokenizer_from_scratch/tokenizer.json")
    parser.add_argument("--output-csv", default="results/data/sampling_results.csv")
    parser.add_argument("--plot-dir", default="results/plots")
    args = parser.parse_args()

    import json
    from diffusion_lm.config import DiffusionLMConfig
    from diffusion_lm.model import DiffusionTransformerLM
    from diffusion_lm.tokenizer import load_tokenizer

    checkpoint_dir = os.path.dirname(args.model)
    with open(os.path.join(checkpoint_dir, "config.json"), "r", encoding="utf-8") as f:
        cfg = DiffusionLMConfig.from_dict(json.load(f))
    tokenizer = load_tokenizer(args.tokenizer)
    model = DiffusionTransformerLM(cfg)
    model.load_state_dict(torch.load(args.model, map_location="cpu", weights_only=True))
    model.to("cuda" if torch.cuda.is_available() else "cpu")

    rows = []
    for temperature, top_k in CONFIGS:
        outputs = []
        for prompt in PROMPTS:
            output, _ = diffusion_generate(
                model, tokenizer, chat_prompt(prompt),
                bos_id=tokenizer.bos_token_id,
                mask_id=tokenizer.mask_token_id,
                max_new_tokens=100,
                diffusion_steps=50,
                temperature=temperature,
                top_k=top_k,
                record_steps=False,
            )
            outputs.append(output)
        rows.append({"temperature": temperature, "top_k": top_k, **calculate_metrics("\n".join(outputs))})

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    os.makedirs(args.plot_dir, exist_ok=True)
    df.to_csv(args.output_csv, index=False)

    for top_k in df["top_k"].unique():
        subset = df[df["top_k"] == top_k]
        plt.plot(subset["temperature"], subset["Distinct-2"], marker="o", label=f"top-k={top_k}")
    plt.xlabel("Temperature")
    plt.ylabel("Distinct-2")
    plt.title("Generation Diversity vs Temperature")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(args.plot_dir, "diversity_vs_temperature.png"), dpi=200)
    plt.close()

    for top_k in df["top_k"].unique():
        subset = df[df["top_k"] == top_k]
        plt.plot(subset["temperature"], subset["Repetition-2"], marker="o", label=f"top-k={top_k}")
    plt.xlabel("Temperature")
    plt.ylabel("Bigram repetition rate")
    plt.title("Repetition vs Temperature")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(args.plot_dir, "repetition_vs_temperature.png"), dpi=200)
    plt.close()
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
