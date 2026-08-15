import argparse
import json
import os

import pandas as pd
import torch

from diffusion_lm.config import DiffusionLMConfig
from diffusion_lm.generation import chat_prompt, diffusion_generate
from diffusion_lm.metrics import calculate_metrics
from diffusion_lm.model import DiffusionTransformerLM
from diffusion_lm.tokenizer import load_tokenizer

PROMPTS = [
    "Tell me about robotics and autonomous drones.",
    "Explain why mathematics is important in machine learning.",
    "Tell me about a person interested in LLMs, mathematics and robotics.",
    "What is artificial intelligence?",
    "Explain neural networks in simple terms.",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", default="tokenizer_from_scratch/tokenizer.json")
    parser.add_argument("--output", default="results/data/baseline_generation_results.csv")
    args = parser.parse_args()

    checkpoint_dir = os.path.dirname(args.checkpoint)
    with open(os.path.join(checkpoint_dir, "config.json"), "r", encoding="utf-8") as f:
        cfg = DiffusionLMConfig.from_dict(json.load(f))

    tokenizer = load_tokenizer(args.tokenizer)
    model = DiffusionTransformerLM(cfg)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))
    model.to("cuda" if torch.cuda.is_available() else "cpu")

    rows = []
    for prompt in PROMPTS:
        output, _ = diffusion_generate(
            model, tokenizer, chat_prompt(prompt),
            bos_id=tokenizer.bos_token_id,
            mask_id=tokenizer.mask_token_id,
            max_new_tokens=128,
            diffusion_steps=50,
            temperature=1.0,
            top_k=50,
            record_steps=False,
        )
        rows.append({"prompt": prompt, "generation": output, **calculate_metrics(output)})

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df.to_csv(args.output, index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
