import argparse
import json

import torch

from diffusion_lm.config import DiffusionLMConfig
from diffusion_lm.generation import chat_prompt, diffusion_generate
from diffusion_lm.model import DiffusionTransformerLM
from diffusion_lm.tokenizer import load_tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", default="tokenizer_from_scratch/tokenizer.json")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--diffusion-steps", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=50)
    args = parser.parse_args()

    checkpoint_dir = args.checkpoint.rsplit("/", 1)[0]
    with open(f"{checkpoint_dir}/config.json", "r", encoding="utf-8") as f:
        cfg = DiffusionLMConfig.from_dict(json.load(f))

    tokenizer = load_tokenizer(args.tokenizer)
    model = DiffusionTransformerLM(cfg)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))
    model.to("cuda" if torch.cuda.is_available() else "cpu")

    output, _ = diffusion_generate(
        model, tokenizer, chat_prompt(args.prompt),
        bos_id=tokenizer.bos_token_id,
        mask_id=tokenizer.mask_token_id,
        max_new_tokens=args.max_new_tokens,
        diffusion_steps=min(args.diffusion_steps, cfg.diffusion_steps),
        temperature=args.temperature,
        top_k=args.top_k,
        record_steps=False,
    )
    print(output)


if __name__ == "__main__":
    main()
