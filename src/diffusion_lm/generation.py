import math

import torch
import torch.nn.functional as F


def chat_prompt(user_msg: str, system_msg: str | None = None) -> str:
    parts = []
    if system_msg:
        parts.append(f"<|system|>\n{system_msg}\n")
    parts.append(f"<|user|>\n{user_msg}\n")
    parts.append("<|assistant|>\n")
    return "".join(parts)


@torch.no_grad()
def diffusion_generate(model, tokenizer, prompt_text, bos_id, mask_id, max_new_tokens=128, diffusion_steps=64, temperature=1.0, top_k=0, record_steps=True):
    model.eval()
    device = next(model.parameters()).device
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    prompt_ids = [bos_id] + prompt_ids
    prompt_ids = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)
    prompt_len = prompt_ids.size(1)
    length = min(model.pos_emb.num_embeddings, prompt_len + max_new_tokens)
    generation_len = length - prompt_len

    x = torch.full((1, length), mask_id, dtype=torch.long, device=device)
    x[:, :prompt_len] = prompt_ids[:, :prompt_len]
    fixed = torch.zeros((1, length), dtype=torch.bool, device=device)
    fixed[:, :prompt_len] = True
    attention_mask = torch.ones((1, length), dtype=torch.bool, device=device)
    frames = []

    def sample_from_logits(logits):
        if temperature != 1.0:
            logits = logits / temperature
        if top_k and top_k > 0:
            k = min(top_k, logits.size(-1))
            topk_values, topk_indices = torch.topk(logits, k=k, dim=-1)
            filtered = torch.full_like(logits, float("-inf"))
            filtered.scatter_(-1, topk_indices, topk_values)
            logits = filtered
        probs = F.softmax(logits, dim=-1)
        sampled = torch.multinomial(probs.view(-1, probs.size(-1)), 1).view(1, length)
        sampled_prob = probs.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)
        return sampled, sampled_prob

    for step in range(diffusion_steps, 0, -1):
        t = torch.tensor([step], device=device, dtype=torch.long)
        logits = model(x, timesteps=t, attention_mask=attention_mask)
        sampled, confidence = sample_from_logits(logits)
        x[~fixed] = sampled[~fixed]

        next_ratio = float(step - 1) / float(diffusion_steps)
        target_masks = int(math.ceil(generation_len * next_ratio))
        generation_positions = torch.arange(length, device=device) >= prompt_len
        candidate_indices = torch.where(generation_positions & (~fixed[0]))[0]
        if target_masks > 0 and candidate_indices.numel() > 0:
            candidate_confidence = confidence[0, candidate_indices]
            k = min(target_masks, candidate_indices.numel())
            _, low_confidence = torch.topk(candidate_confidence, k=k, largest=False)
            x[0, candidate_indices[low_confidence]] = mask_id
        if record_steps:
            decoded = tokenizer.decode(x[0].tolist()).replace("[MASK]", "█")
            frames.append((step, decoded))

    final = tokenizer.decode(x[0].tolist())
    model.train()
    return final, frames
