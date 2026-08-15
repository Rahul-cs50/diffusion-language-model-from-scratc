import torch
import torch.nn.functional as F


def mask_ratio_schedule(t, T: int):
    return t.float() / float(T)


@torch.no_grad()
def corrupt_with_mask(input_ids, attention_mask, t, mask_token_id, T, bos_id, eos_id, pad_id):
    """Corrupt x0 by masking a timestep-dependent fraction of tokens."""
    batch_size, length = input_ids.shape
    ratio = mask_ratio_schedule(t, T).unsqueeze(1)
    can_mask = attention_mask.clone()
    can_mask &= input_ids != bos_id
    can_mask &= input_ids != eos_id
    can_mask &= input_ids != pad_id
    random_values = torch.rand((batch_size, length), device=input_ids.device)
    mask_positions = (random_values < ratio) & can_mask
    noisy = input_ids.clone()
    noisy[mask_positions] = mask_token_id
    labels = torch.full_like(input_ids, -100)
    labels[mask_positions] = input_ids[mask_positions]
    return noisy, labels, mask_positions


def diffusion_loss(model, batch, T, mask_token_id, bos_id, eos_id, pad_id):
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    t = torch.randint(1, T + 1, (input_ids.size(0),), device=input_ids.device)
    noisy_ids, labels, _ = corrupt_with_mask(input_ids, attention_mask, t, mask_token_id, T, bos_id, eos_id, pad_id)
    logits = model(noisy_ids, timesteps=t, attention_mask=attention_mask)
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=-100)
