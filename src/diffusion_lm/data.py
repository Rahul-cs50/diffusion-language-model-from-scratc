import random

import torch
from torch.utils.data import DataLoader, IterableDataset

from .tokenizer import format_story


class TokenBlockDataset(IterableDataset):
    def __init__(self, hf_ds, tokenizer, seq_len, shuffle=False, seed=0):
        self.hf_ds = hf_ds
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self):
        indices = list(range(len(self.hf_ds)))
        if self.shuffle:
            rng = random.Random(self.seed)
            rng.shuffle(indices)
        buffer = []
        for idx in indices:
            text = format_story(self.hf_ds[idx]["text"])
            ids = self.tokenizer.encode(text, add_special_tokens=True)
            buffer.extend(ids)
            while len(buffer) >= self.seq_len:
                block = buffer[:self.seq_len]
                buffer = buffer[self.seq_len:]
                yield torch.tensor(block, dtype=torch.long)


def make_loaders(train_ds, val_ds, tokenizer, seq_len, batch_size, pad_id):
    train_blocks = TokenBlockDataset(train_ds, tokenizer, seq_len, shuffle=True, seed=42)
    val_blocks = TokenBlockDataset(val_ds, tokenizer, seq_len, shuffle=False)

    def collate_blocks(batch):
        input_ids = torch.stack(batch, dim=0)
        return {"input_ids": input_ids, "attention_mask": input_ids != pad_id}

    return (
        DataLoader(train_blocks, batch_size=batch_size, collate_fn=collate_blocks),
        DataLoader(val_blocks, batch_size=batch_size, collate_fn=collate_blocks),
    )
