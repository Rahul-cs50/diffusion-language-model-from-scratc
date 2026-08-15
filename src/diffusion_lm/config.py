from dataclasses import dataclass


@dataclass
class DiffusionLMConfig:
    vocab_size: int
    seq_len: int
    d_model: int
    n_layers: int
    n_heads: int
    d_ff: int
    dropout: float
    diffusion_steps: int

    @classmethod
    def from_dict(cls, data):
        return cls(**{k: data[k] for k in cls.__dataclass_fields__})

    def to_dict(self):
        return self.__dict__.copy()
