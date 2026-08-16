# Diffusion Language Model from Scratch

A small **diffusion language model (DLM) built from first principles** on the TinyStories corpus.

Instead of an autoregressive next-token objective, the model learns to reconstruct tokens corrupted by progressively stronger masking. Generation starts with a masked answer region and repeatedly samples, scores, and re-masks tokens until the sequence is denoised.

> **Experiment:** 1-hour RTX 4090 run · TinyStories · 50,000 optimizer steps · ~45M parameters

## Highlights

- Byte-level BPE tokenizer trained from scratch
- Bidirectional Transformer with diffusion-timestep embeddings
- Mask-based discrete diffusion objective
- Progressive unmasking during generation
- Temperature and top-k sampling experiments
- Timestep-wise validation analysis
- Diversity and repetition evaluation
- Terminal-style inference visualization

## Model configuration

| Component | Value |
|---|---:|
| Dataset | TinyStories |
| Training examples | 1,000,000 |
| Tokenizer training examples | 150,000 |
| Vocabulary size | 26,000 |
| Sequence length | 256 |
| Embedding dimension | 512 |
| Transformer layers | 10 |
| Attention heads | 8 |
| Feed-forward dimension | 2,048 |
| Diffusion steps | 128 |
| Parameters | ~45M |
| Optimizer | AdamW |
| Learning rate | 2e-4 |
| Completed steps | 50,000 |
| Recorded training time | ~42.5 minutes |

## Results

### Validation loss across diffusion timesteps

The lowest measured validation loss was **1.5457 at timestep 1**, while the highest was **5.2850 at timestep 50**.

![Validation loss across diffusion timesteps](results/plots/loss_vs_diffusion_timestep.png)

### Training vs validation loss

![Training vs validation loss](results/plots/training_validation_loss.png)

The batch-level losses are noisy because each diffusion-loss evaluation samples a random timestep and mask pattern.

### Sampling diversity vs temperature

![Generation diversity vs temperature](results/plots/diversity_vs_temperature.png)

The tested **temperature 1.1 / top-k 50** configuration produced the highest recorded Distinct-2 score: **0.6544**.

### Repetition vs temperature

![Repetition vs temperature](results/plots/repetition_vs_temperature.png)

The same **temperature 1.1 / top-k 50** configuration produced the lowest recorded bigram repetition rate: **0.3456**.

### Inference

The model generates text through iterative denoising, progressively resolving the masked answer region.

![Inference during generation](assets/inference.gif)

## Generation

Example command after training:

```bash
python scripts/generate.py --checkpoint checkpoints/final/model.pt --tokenizer tokenizer_from_scratch/tokenizer.json --prompt "Explain neural networks in simple terms."
```

The baseline experiment used 5 prompts, 128 generated tokens, 50 diffusion sampling steps, temperature 1.0, and top-k 50. Recorded outputs and metrics are available in `results/data/baseline_generation_results.csv`.

## Reproducibility

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

Train the tokenizer:

```bash
python scripts/train_tokenizer.py
```

Train the model:

```bash
python scripts/train.py --config configs/one_hour_4090.json
```

Run evaluation and analysis:

```bash
python scripts/evaluate.py
python scripts/analyze_timestep.py
python scripts/analyze_sampling.py
```

## Repository structure

```text
diffusion-language-model-from-scratch/
├── README.md
├── requirements.txt
├── pyproject.toml
├── .gitignore
├── .gitattributes
├── configs/
│   └── one_hour_4090.json
├── src/diffusion_lm/
│   ├── config.py
│   ├── tokenizer.py
│   ├── data.py
│   ├── model.py
│   ├── diffusion.py
│   ├── generation.py
│   ├── metrics.py
│   └── visualization.py
├── scripts/
│   ├── train_tokenizer.py
│   ├── train.py
│   ├── generate.py
│   ├── evaluate.py
│   ├── analyze_timestep.py
│   ├── analyze_training_loss.py
│   ├── analyze_sampling.py
│   └── render_inference.py
├── tokenizer_from_scratch/
│   └── tokenizer.json
├── checkpoints/
│   └── final/
│       └── model.pt
├── results/
│   ├── data/
│   │   ├── baseline_generation_results.csv
│   │   └── sampling_results.csv
│   └── plots/
│       ├── loss_vs_diffusion_timestep.png
│       ├── training_validation_loss.png
│       ├── diversity_vs_temperature.png
│       └── repetition_vs_temperature.png
└── assets/
    └── inference.gif
```

## Checkpoint

The trained `model.pt` is tracked with **Git LFS**. The repository also contains the tokenizer and checkpoint metadata required to reproduce inference.

## Limitations

- TinyStories is small and stylistically narrow.
- The model is ~45M parameters.
- Training was constrained to roughly one hour on an RTX 4090.
- Generation quality is not comparable with modern production LLMs.
- Diversity and repetition metrics do not measure factuality, coherence, or instruction following.

## License

MIT
