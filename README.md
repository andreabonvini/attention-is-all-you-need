# Attention Is All You Need

Purely educational PyTorch implementation of [Attention Is All You Need](https://arxiv.org/abs/1706.03762).

## Setup

```bash
# === [Optional] ====
# Create VirtualEnv
python -m venv .my-venv
# Activate VirtualEnv
.my-venv/Scripts/activate  # On Windows
source .my-venv/bin/activate  # On Linux
# ===================
python install_requirements.py  --device=<CPU|GPU>
python download_data.py
```
## Train model 
We'll use the [Multi30K](https://www.researchgate.net/publication/306094000_Multi30K_Multilingual_English-German_Image_Descriptions) dataset to train a translation model from German to English.

```bash
python train.py --config=configurations/original.json
```

## Test model

...