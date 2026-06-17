# R3-VAE

[[official-code](https://github.com/wwqq/R3-VAE)] [[paper](https://arxiv.org/abs/2604.11440)]

## Usage

1. Encode textual item features.

```bash
python encode_textual_features.py
```

2. Train the R3-VAE tokenizer and export `sid_vocab.json`.

```bash
python train_r3vae.py --config=configs/r3vae/Amazon2014Beauty_550_LOU.yaml
```

3. Train or evaluate the TIGER T5 recommender with the exported vocabulary.

```bash
python train_t5.py --config=configs/t5/Amazon2014Beauty_550_LOU.yaml \
    --sid-vocab-file=/path/to/sid_vocab.json --ranking=full
```

```bash
python train_t5.py --config=configs/t5/Amazon2014Beauty_550_LOU.yaml \
    --sid-vocab-file=/path/to/sid_vocab.json --ranking=pool
```

## Hyperparameters

### R3-VAE Tokenizer

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--sem-feat-file` | str | `None` | Pickle file containing item semantic features. |
| `--num-codebooks` | int | `3` | Number of residual rating codebooks. |
| `--num-codewords` | int | `256` | Number of codewords per codebook. |
| `--rec-loss-weight` | float | `1.0` | Reconstruction loss weight. |
| `--sc-loss-weight` | float | `0.1` | Semantic cohesion loss weight. |
| `--pd-loss-weight` | float | `0.05` | Preference discrimination loss weight. |
| `--pd-temperature` | float | `2.0` | Temperature for preference discrimination loss. |

## Configuration Example

```yaml
# Data
root: ../../data
dataset: Amazon2014Beauty_550_LOU
tasktag: NEXTITEM
sem_feat_file: sentence-t5-xl_title_categories_brand.pkl

# Model
num_codewords: 256
num_codebooks: 3
rec_loss_weight: 1.0
sc_loss_weight: 0.1
pd_loss_weight: 0.05
pd_temperature: 2.0

# Training
epochs: 10000
batch_size: 1024
optimizer: AdamW
lr: 1.e-3
weight_decay: 0.0

# Evaluation
eval_freq: 100
which4best: PPL
```
