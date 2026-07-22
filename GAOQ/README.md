# GAOQ

[[official-code](https://github.com/FuCongResearchSquad/ReSID)] [[paper](https://arxiv.org/abs/2602.02338)]

This directory ports only ReSID's Globally Aligned Orthogonal Quantization
(GAOQ) tokenizer into RecBoard. It uses TIGER-style precomputed item features
and exports `sid_vocab.json` for the included T5 recommender.

## Usage

1. Quantize the precomputed item features with GAOQ.

```bash
python train_gaoq.py --config=configs/gaoq/Amazon2014Beauty_550_LOU.yaml
```

2. Train or evaluate T5 with the exported semantic-ID vocabulary.

Run with full-ranking constrained beam generation:

```bash
python train_t5.py --config=configs/t5/Amazon2014Beauty_550_LOU.yaml \
    --sid-vocab-file=/path/to/sid_vocab.json --ranking=full
```

Run with sampled-pool candidate scoring:

```bash
python train_t5.py --config=configs/t5/Amazon2014Beauty_550_LOU.yaml \
    --sid-vocab-file=/path/to/sid_vocab.json --ranking=pool
```

## Hyperparameters

### GAOQ

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--sem-feat-file` | str | `None` | Pickle file containing item features. |
| `--num-codewords` | str | `96,192` | Comma-separated codeword counts for fixed clustering levels. |
| `--l2norm` | bool | `False` | L2-normalize item features before GAOQ. |
| `--use-balancedkmeans` | bool | `True` | Use constrained balanced K-Means. |

The number of output codebooks is `len(num_codewords) + 1`. Each configured
level after the first uses the same number of local clusters and global
orthogonal anchors. The final codebook size is inferred from the maximum
collision count and uniquely identifies items sharing the configured prefix.
Every configured count and the inferred final count must not exceed the item
feature dimension.

### T5

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--maxlen` | int | `20` | Maximum item-history length. |
| `--embedding-dim` | int | `128` | T5 hidden dimension. |
| `--attention-size` | int | `64` | T5 key/value dimension. |
| `--intermediate-size` | int | `256` | T5 feed-forward dimension. |
| `--num-heads` | int | `4` | Number of attention heads. |
| `--num-layers` | int | `6` | Number of encoder and decoder layers. |
| `--dropout-rate` | float | `0.1` | T5 dropout rate. |
| `--sid-vocab-file` | str | `sid_vocab.json` | GAOQ semantic-ID vocabulary. |
| `--num-beams` | int | `20` | Beam width for full ranking. |

## Configuration Example

### GAOQ

```yaml
# Data
root: ../../data
dataset: Amazon2014Beauty_550_LOU
tasktag: NEXTITEM
sem_feat_file: sentence-t5-xl_title_categories_brand.pkl

# Model
num_codewords: 96,192
l2norm: false
use_balancedkmeans: true

# Training
epochs: 1

# Evaluation
which4best: PPL
```

### T5

```yaml
# Data
root: ../../data
dataset: Amazon2014Beauty_550_LOU

# Model
maxlen: 20
embedding_dim: 128
attention_size: 64
intermediate_size: 256
num_layers: 6
num_heads: 4
dropout_rate: 0.1
sid_vocab_file: sid_vocab.json

# Training
epochs: 200
optimizer: AdamW
batch_size: 512
lr: 5.e-4
weight_decay: 1.e-3

# Evaluation
num_beams: 20
apply_constrained_beam_search: true
monitors: [LOSS, HitRate@1, HitRate@5, HitRate@10, HitRate@20, NDCG@5, NDCG@10, NDCG@20]
which4best: NDCG@10
```
