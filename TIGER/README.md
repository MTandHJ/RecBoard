# TIGER

[[reference implementation](https://github.com/EdoardoBotta/RQ-VAE-Recommender)]

## Usage

1. Encode textual item features with `encode_textual_features.py`. The script writes an
   embedding pickle file to the processed dataset directory.

```bash
python encode_textual_features.py
```

2. Quantize item embeddings with Residual K-Means or RQ-VAE. Both pipelines export
   `sid_vocab.json` for downstream T5 training.

```bash
python train_residual_kmeans.py --config=configs/kmeans/Amazon2014Beauty_550_LOU.yaml
```

```bash
python train_rqvae.py --config=configs/rqvae/Amazon2014Beauty_550_LOU.yaml
```

3. Train or evaluate T5 using the exported semantic-ID vocabulary.

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

### Residual K-Means

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--sem-feat-file` | str | `None` | Pickle file containing textual item embeddings. |
| `--num-codebooks` | int | `3` | Number of residual quantization levels. |
| `--num-codewords` | int | `256` | Number of centroids at each level. |
| `--num-iters` | int | `10` | Number of K-Means iterations. |
| `--kmeans-init-method` | str | `random` | SciPy initialization method: `random`, `points`, `++`, or `matrix`. |

### RQ-VAE

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--sem-feat-file` | str | `None` | Pickle file containing textual item embeddings. |
| `--num-codebooks` | int | `3` | Number of residual codebooks. |
| `--num-codewords` | int | `512` | Number of codewords per codebook. |
| `--codebook-dim` | int | `128` | Dimension of each quantized representation. |
| `--apply-shared-codebook` | bool | `False` | Share parameters across residual codebooks. |
| `--sk-epsilons` | str | `0.,0.,0.` | Comma-separated Sinkhorn epsilon values. |
| `--sk-iters` | float | `50` | Number of Sinkhorn iterations. |
| `--hidden-dims` | str | `512,256,128` | Comma-separated encoder hidden sizes. |
| `--commit-weight` | float | `0.25` | Commitment-loss weight. |
| `--dropout-rate` | float | `0.0` | Encoder and decoder dropout rate. |

### T5

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--maxlen` | int | `50` | Maximum item-history length. |
| `--embedding-dim` | int | `128` | T5 hidden dimension. |
| `--attention-size` | int | `64` | T5 key/value dimension. |
| `--intermediate-size` | int | `256` | T5 feed-forward dimension. |
| `--num-heads` | int | `6` | Number of attention heads. |
| `--num-layers` | int | `4` | Number of encoder and decoder layers. |
| `--dropout-rate` | float | `0.1` | T5 dropout rate. |
| `--sid-vocab-file` | str | `sid_vocab.json` | Semantic-ID vocabulary emitted by quantization. |
| `--num-beams` | int | `20` | Beam width for full ranking. Must be greater than `1`. |

## Configuration Example

### Residual K-Means

```yaml
# Data
root: ../../data
dataset: Amazon2014Beauty_550_LOU
tasktag: NEXTITEM
sem_feat_file: sentence-t5-xl_title_categories_brand.pkl

# Model
num_codewords: 256
num_codebooks: 3
num_iters: 100
kmeans_init_method: random

# Training
epochs: 1

# Evaluation
```

### RQ-VAE

```yaml
# Data
root: ../../data
dataset: Amazon2014Beauty_550_LOU
tasktag: NEXTITEM
sem_feat_file: sentence-t5-xl_title_categories_brand.pkl

# Model
num_codewords: 256
num_codebooks: 3
codebook_dim: 32
apply_shared_codebook: false
sk_epsilons: 0.,0.,0.
sk_iters: 50
hidden_dims: 512,256,128
commit_weight: 0.5
dropout_rate: 0.1

# Training
epochs: 20000
batch_size: 1024
optimizer: AdamW
lr: 1.e-3
weight_decay: 0.0

# Evaluation
which4best: PPL#0
```

### T5

```yaml
# Data
root: ../../data
dataset: Amazon2014Beauty_550_LOU

# Model
maxlen: 50
embedding_dim: 128
attention_size: 64
intermediate_size: 256
num_layers: 6
num_heads: 4
dropout_rate: 0.1
sid_vocab_file: ./logs/R-KMeans/Amazon2014Beauty_550_LOU/1117130719/sid_vocab.json

# Training
epochs: 500
optimizer: AdamW
batch_size: 512
lr: 5.e-4
weight_decay: 1.e-3

# Evaluation
num_beams: 20
monitors: [LOSS, HitRate@1, HitRate@5, HitRate@10, HitRate@20, HitRate@50, NDCG@5, NDCG@10, NDCG@20, NDCG@50]
which4best: NDCG@10
```
