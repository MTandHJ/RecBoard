# RUM

[[paper](https://dl.acm.org/doi/10.1145/3159652.3159668)]

## Usage

Run with full-ranking:

    python main.py --config=configs/xxx.yaml --ranking=full

or with sampled-based ranking:

    python main.py --config=configs/xxx.yaml --ranking=pool

## Hyperparameters

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| --maxlen | int | 50 | Maxlen |
| --embedding-dim | int | 64 | Embedding Dim |
| --dropout-rate | float | 0.2 | Dropout Rate |
| --alpha | float | 0.2 | Alpha |

## Configuration Example

```yaml
# Data
root: ../../data
dataset: Amazon2014Tools_550_LOU

# Model
maxlen: 20
embedding_dim: 64
dropout_rate: 0.
alpha: 0.2

# Training
epochs: 300
batch_size: 512
optimizer: adam
lr: 1.e-3
weight_decay: 1.e-6

# Evaluation
monitors: [LOSS, HitRate@1, HitRate@5, HitRate@10, HitRate@20, HitRate@50, NDCG@5, NDCG@10, NDCG@20, NDCG@50]
which4best: NDCG@10
```
