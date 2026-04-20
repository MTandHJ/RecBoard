# STOSA

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
| --num-blocks | int | 1 | Num Blocks |
| --num-heads | int | 4 | Num Heads |
| --hidden-dropout-rate | float | 0.3 | Hidden Dropout Rate |
| --attn-dropout-rate | float | 0.0 | Attn Dropout Rate |
| --distance-metric | str | "wasserstein" | Distance Metric |
| --pvn-weight | float | 0.005 | the weight for postives versus negatives |

## Configuration Example

```yaml
# Data
root: ../../data
dataset: Amazon2014Beauty_550_LOU

# Model
maxlen: 50
num_blocks: 1
num_heads: 4
embedding_dim: 64
attn_dropout_rate: 0.1
hidden_dropout_rate: 0.3
pvn_weight: 0.001
distance_metric: wasserstein

# Training
epochs: 500
batch_size: 512
optimizer: adam
lr: 1.e-3
weight_decay: 0.

# Evaluation
monitors: [LOSS, HitRate@1, HitRate@5, HitRate@10, HitRate@20, HitRate@50, NDCG@5, NDCG@10, NDCG@20, NDCG@50]
which4best: NDCG@10
```
