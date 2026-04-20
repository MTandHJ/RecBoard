# BERT4Rec

## Usage

Run with full-ranking:

    python main.py --config=configs/xxx.yaml --ranking=full

or with sampled-based ranking:

    python main.py --config=configs/xxx.yaml --ranking=pool

## Hyperparameters

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| --maxlen | int | 50 | Maxlen |
| --num-heads | int | 4 | Num Heads |
| --num-blocks | int | 2 | Num Blocks |
| --embedding-dim | int | 64 | Embedding Dim |
| --mask-ratio | float | 0.3 | Mask Ratio |
| --dropout-rate | float | 0.2 | Dropout Rate |

## Configuration Example

```yaml
# Data
root: ../../data
dataset: Amazon2014Beauty_550_LOU

# Model
maxlen: 50
num_heads: 4
num_blocks: 2
embedding_dim: 64
dropout_rate: 0.2
mask_ratio: 0.2

# Training
epochs: 500
batch_size: 512
optimizer: adam
lr: 5.e-3
weight_decay: 1.e-4

# Evaluation
monitors: [LOSS, HitRate@1, HitRate@5, HitRate@10, HitRate@20, HitRate@50, NDCG@5, NDCG@10, NDCG@20, NDCG@50]
which4best: NDCG@10
```
