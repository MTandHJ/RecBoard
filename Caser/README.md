# Caser

[[official-code](https://github.com/graytowne/caser_pytorch)]

## Usage

Run with full-ranking:

    python main.py --config=configs/xxx.yaml --ranking=full

or with sampled-based ranking:

    python main.py --config=configs/xxx.yaml --ranking=pool

## Hyperparameters

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| --maxlen | int | 5 | Maxlen |
| --embedding-dim | int | 64 | Embedding Dim |
| --dropout-rate | float | 0.5 | Dropout Rate |
| --num-vert | int | 4 | number of vertical filters |
| --num-horiz | int | 16 | number of horizontal filters |
| --num-poss | int | 3 | number of positive samples |
| --num-negs | int | 3 | number of negative samples |

## Configuration Example

```yaml
# Data
root: ../../data
dataset: Amazon2014Beauty_550_LOU

# Model
maxlen: 5
embedding_dim: 64
num_vert: 4
num_horiz: 16
dropout_rate: 0.7
num_poss: 3
num_negs: 2

# Training
epochs: 300
batch_size: 512
optimizer: adam
lr: 1.e-3
weight_decay: 0.

# Evaluation
monitors: [LOSS, HitRate@1, HitRate@5, HitRate@10, NDCG@5, NDCG@10]
which4best: NDCG@10
```
