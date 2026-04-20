# UltraGCN

[[official-code](https://github.com/xue-pai/UltraGCN)]

## Usage

Run with full-ranking:

    python main.py --config=configs/xxx.yaml --ranking=full

or with sampled-based ranking:

    python main.py --config=configs/xxx.yaml --ranking=pool

## Hyperparameters

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| --embedding-dim | int | 64 | Embedding Dim |
| --num-negs | int | 1500 | Num Negs |
| --num-neighbors | int | 10 | Num Neighbors |
| --neg-weight | float | 300 | Neg Weight |
| --unseen-only | eval | False | Unseen Only |
| --item-weight | float | 5e-4 | for item constraint |
| --init-weight | float | 1e-4 | std for init |
| --w1 | float | 1e-6 | W1 |
| --w2 | float | 1.0 | W2 |
| --w3 | float | 1e-6 | W3 |
| --w4 | float | 1.0 | W4 |

## Configuration Example

```yaml
# Data
root: ../../data
dataset: Yelp2018_10104811_ROU

# Model
embedding_dim: 64
num_neighbors: 10
unseen_only: False
num_negs: 800
neg_weight: 300
item_weight: 0.5
init_weight: 1.e-4
w1: 1.e-8
w2: 1.
w3: 1.e-8
w4: 1.

# Training
lr: 1.e-3
weight_decay: 1.e-4
epochs: 300
batch_size: 1024
optimizer: adam

# Evaluation
monitors: [LOSS, Recall@1, Recall@10, Recall@20, NDCG@10, NDCG@20]
which4best: NDCG@20
```
