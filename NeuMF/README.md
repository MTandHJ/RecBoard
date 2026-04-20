# NeuMF

[[official-code](https://github.com/hexiangnan/neural_collaborative_filtering)]

## Usage

Run with full-ranking:

    python main.py --config=configs/xxx.yaml --ranking=full

or with sampled-based ranking:

    python main.py --config=configs/xxx.yaml --ranking=pool

## Hyperparameters

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| --embedding-dim | int | 64 | Embedding Dim |
| --hidden-dims | str | "64,32,16,8" | Hidden Dims |
| --num-negs | int | 4 | Num Negs |

## Configuration Example

```yaml
# Data
root: ../../data
dataset: Yelp2018_10104811_ROU

# Model
embedding_dim: 8
hidden_dims: "64,32,16,8"
num_negs: 4

# Training
epochs: 200
batch_size: 1024
optimizer: adam
lr: 1.e-3
weight_decay: 0.

# Evaluation
monitors: [LOSS, HitRate@1, HitRate@5, HitRate@10, HitRate@20, HitRate@50, NDCG@5, NDCG@10, NDCG@20, NDCG@50]
which4best: NDCG@10
```
