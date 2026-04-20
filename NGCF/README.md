# NGCF

[[official-code](https://github.com/xiangwang1223/neural_graph_collaborative_filtering)]

## Usage

Run with full-ranking:

    python main.py --config=configs/xxx.yaml --ranking=full

or with sampled-based ranking:

    python main.py --config=configs/xxx.yaml --ranking=pool

## Hyperparameters

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| --embedding-dim | int | 64 | Embedding Dim |
| --num-layers | int | 3 | Num Layers |
| --dropout-rate | float | 0.1 | Dropout Rate |

## Configuration Example

```yaml
# Data
root: ../../data
dataset: Yelp2018_10104811_ROU

# Model
embedding_dim: 64
num_layers: 4
dropout_rate: 0.

# Training
lr: 1.e-4
weight_decay: 1.e-6
epochs: 1000
batch_size: 2048
optimizer: adam

# Evaluation
monitors: [LOSS, HitRate@1, HitRate@5, HitRate@10, HitRate@20, HitRate@50, NDCG@5, NDCG@10, NDCG@20, NDCG@50]
which4best: NDCG@10
```
