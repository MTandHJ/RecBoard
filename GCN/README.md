# GCN

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

## Configuration Example

```yaml
# Data
root: ../../data
dataset: Yelp2018_554311_ROU

# Model
embedding_dim: 64
num_layers: 4

# Training
epochs: 1000
batch_size: 4096
optimizer: adam
lr: 1.e-3
weight_decay: 1.e-6

# Evaluation
monitors: [LOSS, Recall@1, Recall@10, Recall@20, NDCG@10, NDCG@20]
which4best: NDCG@20
```
