# LightGCN

[[official-code](https://github.com/gusye1234/LightGCN-PyTorch)]

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
dataset: Allrecipes_MMSSL

# Model
embedding_dim: 64
num_layers: 2

# Training
epochs: 100
batch_size: 2048
optimizer: adam
lr: 1.e-3
weight_decay: 1.e-4
eval_freq: 1

# Evaluation
monitors: [LOSS, Recall@1, Recall@10, Recall@20, NDCG@10, NDCG@20]
which4best: NDCG@20
```
