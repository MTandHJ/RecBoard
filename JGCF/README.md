# JGCF

[[official-code](https://github.com/IE-NITK/JGCF)]

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
| --scaling-factor | float | 3.0 | hyper-parameter for rescaling |
| --alpha | float | 1.0 | hyper-parameter for Jacobi Polynomial |
| --beta | float | 1.0 | hyper-parameter for Jacobi Polynomial |
| --weight4mid | float | 0.1 | weight for scaling mid |

## Configuration Example

```yaml
# Data
root: ../../data
dataset: Yelp2018_10104811_ROU

# Model
embedding_dim: 64
num_layers: 3
scaling_factor: 5.
alpha: 1.5
beta: 0.5
weight4mid: 0.

# Training
lr: 1.e-3
weight_decay: 1.e-3
optimizer: adam
epochs: 1000
batch_size: 2048

# Evaluation
monitors: [LOSS, Recall@1, Recall@10, Recall@20, NDCG@10, NDCG@20]
which4best: NDCG@20
```
