# PairNorm

[[official-code](https://github.com/LingxiaoShawn/PairNorm)]

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
| --norm-mode | str | "PN" | Norm Mode |
| --norm-scale | float | 10.0 | Norm Scale |

## Configuration Example

```yaml
# Data
root: ../../data
dataset: Amazon2014Baby_550_MMRec

# Model
embedding_dim: 64
num_layers: 3
norm_mode: PN
norm_scale: 1

# Training
epochs: 500
batch_size: 2048
optimizer: adam
lr: 1.e-3
weight_decay: 1.e-2

# Evaluation
monitors: [LOSS, Recall@1, Recall@10, Recall@20, NDCG@10, NDCG@20]
which4best: NDCG@20
```
