# CAGCN

[[official-code](https://github.com/YuWVandy/CAGCN)]

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
| --trend-type | str | "jc" | Trend Type |
| --trend-coeff | float | 2 | Trend Coeff |
| --fusion | eval | "True" | Fusion |

## Configuration Example

```yaml
# Data
root: ../../data
dataset: Yelp2018_10104811_ROU

# Model
embedding_dim: 64
num_layers: 3
trend_type: jc
trend_coeff: 1
fusion: True

# Training
epochs: 1000
batch_size: 2048
optimizer: adam
lr: 1.e-3
weight_decay: 1.e-3

# Evaluation
monitors: [LOSS, HitRate@1, HitRate@5, HitRate@10, HitRate@20, HitRate@50, NDCG@5, NDCG@10, NDCG@20, NDCG@50]
which4best: NDCG@10
```
