# SimpleX

[[official-code](https://github.com/xue-pai/SimpleX)]

## Usage

Run with full-ranking:

    python main.py --config=configs/xxx.yaml --ranking=full

or with sampled-based ranking:

    python main.py --config=configs/xxx.yaml --ranking=pool

## Hyperparameters

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| --embedding-dim | int | 64 | Embedding Dim |
| --num-negs | int | 1000 | Num Negs |
| --gamma | float | 1.0 | Gamma |
| --margin | float | 0.9 | Margin |
| --weight-for-negative | float | 150 | Weight For Negative |
| --dropout-rate | float | 0.1 | Dropout Rate |
| --unseen-only | eval | False | Unseen Only |

## Configuration Example

```yaml
# Data
root: ../../data
dataset: Yelp2018_10104811_ROU

# Model
gamma: .5
embedding_dim: 64
dropout_rate: .1
num_negs: 1000
margin: 0.9
weight_for_negative: 100
unseen_only: False

# Training
lr: 1.e-4
weight_decay: 1.e-8
epochs: 100
batch_size: 512

# Evaluation
monitors: [LOSS, HitRate@1, HitRate@5, HitRate@10, HitRate@20, HitRate@50, NDCG@5, NDCG@10, NDCG@20, NDCG@50]
which4best: NDCG@10
```
