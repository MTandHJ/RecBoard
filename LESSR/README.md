# LESSR

[[official-code](https://github.com/twchen/lessr)]

## Usage

Run with full-ranking:

    python main.py --config=configs/xxx.yaml --ranking=full

or with sampled-based ranking:

    python main.py --config=configs/xxx.yaml --ranking=pool

## Hyperparameters

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| --maxlen | int | 50 | Maxlen |
| --num-layers | int | 3 | Num Layers |
| --embedding-dim | int | 64 | Embedding Dim |
| --dropout-rate | float | 0.2 | Dropout Rate |
| --batch-norm | bool | True | Batch Norm |

## Configuration Example

```yaml
# Data
root: ../../data
dataset: Amazon2014Beauty_550_LOU

# Model
maxlen: 50
num_layers: 3
embedding_dim: 64
dropout_rate: 0.7

# Training
lr: 1.e-3
epochs: 100
batch_size: 512
optimizer: adamw
weight_decay: 1.e-6

# Evaluation
monitors: [LOSS, HitRate@1, HitRate@5, HitRate@10, NDCG@5, NDCG@10]
which4best: NDCG@10
```
