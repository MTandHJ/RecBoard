# FPMC

[RecBole-FPMC](https://github.com/RUCAIBox/RecBole/blob/master/recbole/model/sequential_recommender/fpmc.py)

## Usage

Run with full-ranking:

    python main.py --config=configs/xxx.yaml --ranking=full

or with sampled-based ranking:

    python main.py --config=configs/xxx.yaml --ranking=pool

## Hyperparameters

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| --maxlen | int | 50 | Max sequence length for trainpipe |
| --embedding-dim | int | 64 | Hidden size of model |
| --loss | str | "BPR" | Loss function (BPR, BCE, or CE) |

## Configuration Example

```yaml
# Data
root: ../../data
dataset: Amazon2014Beauty_550_LOU

# Model
maxlen: 50
embedding_dim: 64
loss: BPR

# Training
epochs: 300
batch_size: 512
optimizer: AdamW
lr: 1.e-3
weight_decay: 0.

# Evaluation
monitors: [LOSS, HitRate@1, HitRate@5, HitRate@10, HitRate@20, HitRate@50, NDCG@5, NDCG@10, NDCG@20, NDCG@50]
which4best: NDCG@10
```