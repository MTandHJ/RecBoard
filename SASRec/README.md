# SASRec

[[official-code](https://github.com/kang205/SASRec)] [[pmixer-SASRec.pytorch](https://github.com/pmixer/SASRec.pytorch)]

## Usage

Run with full-ranking:

    python main.py --config=configs/xxx.yaml --ranking=full

or with sampled-based ranking:

    python main.py --config=configs/xxx.yaml --ranking=pool

## Hyperparameters

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| --maxlen | int | 50 | Maxlen |
| --num-heads | int | 1 | Num Heads |
| --num-blocks | int | 2 | Num Blocks |
| --embedding-dim | int | 64 | Embedding Dim |
| --dropout-rate | float | 0.2 | Dropout Rate |
| --loss | str | "BCE" | Loss |

## Configuration Example

```yaml
# Data
root: ../../data
dataset: Amazon2014Beauty_10100_LOU
tasktag: NEXTITEM

# Model
maxlen: 50
num_heads: 1
num_blocks: 2
embedding_dim: 64
dropout_rate: 0.2
loss: BCE

# Training
epochs: 300
batch_size: 512
optimizer: adam
lr: 5.e-3
weight_decay: 0.

# Evaluation
monitors: [LOSS, HitRate@1, HitRate@5, HitRate@10, NDCG@5, NDCG@10]
which4best: NDCG@10
```
