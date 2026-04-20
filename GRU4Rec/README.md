# GRU4Rec

[[official-code](https://github.com/hidasib/GRU4Rec)]

## Usage

Run with full-ranking:

    python main.py --config=configs/xxx.yaml --ranking=full

or with sampled-based ranking:

    python main.py --config=configs/xxx.yaml --ranking=pool

## Hyperparameters

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| --maxlen | int | 50 | Maxlen |
| --embedding-dim | int | 64 | Embedding Dim |
| --hidden-size | int | 128 | Hidden Size |
| --emb-dropout-rate | float | 0.2 | Emb Dropout Rate |
| --hidden-dropout-rate | float | 0.2 | Hidden Dropout Rate |
| --num-blocks | int | 1 | the number of GRU layers |
| --loss | str | "BCE" | Loss |

## Configuration Example

```yaml
# Data
root: ../../data
dataset: Amazon2014Beauty_550_LOU

# Model
maxlen: 50
embedding_dim: 64
hidden_size: 128
emb_dropout_rate: 0.1
hidden_dropout_rate: 0.
num_blocks: 1
loss: BCE

# Training
epochs: 300
batch_size: 512
lr: 1.e-3
weight_decay: 1.e-8

# Evaluation
monitors: [LOSS, HitRate@1, HitRate@5, HitRate@10, NDCG@5, NDCG@10]
which4best: NDCG@10
```
