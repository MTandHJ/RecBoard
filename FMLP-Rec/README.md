# FMLP-Rec

[[official-code](https://github.com/Woeee/FMLP-Rec)]

## Usage

Run with full-ranking:

    python main.py --config=configs/xxx.yaml --ranking=full

or with sampled-based ranking:

    python main.py --config=configs/xxx.yaml --ranking=pool

## Hyperparameters

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| --maxlen | int | 50 | Maxlen |
| --embedding-dim | int | 64 | hidden size of model |
| --num-blocks | int | 2 | number of filter-enhanced blocks |
| --num-heads | int | 2 | Num Heads |
| --hidden-act | str | "gelu" | Hidden Act |
| --hidden-dropout-rate | float | 0.5 | Hidden Dropout Rate |
| --loss | str | "BPR" | Loss |

## Configuration Example

```yaml
# Data
root: ../../data
dataset: Amazon2014Beauty_550_LOU

# Model
maxlen: 50
embedding_dim: 64
num_blocks: 4
num_heads: 2
hidden_act: gelu
hidden_dropout_rate: 0.4
loss: BPR

# Training
epochs: 300
batch_size: 512
optimizer: adam
lr: 1.e-4
weight_decay: 0.

# Evaluation
monitors: [LOSS, HitRate@1, HitRate@5, HitRate@10, NDCG@5, NDCG@10]
which4best: NDCG@10
```
