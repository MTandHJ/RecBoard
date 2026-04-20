# GLINT-RU

[[official-code](https://github.com/szhang-cityu/GLINT-RU)]

## Usage

Run with full-ranking:

    python main.py --config=configs/xxx.yaml --ranking=full

or with sampled-based ranking:

    python main.py --config=configs/xxx.yaml --ranking=pool

## Hyperparameters

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| --maxlen | int | 50 | Maxlen |
| --num-heads | int | 8 | Num Heads |
| --num-layers | int | 1 | Num Layers |
| --embedding-dim | int | 128 | Embedding Dim |
| --hidden-size | int | 128 | Hidden Size |
| --emb-dropout-rate | float | 0.0 | Emb Dropout Rate |
| --hidden-dropout-rate | float | 0.2 | Hidden Dropout Rate |
| --attn-dropout-rate | float | 0.2 | Attn Dropout Rate |
| --layer-norm-eps | float | 1.0e-12 | Layer Norm Eps |
| --loss | str | "BCE" | Loss |

## Configuration Example

```yaml
# Data
root: ../../data
dataset: Amazon2014Beauty_550_LOU

# Model
maxlen: 50
num_heads: 8
num_layers: 1
embedding_dim: 128
hidden_size: 128
emb_dropout_rate: 0.
hidden_dropout_rate: 0.2
attn_dropout_rate: 0.2
loss: BCE

# Training
epochs: 200
batch_size: 2048
optimizer: Adam
lr: 1.e-3
weight_decay: 0.

# Evaluation
monitors: [LOSS, HitRate@1, HitRate@5, HitRate@10, NDCG@5, NDCG@10]
which4best: NDCG@10
```
