# BSARec

[[official-code](https://github.com/yehjin-shin/BSARec.git)]

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
| --hidden-act | str | "gelu" | Hidden Act |
| --embedding-dim | int | 64 | Embedding Dim |
| --hidden-dropout-rate | float | 0.5 | Hidden Dropout Rate |
| --attn-dropout-rate | float | 0.5 | Attn Dropout Rate |
| --loss | str | "CE" | Loss |
| --c | int | 5 | the number of low-pass filters |
| --alpha | float | 0.7 | the ratio for frequency domain |

## Configuration Example

```yaml
# Data
root: ../../data
dataset: Amazon2014Beauty_550_LOU

# Model
maxlen: 50
embedding_dim: 64
num_blocks: 2
num_heads: 1
hidden_act: gelu
hidden_dropout_rate: 0.5
attn_dropout_rate: 0.5
loss: CE
alpha: 0.7
c: 5

# Training
epochs: 200
batch_size: 256
optimizer: adam
lr: 1.e-4
weight_decay: 1.e-4

# Evaluation
monitors: [LOSS, HitRate@1, HitRate@5, HitRate@10, HitRate@20, HitRate@50, NDCG@5, NDCG@10, NDCG@20, NDCG@50]
which4best: NDCG@10
```
