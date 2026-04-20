# HSTU

[[official-code](https://github.com/meta-recsys/generative-recommenders)]

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
| --num-blocks | int | 16 | Num Blocks |
| --embedding-dim | int | 64 | Embedding Dim |
| --linear-hidden-dim | int | 8 | Linear Hidden Dim |
| --attention-dim | int | 8 | Attention Dim |
| --emb-dropout-rate | float | 0.0 | Emb Dropout Rate |
| --hidden-dropout-rate | float | 0.0 | Hidden Dropout Rate |
| --num-negs | int | 512 | Num Negs |
| --num-buckets | int | 100 | Num Buckets |
| --temperature | float | 0.05 | Temperature |

## Configuration Example

```yaml
# Data
root: ../../data
dataset: Amazon2014Beauty_550_LOU

# Model
maxlen: 50
num_heads: 2
num_blocks: 4
embedding_dim: 64
linear_hidden_dim: 4
attention_dim: 4
emb_dropout_rate: 0.4
hidden_dropout_rate: 0.2
beta1: 0.9
beta2: 0.98
num_negs: 512
temperature: 0.1
num_buckets: 128

# Training
epochs: 300
batch_size: 256
optimizer: AdamW
lr: 1.e-3
weight_decay: 1.e-3

# Evaluation
monitors: [LOSS, HitRate@1, HitRate@5, HitRate@10, HitRate@20, HitRate@50, NDCG@5, NDCG@10, NDCG@20, NDCG@50]
which4best: NDCG@10
```
