# NARM

[[official-code](https://github.com/lijingsdu/sessionRec_NARM)]
[[RecBole](https://github.com/RUCAIBox/RecBole/blob/master/recbole/model/sequential_recommender/narm.py)]

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
| --hidden-dropout-rate | float | 0.0 | Hidden Dropout Rate |
| --ct-dropout-rate | float | 0.5 | Ct Dropout Rate |
| --num-blocks | int | 1 | the number of GRU layers |

## Configuration Example

```yaml
# Data
root: ../../data
dataset: Amazon2014Beauty_550_LOU

# Model
maxlen: 50
hidden_size: 64
embedding_dim: 64
emb_dropout_rate: 0.2
hidden_dropout_rate: 0.
ct_dropout_rate: 0.3
num_blocks: 1

# Training
epochs: 300
batch_size: 512
lr: 1.e-3
weight_decay: 1.e-6

# Evaluation
monitors: [LOSS, HitRate@1, HitRate@5, HitRate@10, NDCG@5, NDCG@10]
which4best: NDCG@10
```
