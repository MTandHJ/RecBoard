# SimGCL

[[official-code](https://github.com/Coder-Yu/QRec/blob/master/model/ranking/SimGCL.py)]

## Usage

Run with full-ranking:

    python main.py --config=configs/xxx.yaml --ranking=full

or with sampled-based ranking:

    python main.py --config=configs/xxx.yaml --ranking=pool

## Hyperparameters

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| --embedding-dim | int | 64 | Embedding Dim |
| --num-layers | int | 3 | Num Layers |
| --eps | float | 0.2 | the magnitude of the noise |
| --temperature | float | 0.2 | Temperature |
| --ssl-weight | float | 0.2 | the weight for contrastive loss |

## Configuration Example

```yaml
# Data
dataset: Yelp2018_10104811_ROU
root: ../../data

# Model
embedding_dim: 64
num_layers: 3
eps: 0.1
temperature: 0.2
ssl_weight: 0.3

# Training
epochs: 1000
batch_size: 2048
optimizer: adam
lr: 1.e-3
weight_decay: 1.e-3

# Evaluation
monitors: [LOSS, Recall@1, Recall@10, Recall@20, NDCG@10, NDCG@20]
which4best: NDCG@20
```
