# AlphaRec

[[official-code](https://github.com/LehengTHU/AlphaRec)]

## Usage

Run with full-ranking:

    python main.py --config=configs/xxx.yaml --ranking=full

or with sampled-based ranking:

    python main.py --config=configs/xxx.yaml --ranking=pool

## Hyperparameters

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| --embedding-dim | int | 64 | Embedding Dim |
| --num-layers | int | 2 | Num Layers |
| --tfile | str | "textual_modality.pkl" | the file of textual modality features |
| --projector | str | "linear" | Projector |
| --num_negs | int | 256 | Num_Negs |
| --tau | float | 0.15 | Tau |

## Configuration Example

```yaml
# Data
root: ../../data
dataset: AmazonMovies_Alpha
tasktag: Matching

# Model
embedding_dim: 64
num_layers: 2
tau: 0.15
num_negs: 256
projector: mlp

# Training
epochs: 500
batch_size: 4096
optimizer: adam
lr: 5.e-4
weight_decay: 1.e-6

# Evaluation
monitors: [LOSS, Recall@1, Recall@10, Recall@20, HitRate@10, HitRate@20, NDCG@10, NDCG@20]
which4best: Recall@20
```
