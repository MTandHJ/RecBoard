# LATTICE

[[official-code](https://github.com/CRIPAC-DIG/LATTICE)]

## Usage

Run with full-ranking:

    python main.py --config=configs/xxx.yaml --ranking=full

or with sampled-based ranking:

    python main.py --config=configs/xxx.yaml --ranking=pool

## Hyperparameters

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| --embedding-dim | int | 64 | Embedding Dim |
| --num-ui-layers | int | 2 | the number of layers for U-I graph |
| --num-ii-layers | int | 1 | the number of layers for I-I graph |
| --knn-k | int | 10 | top-k knn graph |
| --origin-ratio | float | 0.5 | ratio of fixed graph to learnable graph |
| --vfile | str | "visual_modality.pkl" | the file of visual modality features |
| --tfile | str | "textual_modality.pkl" | the file of textual modality features |

## Configuration Example

```yaml
# Data
root: ../../data
dataset: Amazon2014Baby_550_MMRec

# Model
embedding_dim: 64
num_ui_layers: 2
num_ii_layers: 1
origin_ratio: 0.9
vfile: visual_modality.pkl
tfile: textual_modality.pkl

# Training
epochs: 500
batch_size: 2048
optimizer: adam
lr: 1.e-3
weight_decay: 1.e-6

# Evaluation
monitors: [LOSS, Recall@1, Recall@10, Recall@20, NDCG@10, NDCG@20]
which4best: NDCG@20
```
