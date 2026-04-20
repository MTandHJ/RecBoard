# FREEDOM

[[official-code](https://github.com/enoche/FREEDOM)]

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
| --weight4mAdj | float | 0.1 | weight for fusing vAdj and tAd |
| --weight4modality | float | 1.0e-3 | weight for modality BPR loss |
| --sampling-ratio | float | 0.2 | sampling ratio for U-I graph |
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
weight4mAdj: 0.1
weight4modality: 1.e-3
sampling_ratio: 0.2
vfile: visual_modality.pkl
tfile: textual_modality.pkl

# Training
epochs: 500
batch_size: 2048
optimizer: adam
lr: 1.e-3
weight_decay: 1.e-6

# Evaluation
monitors: [LOSS, HitRate@1, HitRate@5, HitRate@10, HitRate@20, HitRate@50, NDCG@5, NDCG@10, NDCG@20, NDCG@50]
which4best: NDCG@10
```
