# MGCN

[[official-code](https://github.com/demonph10/MGCN)]

## Usage

Run with full-ranking:

    python main.py --config=configs/xxx.yaml --ranking=full

or with sampled-based ranking:

    python main.py --config=configs/xxx.yaml --ranking=pool

## Hyperparameters

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| --embedding-dim | int | 64 | Embedding Dim |
| --num-layers | int | 2 | the number of layers for U-I graph |
| --knn-k | int | 10 | top-k knn graph |
| --weight4cl | float | 0.01 | weight for contrastive loss |
| --temperature | float | 0.2 | temperature for contrastive loss |
| --afile | str | None | the file of acoustic modality features |
| --vfile | str | "visual_modality.pkl" | the file of visual modality features |
| --tfile | str | "textual_modality.pkl" | the file of textual modality features |

## Configuration Example

```yaml
# Data
root: ../../data
dataset: Amazon2014Baby_550_MMRec

# Model
embedding_dim: 64
num_layers: 2
knn_k: 10
weight4cl: 0.01
temperature: 0.2
afile: Null
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
