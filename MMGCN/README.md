# MMGCN

[[official-code](https://github.com/weiyinwei/MMGCN)]

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
| --fusion-mode | str | "add" | Fusion Mode |
| --afile | str | None | the file of acoustic modality features |
| --vfile | str | "visual_modality.pkl" | the file of visual modality features |
| --tfile | str | "textual_modality.pkl" | the file of textual modality features |

## Configuration Example

```yaml
# Data
root: ../../data
dataset: Allrecipes_MMSSL

# Model
embedding_dim: 64
num_layers: 1
fusion_mode: add
vfile: visual_modality.pkl
tfile: textual_modality.pkl
afile: Null

# Training
epochs: 50
batch_size: 1024
optimizer: adam
lr: 5.e-4
weight_decay: 1.e-4
eval_freq: 1

# Evaluation
monitors: [LOSS, HitRate@1, HitRate@5, HitRate@10, HitRate@20, HitRate@50, NDCG@5, NDCG@10, NDCG@20, NDCG@50]
which4best: NDCG@10
```
