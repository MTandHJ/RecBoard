# BM3

[[official-code](https://github.com/enoche/BM3)]

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
| --dropout-rate | float | 0.5 | Dropout Rate |
| --second-l | float | 2.0 | Second L |
| --reg-weight | float | 1.0e-1 | Reg Weight |
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
num_layers: 1
dropout_rate: 0.5
second_l: 2.
reg_weight: 1.e-1
vfile: visual_modality.pkl
tfile: textual_modality.pkl
afile: Null

# Training
epochs: 500
batch_size: 2048
optimizer: adam
lr: 1.e-3
weight_decay: 1.e-4

# Evaluation
monitors: [LOSS, Recall@1, Recall@10, Recall@20, NDCG@10, NDCG@20]
which4best: NDCG@20
```
