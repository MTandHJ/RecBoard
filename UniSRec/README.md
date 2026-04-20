# UniSRec

[[official-code](https://github.com/RUCAIBox/UniSRec)]

> [!NOTE]
> No `word drop` operation and `pos item` filtering.

## Usage

Run with full-ranking:

    torchrun --nproc-per-node 4 main.py --config=configs/xxx.yaml --ranking=full

or with sampled-based ranking:

    torchrun --nproc-per-node 4 main.py --config=configs/xxx.yaml --ranking=pool

## Hyperparameters

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| --maxlen | int | 50 | Maxlen |
| --num-heads | int | 1 | Num Heads |
| --num-blocks | int | 2 | Num Blocks |
| --embedding-dim | int | 64 | Embedding Dim |
| --hidden-dropout-rate | float | 0.2 | Hidden Dropout Rate |
| --attn-dropout-rate | float | 0.2 | Attn Dropout Rate |
| --adaptor-dropout-rate | float | 0.2 | Adaptor Dropout Rate |
| --num-moe-experts | int | 8 | Num Moe Experts |
| --T | float | 0.02 | T(emperature) |
| --mask-ratio | float | 0.2 | Mask Ratio |
| --s2sloss-weight | float | 1.0e-3 | S2Sloss Weight |
| --tfile | str | None | Tfile |

## Configuration Example

```yaml
# Data
root: ../../data
dataset: Amazon2014Beauty_1000_LOU,Amazon2014Home_1000_LOU,Amazon2014Clothing_1000_LOU,Amazon2014CDs_1000_LOU,Amazon2014Movies_1000_LOU
tasktag: NEXTITEM

# Model
maxlen: 50
num_heads: 2
num_blocks: 1
embedding_dim: 64
hidden_dropout_rate: 0.2
attn_dropout_rate: 0.6
adaptor_dropout_rate: 0.
num_moe_experts: 16
T: 0.1
mask_ratio: 0.3
s2sloss_weight: 1.e-4
tfile: all-minilm-l12-v2_title_categories_brand.pkl

# Training
epochs: 300
batch_size: 512
optimizer: AdamW
lr: 1.e-3
weight_decay: 0.01

# Evaluation
monitors: [LOSS, HitRate@1, HitRate@5, HitRate@10, NDCG@5, NDCG@10]
which4best: NDCG@10
```
