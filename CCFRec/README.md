# CCFRec

[[official-code](https://github.com/BishopLiu/CCFRec)]

> [!NOTE]
> I provide scripts for generating SIDs based on the fields "Title", "Categories", and "Brand". To extend this functionality to other settings, the configurations in `encode_textual_features.py` and `generate_semantic_ids.py` should be modified accordingly.

> [!NOTE]
> The implementation of CCFRec slightly differs from the original in the "forward encoding" component.

## Usage

1. (`encode_textual_features.py`) Encoding textual features for Title|Categories|Brand ...

2. (`generate_semantic_ids.py`) Generating SIDs ...

3. (`main.py`)

Run with full-ranking:

    python main.py --config=configs/xxx.yaml --ranking=full

or with sampled-based ranking:

    python main.py --config=configs/xxx.yaml --ranking=pool

## Hyperparameters

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| --maxlen | int | 50 | Maxlen |
| --num-heads | int | 1 | Num Heads |
| --embedding-dim | int | 64 | Embedding Dim |
| --hidden-size | int | 64 * 4 | Hidden Size |
| --hidden-act | str | "gelu" | Hidden Act |
| --tau | float | 0.07 | temperature |
| --num-negs | int | 49 | for rec loss |
| --weight4mlm | float | 0.1 | Weight4Mlm |
| --weight4cl | float | 0.1 | Weight4Cl |
| --mask-ratio | float | 0.5 | Mask Ratio |
| --num-qformer-blocks | int | 2 | Num Qformer Blocks |
| --qformer-dropout-rate | float | 0.3 | Qformer Dropout Rate |
| --num-encoder-blocks | int | 2 | Num Encoder Blocks |
| --encoder-dropout-rate | float | 0.5 | Encoder Dropout Rate |
| --sem-id-ckpt | str | None | checkpoint file of 'sem_ids' |
| --tfiles | str | None | checkpoint file of textual features |

## Configuration Example

```yaml
# Data
root: ../../data
dataset: Amazon2014Beauty_550_LOU

# Model
maxlen: 20
num_heads: 2
embedding_dim: 64
hidden_size: 256
num_qformer_blocks: 2
qformer_dropout_rate: 0.1
num_encoder_blocks: 2
encoder_dropout_rate: 0.3
num_negs: 49
weight4mlm: 0.1
weight4cl: 0.5
mask_ratio: 0.2
sem_id_ckpt: product_12_256.pkl
tfiles: sentence-t5-xl_brand.pkl,sentence-t5-xl_categories.pkl,sentence-t5-xl_title.pkl

# Training
epochs: 100
batch_size: 512
optimizer: AdamW
lr: 1.e-3
weight_decay: 1.e-8

# Evaluation
monitors: [LOSS, HitRate@1, HitRate@5, HitRate@10, HitRate@20, HitRate@50, NDCG@5, NDCG@10, NDCG@20, NDCG@50]
which4best: NDCG@10
```
