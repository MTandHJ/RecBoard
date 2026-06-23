# LETTER

[[official-code](https://github.com/HonghuiBao2000/LETTER)]

## Usage

1. Encode textual item features. The output pickle is written to the processed
   dataset directory and is consumed by LETTER RQ-VAE.

```bash
python encode_textual_features.py
```

2. Encode collaborative item features with SASRec.

```bash
python encode_collab_features.py --config=configs/sasrec/Amazon2014Beauty_550_LOU.yaml
```

3. Train LETTER RQ-VAE. The trainer exports `sid_vocab.json` for downstream
   T5 training.

```bash
python train_rqvae.py --config=configs/rqvae/Amazon2014Beauty_550_LOU.yaml
```

4. Train or evaluate T5 using the exported semantic-ID vocabulary.

Run with full-ranking constrained beam generation:

```bash
python train_t5.py --config=configs/t5/Amazon2014Beauty_550_LOU.yaml \
    --sid-vocab-file=/path/to/sid_vocab.json --ranking=full
```

Run with sampled-pool candidate scoring:

```bash
python train_t5.py --config=configs/t5/Amazon2014Beauty_550_LOU.yaml \
    --sid-vocab-file=/path/to/sid_vocab.json --ranking=pool
```

## Hyperparameters

### RQ-VAE

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--sem-feat-file` | str | `None` | Pickle file containing textual item features. |
| `--collab-feat-file` | str | `None` | Pickle file containing collaborative item features. |
| `--num-codebooks` | int | `4` | Number of residual codebooks. |
| `--num-codewords` | int | `256` | Number of codewords per codebook. |
| `--codebook-dim` | int | `32` | Dimension of each quantized representation. |
| `--hidden-dims` | str | `512,256,128` | Comma-separated encoder hidden sizes. |
| `--sk-epsilons` | str | `0.,0.,0.,0.003` | Comma-separated Sinkhorn epsilon values. |
| `--cf-weight` | float | `0.01` | Collaborative alignment loss weight. |
| `--commit-weight` | float | `0.25` | Commitment-loss weight. |
| `--diversity-weight` | float | `0.0001` | LETTER codebook diversity loss weight. |

### T5

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--maxlen` | int | `50` | Maximum item-history length. |
| `--embedding-dim` | int | `128` | T5 hidden dimension. |
| `--attention-size` | int | `64` | T5 key/value dimension. |
| `--intermediate-size` | int | `256` | T5 feed-forward dimension. |
| `--num-heads` | int | `4` | Number of attention heads. |
| `--num-layers` | int | `6` | Number of encoder and decoder layers. |
| `--dropout-rate` | float | `0.1` | T5 dropout rate. |
| `--sid-vocab-file` | str | `sid_vocab.json` | Semantic-ID vocabulary emitted by LETTER RQ-VAE. |
| `--num-beams` | int | `20` | Beam width for full ranking. |

## Configuration Example

```yaml
# Data
root: ../../data
dataset: Amazon2014Beauty_550_LOU
tasktag: NEXTITEM
sem_feat_file: sentence-t5-xl_title_categories_brand.pkl
collab_feat_file: ./collab/SASRec/Amazon2014Beauty_550_LOU/1222103121/sasrec-Amazon2014Beauty_550_LOU.pkl

# Model
num_codewords: 256
num_codebooks: 4
codebook_dim: 32
hidden_dims: 512,256,128
apply_kmeans_init: true
sk_epsilons: 0.,0.,0.,0.003
sk_iters: 50

# Training
epochs: 20000
batch_size: 1024
optimizer: AdamW
lr: 1.e-3
weight_decay: 0.1
dropout_rate: 0.
cf_weight: 0.01
commit_weight: 1.
diversity_weight: 0.0001

# Evaluation
eval_freq: 100
CHECKPOINT_FREQ: 100
which4best: PPL
```
