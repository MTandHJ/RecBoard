# ETEGRec

[[official implementation](https://github.com/BishopLiu/ETEGRec)]

## Usage

1. Encode textual item features with `encode_textual_features.py`. The script writes an
   embedding pickle file to the processed dataset directory.

```bash
python encode_textual_features.py
```

2. Pretrain RQ-VAE tokenizer, which export `rqvae-model.pt` for downstream T5 training.

```bash
python train_rqvae.py --config=configs/rqvae/Amazon2014Beauty_550_LOU.yaml
```

3. Train or evaluate T5 using the pretrained RQ-VAE checkpoint.

Run with full-ranking constrained beam generation:

```bash
python train_etegrec.py --config=configs/etegrec/Amazon2014Beauty_550_LOU.yaml \
    --rqvae-path=/path/to/rqvae-model.pt --ranking=full
```

Run with sampled-pool candidate scoring:

```bash
python train_etegrec.py --config=configs/etegrec/Amazon2014Beauty_550_LOU.yaml \
    --rqvae-path=/path/to/rqvae-model.pt --ranking=pool
```

## Hyperparameters for ETEGRec

| Argument | Type | Default | Description |
| --- | --- | --- | --- |
| `--sem-feat-file` | str | `None` | Semantic item embedding file, relative to the processed dataset directory unless absolute. |
| `--maxlen` | int | `50` | Maximum interaction history length. |
| `--semantic-hidden-size` | int | `768` | Semantic item embedding dimension. |
| `--code-num` | int | `256` | Number of codewords for each code position. |
| `--code-length` | int | `4` | Code length including the collision-resolution suffix. |
| `--num-emb-list` | str | `256,256,256` | Residual codebook sizes before the suffix token. |
| `--e-dim` | int | `32` | RQ-VAE latent/codebook dimension. |
| `--layers` | str | `512,256,128` | RQ-VAE MLP hidden dimensions. |
| `--commit-weight` | float | `0.5` | TIGER-style residual quantizer commitment weight. |
| `--apply-shared-codebook` | bool | `False` | Share parameters across residual codebooks. |
| `--sk-epsilons` | str | `0.,0.,0.` | Sinkhorn epsilon values for each residual codebook. |
| `--sk-iters` | int | `50` | Number of Sinkhorn iterations. |
| `--rqvae-path` | str | `None` | Optional pretrained tokenizer checkpoint; TIGER `train_rqvae.py` checkpoints are supported. |
| `--embedding-dim` | int | `128` | T5 hidden size. |
| `--attention-size` | int | `64` | T5 key/value dimension. |
| `--intermediate-size` | int | `256` | T5 feed-forward dimension. |
| `--num-heads` | int | `4` | Number of attention heads. |
| `--num-layers` | int | `6` | Number of encoder layers. |
| `--num-decoder-layers` | int | `6` | Number of decoder layers. |
| `--num-beams` | int | `10` | Beam width for generated code ranking. |
| `--lr-rec` | float | `5.e-4` | Recommender learning rate. |
| `--lr-id` | float | `5.e-4` | Item tokenizer learning rate. |
| `--cycle` | int | `4` | Alternating cycle: one tokenizer epoch followed by three recommender epochs. |
| `--warm-epoch` | int | `10` | Epoch before alignment losses are enabled. |
| `--base-auxiliary-loss` | float | `1.0` | Global multiplier for KL and decoder contrastive auxiliary losses. |
| `--finetune-epochs` | int | `100` | Code-loss-only finetuning epochs after alternating training. |

## Configuration Example

```yaml
# Data
root: ../../data
dataset: Amazon2014Beauty_550_LOU
tasktag: NEXTITEM
sem_feat_file: sentence-t5-xl_title_categories_brand.pkl

# RQVAE
maxlen: 20
semantic_hidden_size: 768
code_num: 256
code_length: 4
num_emb_list: 256,256,256
e_dim: 32
layers: 512,256,128
alpha: 1.0
commit_weight: 0.5
apply_shared_codebook: false
sk_epsilons: 0.,0.,0.
sk_iters: 50
kmeans_init: false
kmeans_iters: 100
dropout_prob: 0.1
rqvae_path: /path/to/rqvae-model.pt

# T5
embedding_dim: 128
attention_size: 64
intermediate_size: 256
num_heads: 4
num_layers: 6
num_decoder_layers: 6
dropout_rate: 0.1
activation_function: relu
feed_forward_proj: relu

# Training
epochs: 400
batch_size: 512
weight_decay: 0.05
lr_rec: 5.e-4
lr_id: 5.e-4
cycle: 4
warm_epoch: 10
finetune_epochs: 0

# Loss weights
base_auxiliary_loss: 1.0
id_vq_loss: 1.0
id_code_loss: 0.0
id_kl_loss: 0.0 # need tuning
id_dec_cl_loss: 0.0 # need tuning
rec_vq_loss: 0.0
rec_code_loss: 1.0
rec_kl_loss: 0.0 # need tuning
rec_dec_cl_loss: 0.0 # need tuning
sim: cos

# Evaluation
num_beams: 10
eval_freq: 10
ranking: full
monitors: [LOSS, Recall@1, Recall@5, NDCG@5, Recall@10, NDCG@10]
which4best: NDCG@10

```
