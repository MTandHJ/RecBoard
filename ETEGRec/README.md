# ETEGRec

[[official implementation](https://github.com/BishopLiu/ETEGRec)]

ETEGRec first pretrains an item RQ-VAE tokenizer, then alternates tokenizer and
T5 recommender training. This RecBoard version keeps that training flow while
exposing full/pool item ranking through the freerec evaluator.

## Data

`--sem-feat-file` should point to a `.pkl` tensor with shape `[Item.count, D]`.
The path is used as provided. The semantic feature dimension is inferred from
the loaded tensor.

## Usage

Pretrain the RQ-VAE tokenizer:

```bash
python train_rqvae.py --config=configs/rqvae/Amazon2014Beauty_550_LOU.yaml
```

Then train ETEGRec with the pretrained tokenizer checkpoint:

```bash
python train_etegrec.py --config=configs/etegrec/Amazon2014Beauty_550_LOU.yaml \
    --rqvae-path=/path/to/tokenizer.pt --ranking=full
```

Use sampled-pool scoring:

```bash
python train_etegrec.py --config=configs/etegrec/Amazon2014Beauty_550_LOU.yaml \
    --rqvae-path=/path/to/tokenizer.pt --ranking=pool
```

## Notes

- `cycle=2` matches the official reproduction script: one tokenizer epoch
  followed by one recommender epoch.
- `lr_rec=0.005`, `lr_id=0.0001`, `lr_scheduler_type=cosine`,
  `warmup_steps=8000`, `n_positions=210`, `num_beams=20`,
  `commit_weight=0.25`, `codebook_dim=128`, and `hidden_dims=512,256` follow
  the source configuration.
- `id_kl_loss`, `id_dec_cl_loss`, `rec_kl_loss`, and `rec_dec_cl_loss` are direct
  source weights. There is no extra global auxiliary-loss multiplier.
- Source ETEGRec reports code-level exact-match metrics. RecBoard returns item
  score matrices so evaluation stays consistent with TIGER/freerec full and
  pool ranking.
