
# TIGER

[[EdoardoBotta-RQ-VAE-Recommender](https://github.com/EdoardoBotta/RQ-VAE-Recommender)]

> [!NOTE]
> We implement the recommendation backbone on SASRec instead of T5.


## Usage

1. Encoding the textual features first according to `encode_textual_features.py`.

2. Performing vector quantization by RQ-VAE (`train_rqvae.py`) or Residual Kmeans (`train_residual_kmeans.py`).

3. Evaluation:

Run with full ranking:

    torchrun --nproc-per-node 4 main.py --config=configs/sasrec/xxx.yaml --ranking=full

or with sampled-based ranking

    torchrun --nproc-per-node 4 main.py --config=configs/sasrec/xxx.yaml --ranking=pool