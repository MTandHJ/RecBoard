
# TIGER

[[EdoardoBotta-RQ-VAE-Recommender](https://github.com/EdoardoBotta/RQ-VAE-Recommender)]


## Usage

1. Encoding the textual features first according to `encode_textual_features.py`.

2. Performing vector quantization by RQ-VAE (`train_rqvae.py`) or Residual Kmeans (`train_residual_kmeans.py`).

3. Evaluation:

Run with full ranking:

    python train_t5.py --config=configs/t5/xxx.yaml --ranking=full

or with sampled-based ranking

    python train_t5.py --config=configs/t5/xxx.yaml --ranking=pool