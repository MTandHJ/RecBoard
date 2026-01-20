
# LETTER

[[official-code](https://github.com/HonghuiBao2000/LETTER)]


## Usage

1. Encoding the textual features first according to `encode_textual_features.py`.

2. Encoding the collaborative features as follows:

    python encode_collab_features.py --config configs/sasrec/Amazon2014Beauty_550_LOU.yaml

2. Performing vector quantization by RQ-VAE (`train_rqvae.py`) as follows:

    python train_rqvae.py --config configs/rqvae/Amazon2014Beauty_550_LOU.yaml

3. Evaluation:

Run with full ranking:

    python train_t5.py --config=configs/t5/xxx.yaml --ranking=full

or with sampled-based ranking

    python train_t5.py --config=configs/t5/xxx.yaml --ranking=pool