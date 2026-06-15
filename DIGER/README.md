# DIGER

[[official-code](https://github.com/junchen-fu/DIGER)]

## Usage

1. Encode textual item features with `encode_textual_features.py`. The script writes an embedding pickle file to the processed dataset directory.

```bash
python encode_textual_features.py --model=/path/to/transformers-model
```

2. Pretrain the RQVAE:

```bash
python train_rqvae.py --config=configs/rqvae/Amazon2014Beauty_550_LOU.yaml
```

3. Train or evaluate DIGER:

```bash
python main.py --config=configs/diger/Amazon2014Beauty_550_LOU.yaml --rqvae-path /path/to/rqvae.pt
```

## Hyperparameters

| Argument | Type | Default | Description |
| --- | --- | --- | --- |
| `--item-feat-file` | str | `None` | Pickle file containing item feature tensor `[Item.count, D]`. |
| `--maxlen` | int | `50` | Maximum item-history length. |
| `--embedding-dim` | int | `128` | T5 hidden dimension. |
| `--num-codebooks` | int | `3` | Number of residual quantization levels before collision code. |
| `--num-codewords` | int | `256` | Number of codewords per codebook and collision bucket. |
| `--codebook-dim` | int | `256` | ID tokenizer latent/codebook dimension. |
| `--hidden-dims` | str | `2048,1024,512` | ID encoder/decoder hidden sizes. |
| `--rqvae-path` | str | `None` | Module-wise RQ-VAE checkpoint from `train_rqvae.py`. |
| `--freeze-id-encoder` | bool | `False` | Freeze the pretrained ID encoder during main DIGER training. |
| `--use-adaptive-selection` | bool | `False` | Enable FrqUD adaptive deterministic/Gumbel code selection. |
| `--use-learnable-sigma-gumbel` | bool | `False` | Enable SDUD learnable Gumbel scale. |
| `--num-beams` | int | `20` | Beam width used to produce freerec ranking scores. |
