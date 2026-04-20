# E4SRec

[[official-code](https://github.com/HestiaSky/E4SRec)]

## Usage

1. Encoding the item embeddings via `train_sasrec.py`:

        python train_sasrec.py --config=configs/sasrec.yaml --dataset XXX

2. Evaluation:

Run with full-ranking:

    python main.py --config=configs/e4srec.yaml --dataset XXX --ranking=full

or with sampled-based ranking:

    python main.py --config=configs/e4srec.yaml --dataset XXX --ranking=pool

Distributed training can be performed by (accordingly adjusting the `batch_size` and `gradient_accumulation_steps` parameters)

    torchrun --nproc_per_node=4 main.py --config=configs/e4srec.yaml --dataset XXX

## Hyperparameters

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| --saved-model | str | "./models/Platypus2-7B" | Saved Model |
| --prompt-template | str | "./configs/alpaca.json" | Prompt Template |
| --maxlen | int | 50 | Maxlen |
| --lora-rank | int | 16 | Lora Rank |
| --lora-alpha | int | 16 | Lora Alpha |
| --lora-dropout-rate | float | 0.05 | Lora Dropout Rate |

## Configuration Example

```yaml
# Data
root: ../../data
tasktag: NEXTITEM

# Model
maxlen: 50
lora_rank: 16
lora_alpha: 16
lora_dropout_rate: 0.05

# Training
epochs: 3
batch_size: 128 # batch_size per device
gradient_accumulation_steps: 1
optimizer: AdamW
lr: 3.e-4
weight_decay: 0.

# Evaluation
monitors: [LOSS, HitRate@1, HitRate@10, HitRate@20, NDCG@10, NDCG@20]
which4best: NDCG@10
```
