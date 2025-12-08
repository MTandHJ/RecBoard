
# E4SRec

[[official-code](https://github.com/HestiaSky/E4SRec)]


## Usage

1. Encoding the item embeddings via `train_sasrec.py`:

    python train_sasrec.py --config=configs/sasrec.yaml --dataset XXX

2. Evaluation:

Run with full ranking:

    python main.py --config=configs/e4srec.yaml --dataset XXX --ranking=full

or with sampled-based ranking

    python main.py --config=configs/e4srec.yaml --dataset XXX --ranking=pool


Distributed training can be performed by (accordingly adjusting the `batch_size` and `gradient_accumulation_steps` parameters)

    torchrun --nproc_per_node=4 main.py --config=configs/e4srec.yaml --dataset XXX
