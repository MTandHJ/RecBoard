

# Adafactor

[[official-code](https://github.com/facebookresearch/fairseq/blob/main/fairseq/optim/adafactor.py)]


**Note:** Adafactor is a memory-efficient optimizer.


## Usage

Run with full ranking:

    python run_sasrec.py --config=configs/xxx.yaml --ranking=full

or with sampled-based ranking

    python run_sasrec.py --config=configs/xxx.yaml --ranking=pool