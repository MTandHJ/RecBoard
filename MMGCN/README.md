

# MMGCN

[[official-code](https://github.com/weiyinwei/MMGCN)]


## Usage

Run with full ranking:

    python main.py --config=configs/xxx.yaml --ranking=full

or with sampled-based ranking

    python main.py --config=configs/xxx.yaml --ranking=pool

The following command can be used to encode modality features:

    python preprocessing --dataset=XXX --download-images --require-visual-modality --require-textual-modality