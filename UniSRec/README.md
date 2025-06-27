

# UniSRec

[[official-code](https://github.com/RUCAIBox/UniSRec)]


> [!NOTE]
> No `word drop` operation and `pos item` filtering.


## Usage

Run with full ranking:

    torchrun --nproc-per-node 4 main.py --config=configs/xxx.yaml --ranking=full

or with sampled-based ranking

    torchrun --nproc-per-node 4 main.py --config=configs/xxx.yaml --ranking=pool