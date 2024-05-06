

# SimpleX

[[official-code](https://github.com/xue-pai/SimpleX)]


**Note:** This implementation includes `average pooling` only, and the `reduce_learning_rate` operation will not be performed here.
Besides, the user historic interacted items will not be truncated by `maxlen`.


## Usage

Run with full-ranking

    python main.py --config=configs/xxx.yaml --ranking=full

or with sampled-based ranking

    python main.py --config=configs/xxx.yaml --ranking=pool