

# NeuMF

[[official-code](https://github.com/hexiangnan/neural_collaborative_filtering)]

**Note:** A implementation of NeuMF without pre-training. Only `weight_decay` is re-searched.


## Usage

Run with full-ranking

    python main.py --config=configs/xxx.yaml --ranking=full

or with sampled-based ranking

    python main.py --config=configs/xxx.yaml --ranking=pool
