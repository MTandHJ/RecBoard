

# FREEDOM

[[official-code](https://github.com/enoche/FREEDOM)]


**Note:** The offical implementation did not use `weight_decay`, but I found it is a bit useful and thus performed this operation.


## Usage

Run with full-ranking

    python main.py --config=configs/xxx.yaml --ranking=full

or with sampled-based ranking

    python main.py --config=configs/xxx.yaml --ranking=pool