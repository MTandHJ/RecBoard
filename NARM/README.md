

# NARM

[[official-code](https://github.com/lijingsdu/sessionRec_NARM)]
[[RecBole](https://github.com/RUCAIBox/RecBole/blob/master/recbole/model/sequential_recommender/narm.py)]


## Usage

Run with full ranking

    python main.py --config=configs/xxx.yaml --ranking=full

or with sampled-based ranking

    python main.py --config=configs/xxx.yaml --ranking=pool