# SEvo

[[paper](https://arxiv.org/abs/2310.03032)] [[official-code](https://github.com/MTandHJ/SEvo)]

> [!NOTE]
> This implementation only provides the scripts for SASRec. Other recommendation models can be similarly trained by modifying the baselines introduced in [RecBoard](https://github.com/MTandHJ/RecBoard).

## Usage

Run with full-ranking:

    python main.py --config=configs/xxx.yaml --ranking=full

or with sampled-based ranking:

    python main.py --config=configs/xxx.yaml --ranking=pool

## Hyperparameters

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| --maxlen | int | 50 | Maxlen |
| --num-heads | int | 1 | Num Heads |
| --num-blocks | int | 2 | Num Blocks |
| --embedding-dim | int | 64 | Embedding Dim |
| --dropout-rate | float | 0.2 | Dropout Rate |
| --loss | str | "BCE" | Loss |
| --aggr | str | "neumann" | Aggr |
| --L | int | 3 | the number of layers for approximation |
| --beta3 | float | 0.9 | the `beta` in Eq. (6) |
| --H | int | 1 | the maximum walk length allowing for a pair of neighbors |
| --maxlen4graph | int | 50 | only the last `maxlen` items in a sequence will be used for construciton |

## Configuration Example

```yaml
# Data
root: ../../data
dataset: Amazon2014Beauty_550_LOU

# Model
maxlen: 50
num_heads: 1
num_blocks: 2
embedding_dim: 64
dropout_rate: 0.3
beta1: 0.9
beta2: 0.98
loss: BCE
L: 3
aggr: neumann
beta3: 0.99
H: 1
maxlen4graph: 50

# Training
epochs: 200
batch_size: 512
lr: 1.e-3
weight_decay: 0.1
optimizer: AdamWSEvo

# Evaluation
monitors: [LOSS, HitRate@1, HitRate@5, HitRate@10, NDCG@5, NDCG@10]
which4best: NDCG@10
```
