


This is an official implementation of [Graph-enhanced Optimizers for <u>S</u>tructure-aware Recommendation Embedding <u>Evo</u>lution](https://arxiv.org/abs/2310.03032) based on the latest `FreeRec`. For precise reproduction of the results reported in the paper, please refer to the [0.4.3](https://github.com/MTandHJ/SEvo) branch.


## Requirements

Python >= 3.9 | [PyTorch >=2.0](https://pytorch.org/) | [0.6.0 <= TorchData <= 0.8.0](https://github.com/pytorch/data) | [PyG >=2.3](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html#) | [FreeRec >= 0.8.7](https://github.com/MTandHJ/freerec)

Refer to [here](https://github.com/MTandHJ/freerec/blob/master/dataset%20processing.md) for dataset preparation. For the [0.4.3](https://github.com/MTandHJ/SEvo/tree/0.4.3) branch, the datasets can be automatically downloaded.

## Usage

We provide configs for the Neumann series approximation with re-scaling. You can re-run them and try some other hyperparameters:

```
python main.py --config=configs/xxx.yaml --optimizer=AdamWSEvo --aggr=neumann --L=3 --beta3=0.99 --H=1
```

- optimizer: AdamWSEvo|AdamW|AdamSEvo|Adam|SGDSEvo|SGD
- aggr: neumann|iterative
- L: the number of layers for approximation
- beta3: $\beta$
- H: The maximum walk length allowing for a pair of neighbors


**Note:** This implementation only provides the scripts for SASRec. Other recommendation models can be similarly trained by modifying the baselines introduced in [RecBoard](https://github.com/MTandHJ/RecBoard).