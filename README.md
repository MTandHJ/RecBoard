

![](srcs/logo.png)


<h4 align="center">
    <p>
        <a href="https://github.com/MTandHJ/freerec">FreeRec</a> |
        <a href="https://www.mtandhj.com/RecBoard">LeaderBoard</a>
    </p>
</h4>

This repository collects some classic recommendation methods, and I hope you will find it helpful.


## Directory Structure

The directory structure should be organized as follows, assuming that `root` is set to `../../data`.


```
┌── data # the 'root' path of data
│	├── Processed
│	│	├── Amazon2014Beauty_550_LOU # the training data
│	│	└── ...
│	├── Amazon2014Beauty.zip # the raw data
│	└── ...
└── RecBoard # collection of recommendation baselines
	├── DeepFM
	├── FREEDOM
	├── GRU4Rec
	├── LightGCN
	├── MF-BPR
	├── SASRec
            ├── configs # the configs for various datasets
            ├── main.py
            └── ...
	└── ...
```