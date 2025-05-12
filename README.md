

![](srcs/logo.png)


<h4 align="center">
    <p>
        <a href="https://github.com/MTandHJ/freerec">FreeRec</a> |
        <a href="https://painted-lilac-f2f.notion.site/Baselines-43ed27a7e7d54e3390fbbcbb293df485?pvs=4">LeaderBoard</a>
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

>![NOTE]
> For details on how to convert the raw data `Amazon2014Beauty.zip` into the training dataset `Amazon2014Beauty_550_LOU`, please refer to [Dataset Processing](https://github.com/MTandHJ/freerec/blob/master/dataset%20processing.md).
