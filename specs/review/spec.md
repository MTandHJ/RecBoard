

# 代码规范

RecBoard 是一个基于 freerec 框架实现不同推荐系统 baselines 的仓库，本规范旨在统一约 47 个模型实现的代码风格、文档和结构。

**产出形式：** 规范文档 + 自动修改代码

---

## 1. 代码风格与格式

### 1.1 格式化工具

- 使用 **ruff** 作为格式化和 lint 工具，遵循其默认规则（含 import 排序、行长度等）。

### 1.2 变量命名

- 局部变量采用 **camelCase** 风格（保持现有主流风格）。
- 缩写规则：
  - **Embds**：可学习的嵌入向量（如 `userEmbds`, `itemEmbds`, `posEmbds`）。
  - **Feats**：预训练/外部加载的特征向量（如 `vFeats`, `tFeats`）。
  - 不允许使用 `Embs` 等其他缩写形式。

### 1.3 CLI 参数命名

- 统一使用**连字符**分隔（如 `--embedding-dim`, `--num-heads`, `--dropout-rate`）。
- 禁止使用下划线（如 ~~`--num_ui_layers`~~, ~~`--layer_norm_eps`~~）。

### 1.4 同义参数命名统一

以下参数在不同模型中存在变体，需统一为规范形式：

| 类别 | 规范形式 | 禁止形式 |
|------|---------|---------|
| dropout 后缀 | `-rate`（如 `--dropout-rate`） | ~~`-ratio`~~, ~~无后缀~~ |
| attention 缩写 | `attn`（如 `--attn-dropout-rate`） | ~~`atten`~~ |
| 多层隐藏维度 | `--hidden-dims` | ~~`--hidden-sizes`~~ |

完整的 dropout 参数命名规则：
- 单一 dropout：`--dropout-rate`
- 按位置区分：`--emb-dropout-rate`, `--hidden-dropout-rate`, `--attn-dropout-rate`
- 模块特有：`--{module}-dropout-rate`（如 `--adaptor-dropout-rate`, `--lora-dropout-rate`）

### 1.5 文件头部

- 文件顶部不留多余空行，直接从 import 开始。

---

## 2. 文档与注释

### 2.1 类级别 docstring（强制）

- 每个模型类和辅助模块类**必须**有 docstring。
- 内容为**简述**：模型名称 + 一句话描述，无需引用论文。
- 格式示例：
  ```python
  class LightGCN(freerec.models.GenRecArch):
      """Light Graph Convolution Network for recommendation."""
  ```

### 2.2 方法级别注释（不强制）

- 核心方法（`encode`, `fit`, `recommend_from_*` 等）的 docstring **不强制**。
- shape 行内注释（如 `# (B, S, D)`）**不强制**，但如有则保持准确。

### 2.3 Coach 类 docstring（强制）

- 每个 `CoachFor{Model}` 类也需要简要 docstring。

---

## 3. 代码结构与模式

### 3.1 文件级结构（强制顺序）

`main.py` 内的模块必须按以下顺序排列：

```
1. imports
2. freerec.declare(version='x.x.x')
3. cfg 配置（Parser + add_argument + set_defaults + compile）
4. 辅助类（如 PointWiseFeedForward、IISide 等，如有）
5. 模型类（继承 GenRecArch / SeqRecArch / PredRecArch）
6. Coach 类（CoachFor{Model}）
7. main() 函数
8. if __name__ == "__main__": main()
```

### 3.2 模型类内方法顺序（强制）

```
1. __init__()
2. reset_parameters()
3. sure_trainpipe()
4. sure_validpipe()（如有覆写）
5. sure_testpipe()（如有覆写）
6. 模型特有的辅助方法（如 mark_position, after_one_block 等）
7. encode()
8. fit()
9. reset_ranking_buffers()（如有）
10. recommend_from_full()
11. recommend_from_pool()
```

### 3.3 超参传递

- 模型超参**直接从 `cfg` 读取**，不通过构造函数参数传递。
- 即 `__init__` 签名统一为 `__init__(self, dataset: freerec.data.datasets.RecDataSet) -> None`。
- 示例：
  ```python
  # 正确
  class SASRec(freerec.models.SeqRecArch):
      def __init__(self, dataset: freerec.data.datasets.RecDataSet) -> None:
          super().__init__(dataset)
          self.num_blocks = cfg.num_blocks
          ...

  # 错误
  class SASRec(freerec.models.SeqRecArch):
      def __init__(self, dataset, maxlen=50, embedding_dim=64, ...) -> None:
          ...
  ```

### 3.4 fit() 返回值

- `fit()` 统一返回 **字典**，key 为 loss 名称，value 为 tensor。
- Coach 内按需组合各 loss。
- 示例：
  ```python
  # 单 loss
  def fit(self, data):
      ...
      return {'rec_loss': rec_loss}

  # 多 loss
  def fit(self, data):
      ...
      return {'rec_loss': rec_loss, 'emb_loss': emb_loss}
  ```

### 3.5 encode() 签名

- **不强制统一**。GenRecArch 的 `encode()` 可无参数，SeqRecArch 的 `encode(data)` 可接收数据字典，按各模型需求自行决定。

### 3.6 Coach 类

- 每个模型**保留自定义的 `CoachFor{Model}` 类**，即使逻辑相似也不复用。
- 命名统一为 `CoachFor{ModelName}`。

### 3.7 main() 函数

- 基本结构统一，允许按架构类型（GenRecArch / SeqRecArch / PredRecArch）存在差异。
- 统一模板：
  ```python
  def main():
      try:
          dataset = getattr(freerec.data.datasets, cfg.dataset)(root=cfg.root)
      except AttributeError:
          dataset = freerec.data.datasets.RecDataSet(cfg.root, cfg.dataset, tasktag=cfg.tasktag)

      model = ModelName(dataset)

      trainpipe = model.sure_trainpipe(...)
      validpipe = model.sure_validpipe(...)
      testpipe = model.sure_testpipe(...)

      coach = CoachForModelName(
          dataset=dataset,
          trainpipe=trainpipe,
          validpipe=validpipe,
          testpipe=testpipe,
          model=model,
          cfg=cfg
      )
      coach.fit()
  ```

### 3.8 reset_parameters()

- 各模型**自行实现**，但位置统一放在 `__init__()` 之后（方法顺序第 2 位）。

---

## 4. 配置文件规范（configs/*.yaml）

### 4.1 字段分组与顺序（强制）

配置文件按四组排列，组间空行分隔，每组以注释标记：

```yaml
# Data
dataset: Yelp2018_10104811_ROU
root: ../../data

# Model
embedding_dim: 64
num_layers: 4

# Training
epochs: 1000
batch_size: 2048
optimizer: adam
lr: 1.e-3
weight_decay: 1.e-4

# Evaluation
monitors: [LOSS, Recall@10, Recall@20, NDCG@10, NDCG@20]
which4best: NDCG@20
```

### 4.2 分组规则

| 分组 | 包含字段 |
|------|---------|
| **Data** | `dataset`, `root`, `tasktag`（如有）, `fields`（如有） |
| **Model** | 模型特有超参（如 `embedding_dim`, `num_heads`, `num_blocks`, `dropout_rate`, `loss` 等） |
| **Training** | `epochs`, `batch_size`, `optimizer`, `lr`, `weight_decay`, `lr_scheduler`（如有）, `eval_freq`（如有） |
| **Evaluation** | `monitors`, `which4best` |

### 4.3 key 命名

- YAML key 使用 **snake_case**（与 argparse 转换后的属性名一致）。
- 如 `embedding_dim`, `num_heads`, `dropout_rate`, `weight_decay`。

---

## 5. 模型 README.md 规范

### 5.1 结构模板（强制）

每个模型的 README.md 必须包含以下部分：

```markdown
# ModelName

[[official-code](url)]（如有）

## Usage

Run with full-ranking:

    python main.py --config=configs/xxx.yaml --ranking=full

or with sampled-based ranking:

    python main.py --config=configs/xxx.yaml --ranking=pool

## Hyperparameters

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| --embedding-dim | int | 64 | Embedding dimension |
| --num-layers | int | 3 | Number of GCN layers |
| ... | ... | ... | ... |

## Configuration Example

\```yaml
# Data
dataset: Yelp2018_10104811_ROU
root: ../../data

# Model
embedding_dim: 64
num_layers: 4

# Training
epochs: 1000
batch_size: 2048
optimizer: adam
lr: 1.e-3
weight_decay: 1.e-4

# Evaluation
monitors: [LOSS, Recall@10, Recall@20, NDCG@10, NDCG@20]
which4best: NDCG@20
\```
```

### 5.2 各部分说明

- **标题 + 引用链接**：模型名 + 官方代码/参考实现链接。
- **Usage**：列出运行命令。对于 PredRecArch 类模型无 `--ranking` 参数，只写基本命令。
- **Hyperparameters**：表格列出该模型**特有**的 CLI 参数（不含 freerec 通用参数如 `--lr`, `--epochs` 等）。
- **Configuration Example**：贴出一个具体的配置文件内容作为参考。

### 5.3 可选部分

- **Notes**：如有实现与原论文的差异说明，放在 Usage 之后、Hyperparameters 之前。

---

## 6. 不在本次规范范围内

- freerec 框架本身的代码。
