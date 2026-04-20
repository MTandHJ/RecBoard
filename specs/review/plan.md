
# 代码规范化实现方案

## 总览

共 38 个模型需要修改，按变更类型分为 6 个阶段，由低风险机械性修改到高风险结构性修改逐步推进。每个阶段完成后通过 ruff 检查，确保不引入语法错误。

---

## Phase 1: 基础环境与格式化

**目标：** 配置 ruff，统一基础格式。

### 1.1 添加 ruff 配置

- 在项目根目录创建 `ruff.toml`，配置行长度、lint 规则等。
- 注意：需配置忽略 camelCase 变量名的 lint 规则（N806 等），因为项目约定使用 camelCase。

### 1.2 移除文件头部空行

- **涉及模型：** 全部 38 个（所有 `main.py` 顶部都有 1-2 个空行）。

### 1.3 拆分逗号合并的 import

- **涉及模型（10 个）：** AlphaRec, BM3, CAGCN, CCFRec, E4SRec, FREEDOM, LATTICE, MGCN, MMGCN, UniSRec。

### 1.4 运行 ruff format 和 ruff check --fix

- 对全部 38 个 `main.py` 执行格式化。

---

## Phase 2: 命名修正

**目标：** 修正 CLI 参数名、变量名中的不一致。

### 2.1 CLI 参数：下划线 → 连字符

同时修改 `main.py` 中的 `add_argument` 和对应 `configs/*.yaml` 中的 key。

| 模型 | 修改内容 |
|------|---------|
| AlphaRec | `--num_layers` → `--num-layers` |
| FREEDOM | `--num_ui_layers` → `--num-ui-layers`, `--num_ii_layers` → `--num-ii-layers` |
| LATTICE | `--num_ui_layers` → `--num-ui-layers`, `--num_ii_layers` → `--num-ii-layers` |
| MGCN | `--num_layers` → `--num-layers` |
| GLINT-RU | `--layer_norm_eps` → `--layer-norm-eps` |

**注意：** argparse 自动将连字符转为下划线作为属性名（`--num-ui-layers` → `cfg.num_ui_layers`），因此：
- `main.py` 中 `cfg.xxx` 的访问代码**无需修改**
- YAML key **无需修改**（保持 snake_case，如 `num_ui_layers`）

### 2.2 CLI 参数：同义词统一

| 模型 | 修改内容 |
|------|---------|
| BSARec | `--atten-dropout-rate` → `--attn-dropout-rate`（main.py + configs 中 `atten_dropout_rate` → `attn_dropout_rate`） |
| E4SRec | `--lora-dropout` → `--lora-dropout-rate`（main.py + configs 中 `lora_dropout` → `lora_dropout_rate`） |
| SGL | `--ssl-drop-ratio` → `--ssl-drop-rate`（main.py + configs 中 `ssl_drop_ratio` → `ssl_drop_rate`） |
| NeuMF | `--hidden-sizes` → `--hidden-dims`（main.py + configs 中 `hidden_sizes` → `hidden_dims`） |

**注意：** 同义词修改涉及属性名变更，需同时修改 main.py 中所有 `cfg.xxx` 的访问代码和 YAML key。

### 2.3 变量名：Embs → Embds

| 模型 | 修改内容 |
|------|---------|
| AlphaRec | `userEmbs`→`userEmbds`, `itemEmbs`→`itemEmbds`, `seqEmbs`→`seqEmbds` |
| Caser | `userEmbs`→`userEmbds`, `itemEmbs`→`itemEmbds`, `seqEmbs`→`seqEmbds` |
| FREEDOM | `userEmbs`→`userEmbds`, `itemEmbs`→`itemEmbds`, `iiEmbs`→`iiEmbds` |
| JGCF | `userEmbs`→`userEmbds`, `itemEmbs`→`itemEmbds`（保留已有的 `allEmbds`） |
| LATTICE | `userEmbs`→`userEmbds`, `itemEmbs`→`itemEmbds`, `iiEmbs`→`iiEmbds` |

---

## Phase 3: 结构性修改（模型类）

**目标：** 统一 `__init__` 签名、`fit()` 返回值、方法排列顺序。

### 3.1 `__init__` 签名简化：去掉多余参数，改为读 cfg

**涉及模型（23 个）：**
BERT4Rec, Caser, CAGCN, GCN, GRU4Rec, GTE, JGCF, LESSR, MF-BPR, MMGCN, NARM, NeuMF, NGCF, PairNorm, RUM, SASRec, SEvo, SimpleX, STOSA, UltraGCN, UniSRec

**修改方式：**
1. `__init__` 签名改为 `(self, dataset: freerec.data.datasets.RecDataSet) -> None`
2. 将原参数替换为 `cfg.xxx`
3. 同步修改 `main()` 中的模型实例化调用，去掉多余参数

**示例（SASRec）：**
```python
# Before
def __init__(self, dataset, maxlen=50, embedding_dim=64, dropout_rate=0.2, num_blocks=1, num_heads=2):
    ...
    self.embedding_dim = embedding_dim

# After
def __init__(self, dataset: freerec.data.datasets.RecDataSet) -> None:
    ...
    self.embedding_dim = cfg.embedding_dim
```

### 3.2 `fit()` 返回值改为字典

**涉及模型（14 个）：**
CAGCN, CCFRec, FREEDOM, JGCF, LATTICE, LightGCN, MGCN, MMGCN, NGCF, SGL, SimGCL, STOSA, UltraGCN, UniSRec

**同时需修改对应的 `CoachFor{Model}.train_per_epoch()` 中的解包逻辑。**

其余 24 个模型返回单 loss，也需改为 `return {'rec_loss': rec_loss}` 格式。

**示例（LightGCN）：**
```python
# Before (model)
def fit(self, data):
    ...
    return rec_loss, emb_loss

# After (model)
def fit(self, data):
    ...
    return {'rec_loss': rec_loss, 'emb_loss': emb_loss}

# Before (coach)
rec_loss, emb_loss = self.model(data)
loss = rec_loss + self.cfg.weight_decay * emb_loss

# After (coach)
losses = self.model(data)
loss = losses['rec_loss'] + self.cfg.weight_decay * losses['emb_loss']
```

### 3.3 方法排列顺序调整

需逐个模型检查方法顺序是否符合规范，不符合的进行重排。预计大部分模型已基本符合，需重排的是少数。

---

## Phase 4: 文档补充

**目标：** 添加类级别 docstring。

### 4.1 模型类 docstring

为每个模型主类和辅助类添加一句话 docstring。需逐个模型编写，内容需准确描述模型特点。

### 4.2 Coach 类 docstring

为每个 `CoachFor{Model}` 类添加简要 docstring。

---

## Phase 5: 配置文件与 README

**目标：** 统一 YAML 格式，丰富 README。

### 5.1 YAML 配置文件重排

对所有模型的 `configs/*.yaml` 按 `# Data → # Model → # Training → # Evaluation` 四组重新排列，添加分组注释。

### 5.2 README.md 生成

为每个模型生成/更新 README.md，包含：
- 标题 + 引用链接（保留现有）
- Notes（保留现有，如有）
- Usage 命令示例
- Hyperparameters 表格（从 `main.py` 的 `add_argument` 提取）
- Configuration Example（取一个代表性 yaml 内容）

---

## Phase 6: 最终检查

### 6.1 ruff format + check

全量运行 ruff，确保所有文件符合格式规范。

### 6.2 语法验证

对所有 `main.py` 运行 `python -m py_compile` 确保无语法错误。

---

## 执行策略

- **Phase 1-2** 为机械性修改，风险低，可批量处理。
- **Phase 3** 需逐个模型修改并验证逻辑正确性，是工作量最大的阶段。
- **Phase 4-5** 为文档类工作，不影响代码逻辑。
- **Phase 6** 为收尾验证。
- 每个 Phase 完成后单独 commit。
