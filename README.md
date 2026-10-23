# Urban Joint Encoder

可复现的时空-语义联合城市交通拥堵编码与预测流水线。

> **版本**：0.1.0  
> **Python**：>= 3.11  
> **目标**：在仅有表头缺失的导出矩阵（`feature_matrix_X.csv`、`adj_matrix.csv`、POI 矩阵）条件下，构建一个从数据审计、事件检测、语义图构建、时空编码到跨模态融合与统一评估的完整研究基线。

---

## 1. 项目概述

本项目面向上海市黄浦区蜂窝网格化的交通拥堵数据，设计并实现了一个**时空（Spatiotemporal, ST）+ 语义（Semantic, SE）联合编码器**。核心思想是：

- **时空分支**：使用 GCN + GRU 学习道路邻接结构上的交通速度演化规律。
- **语义分支**：基于历史同期分位数（Q10）与稳健 Z 分数，检测拥堵事件；将网格、POI 类别、拥堵事件与治理规则组织为异构图；通过关系感知图编码器得到语义嵌入。
- **跨模态融合**：使用带掩码的多头交叉注意力（Masked Cross-Attention）将每个网格的时空嵌入与对应语义上下文融合，输出联合预测。
- **因果性约束**：所有语义证据的观测截止点均严格早于预测目标行（`observation_cutoff = target_row - 1`），并内置数据泄漏审计。

---

## 2. 仓库结构

```text
urban-joint-encoder/
├── README.md                       # 本文件
├── pyproject.toml                  # uv 项目配置与依赖
├── uv.lock                         # 锁定依赖版本
├── configs/
│   └── joint_train.yaml            # 联合模型 smoke 训练配置（JSON 格式，兼容 YAML）
├── data/
│   └── raw/Main data/              # 原始数据（不参与 git）
│       ├── prediction_data/
│       │   ├── feature_matrix_X.csv    # [8640, 265] 交通速度矩阵
│       │   ├── adj_matrix.csv          # [265, 265] 道路邻接矩阵
│       │   ├── POI/
│       │   │   └── poi1.csv … poi7.csv # [265] 每类 POI 计数
│       │   └── precipitation/          # 徐家汇、浦东两个站点降水
│       └── POI/                    # 原始 POI shapefile（按类别）
├── notebooks/                      # 分步骤探索与可视化
│   ├── 01_data_overview.ipynb
│   ├── 02_spatiotemporal_encoder.ipynb
│   ├── 03_semantic_event_graph.ipynb
│   └── 04_cross_modal_fusion.ipynb
├── scripts/
│   └── sync_to_gpu.sh              # 同步代码到 GPU 服务器
├── src/urban_joint_encoder/        # 核心源码
│   ├── data_audit.py               # 只读数据审计与可视化
│   ├── data_loader.py              # 数据加载与训练集标准化
│   ├── temporal_dataset.py         # 时间窗口数据集
│   ├── train_st.py                 # Step 4：时空编码器训练
│   ├── semantic_pipeline.py        # Step 5：事件检测、语义图构建与训练
│   ├── causal_semantic_snapshots.py# Step 7：因果语义快照生成器
│   ├── joint_dataset.py            # 联合训练数据集
│   ├── train_joint.py              # Step 7A：联合模型 smoke 训练
│   ├── evaluate_joint.py           # Step 8.2：ST-only / Joint 统一评估
│   ├── ablate_semantics.py         # 语义消融实验入口
│   ├── cross_modal_alignment.py    # 跨模态对齐工具
│   ├── models/
│   │   ├── st_encoder.py           # GCN + GRU 时空编码器
│   │   ├── semantic_encoder.py     # 关系感知图编码器
│   │   ├── cross_attention.py      # Masked Cross-Attention 融合
│   │   └── joint_encoder.py        # 联合编码器（冻结 ST 分支）
│   └── visualization/              # 绘图工具
│       ├── st_plots.py
│       ├── semantic_plots.py
│       └── fusion_plots.py
├── tests/
│   └── test_joint_pipeline.py      # 因果性、梯度与联合前向测试
└── outputs/                        # 运行产物（不参与 git）
    ├── models/                     # 模型检查点
    ├── embeddings/                 # E_st / E_se 嵌入
    ├── metrics/                    # JSON / CSV 评估指标
    ├── figures/                    # 可视化图片
    ├── graphs/                     # 语义图（JSON / GraphML）
    ├── predictions/                # 预测结果 CSV
    └── alignment/                  # 跨模态对齐结果
```

---

## 3. 数据说明

| 文件 | 形状 | 含义 |
|------|------|------|
| `feature_matrix_X.csv` | `[8640, 265]` | 30 天 × 288 个 5 分钟槽的交通速度（局部平均速度，已完成张量补全） |
| `adj_matrix.csv` | `[265, 265]` | 基于道路连通关系的二值邻接矩阵（对称） |
| `POI/poi{1..7}.csv` | `[265]` | 7 类 POI 在每个网格的计数 |
| `precipitation/station{1,2}.csv` | `[8640]` | 徐家汇、浦东站点降水（毫米） |

> **注意**：原始导出文件均缺少表头、时间戳与地理 FID，因此本仓库所有代码均使用列索引作为节点标识，且不在任何阶段伪造时间戳或坐标。

---

## 4. 核心流水线

### Step 3 — 数据审计（`data_audit.py`）

- 读取所有矩阵并做形状、缺失值、对齐性检查。
- 绘制：时空热力图、道路邻接拓扑、POI 分布、代表性节点速度曲线。
- 输出：`outputs/metrics/data_audit.json`、`outputs/figures/*.png/*.svg`。

### Step 4 — 时空编码器（`train_st.py`）

- 模型：`STForecastModel` = `SpatialTemporalEncoder`（GCN + GRU） + 预测头。
- 输入：`[B, T=12, N=265, C=1]` 速度窗口；输出：`[B, N]` 下一时刻预测 + `[B, N, hidden_dim]` 嵌入 `E_st`。
- 数据划分：训练 70% / 验证 15% / 测试 15%（按时间顺序）。
- 训练集上拟合标准化器（`Standardizer`），避免泄漏。
- 早停与最佳检查点保存：`outputs/models/st_encoder_best.pt`。

### Step 5 — 语义事件图（`semantic_pipeline.py`）

- **事件检测**：使用历史同期（同天时刻 ±1 槽）的 Q10 阈值 + 稳健 Z 分数（MAD 稳定化）检测拥堵事件。
- **图构建**：节点类型包括 `SpatialGrid`、`POICategory`、`CongestionEvent`、`GovernanceRule`；关系类型见 `RELATIONS`。
- **图编码器**：`RelationalGraphEncoder`，通过关系权重进行两阶消息传递，联合优化边重建 BCE + 节点属性重建 MSE。
- 产物：`outputs/graphs/semantic_graph.{json,graphml}`、`outputs/embeddings/E_se.npy`、`outputs/models/semantic_encoder_best.pt`。

### Step 6 — 跨模态融合（`cross_attention.py`）

- `MaskedCrossModalFusion`：以 `E_st` 作为 query，`E_se` 作为 key/value，使用每个网格-语义节点允许掩码（`attention_mask`）避免无关语义节点参与。
- 门控残差连接（gated residual）输出联合嵌入 `E_joint`。

### Step 7A — 联合模型 smoke 训练（`train_joint.py`）

- `JointSpatiotemporalSemanticEncoder`：
  - 加载并**冻结**已训练的 Step 4 ST 编码器。
  - 使用全新的关系感知语义编码器（固定种子初始化）。
  - 通过 `MaskedCrossModalFusion` 融合，再经线性头预测。
- 在训练集内选取少量目标行进行可复现的 smoke 测试，验证：
  - 前向输出有限值；
  - 交叉注意力存在非零梯度并更新参数；
  - 不同目标时刻生成不同语义快照；
  - 数据泄漏审计通过（`outputs/metrics/joint_data_leakage_audit.json`）。

### Step 8.2 — 统一评估（`evaluate_joint.py`）

- 在验证集/测试集上同时评估三种模型：
  - **Persistence**：上一时刻速度作为基线；
  - **ST-only**：仅时空编码器；
  - **Joint**：时空-语义联合编码器。
- 输出：`outputs/metrics/model_comparison.csv`、`outputs/metrics/evaluation_report.json`、`outputs/figures/model_comparison.png`。
- 对指定 case 行（默认 7347）导出每个网格的预测对比 CSV。

---

## 5. 模型架构

### `SpatialTemporalEncoder`

```text
Input: [B, T, N, 1]
  ↓  A_norm @ X  (GCN 空间聚合)
Spatial projection + ReLU
  ↓
GRU(T, hidden_dim)  → per-node temporal encoding
  ↓
Output embedding E_st: [B, N, hidden_dim]
```

### `RelationalGraphEncoder`

```text
Input: node features X, edge_index, edge_type
  ↓
Relation-aware message passing layer 1 + ReLU
  ↓
Relation-aware message passing layer 2
  ↓
Output embedding E_se: [num_nodes, hidden_dim]
```

支持边重建（`edge_logits`）与属性重建（`reconstruct_attributes`）用于自监督训练。

### `JointSpatiotemporalSemanticEncoder`

```text
Traffic window X ──→ Frozen ST Encoder ──→ E_st [B, N, H]
                                            ↓  (per grid query)
Causal semantic snapshot ──→ Semantic Encoder ──→ E_se [M, H]
                                            ↓  Masked Cross-Attention
                              context + gate ──→ E_joint [B, N, H]
                                            ↓
                              Linear head ──→ prediction [B, N]
```

---

## 6. 安装与运行

本项目使用 [uv](https://github.com/astral-sh/uv) 作为包管理器。

```bash
# 克隆仓库后进入目录
cd urban-joint-encoder

# 创建虚拟环境并安装依赖
uv sync

# 激活环境（可选，uv run 会自动使用）
source .venv/bin/activate
```

### 运行各步骤

```bash
# Step 4：训练时空编码器
uv run --no-sync python -m urban_joint_encoder.train_st

# Step 5：构建语义图并训练语义编码器
uv run --no-sync python -m urban_joint_encoder.semantic_pipeline

# Step 7A：联合模型 smoke 训练
uv run --no-sync python -m urban_joint_encoder.train_joint

# Step 8.2：统一评估（验证集 + 测试集）
uv run --no-sync python -m urban_joint_encoder.evaluate_joint --device auto

# 运行单元测试
uv run --no-sync python -m unittest tests.test_joint_pipeline
```

### 命令行入口

`pyproject.toml` 注册了两个脚本：

```bash
urban-joint-encoder          # 占位入口，打印欢迎信息
train-joint                  # 等价于 python -m urban_joint_encoder.train_joint
```

---

## 7. 关键配置

`configs/joint_train.yaml` 控制联合模型 smoke 训练：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `seed` | 42 | 全局随机种子 |
| `sequence_length` | 12 | 历史时间步长度 |
| `hidden_dim` | 32 | 嵌入维度 |
| `batch_size` | 2 | smoke 训练批次大小 |
| `learning_rate` | 0.001 | Adam 学习率 |
| `smoke_epochs` | 2 | smoke 训练轮数 |
| `smoke_train_target_start` | 3000 | smoke 训练起始目标行 |
| `smoke_train_window_count` | 8 | smoke 训练窗口数 |
| `recent_event_horizon` | 36 | 动态事件回溯范围 |
| `max_dynamic_events` | 48 | 每快照最大动态事件数 |

---

## 8. 输出产物

| 路径 | 说明 |
|------|------|
| `outputs/models/st_encoder_best.pt` | Step 4 最佳时空模型 |
| `outputs/models/semantic_encoder_best.pt` | Step 5 最佳语义编码器 |
| `outputs/models/joint_smoke_test.pt` | Step 7A smoke 模型（非最终调优模型） |
| `outputs/embeddings/E_st.npy` | 时空嵌入 |
| `outputs/embeddings/E_se.npy` | 语义嵌入 |
| `outputs/graphs/semantic_graph.{json,graphml}` | 语义事件图 |
| `outputs/metrics/*` | 训练曲线、评估报告、泄漏审计、事件检测报告 |
| `outputs/figures/*` | 数据审计图、评估对比图 |
| `outputs/predictions/prediction_case_7347.csv` | case 行逐网格预测对比 |

---

## 9. 设计约束与已知限制

1. **无时间戳与地理标识**：原始数据缺少表头、时间列、FID 与坐标，因此所有分析基于列索引顺序；拓扑图使用 spring layout，不是地图。
2. **速度单位未标注**：`feature_matrix_X.csv` 中的速度物理单位在导出文件中未明确说明。
3. **事件是派生异常**：检测到的拥堵事件是统计意义上的速度异常，不等同于事故或官方拥堵事件标签。
4. **`POTENTIALLY_PROPAGATES_TO` 非因果证据**：该关系基于邻接+时滞+异常强度，仅为候选关联。
5. **`GovernanceRule` 与 `TRIGGERS_RULE` 为示例性质**：不代表真实政策或治理动作。
6. **降水数据未进入当前联合模型**：Step 7A 的语义快照暂未使用降水特征，以避免未来降水泄漏风险。

---

## 10. 同步到 GPU 服务器

```bash
bash scripts/sync_to_gpu.sh
```

默认同步到 `zju:/data/slx/project/urban-joint-encoder`，排除 `.git`、`.venv`、`data/`、`outputs/`、`external/` 等目录。请根据实际服务器地址修改脚本中的 `REMOTE_HOST` 与 `REMOTE_DIR`。

---

## 11. 引用与致谢

本项目为城市交通拥堵时空-语义联合编码的研究基线实现。若使用本仓库代码或数据产物，请注明数据来源与实现限制。
