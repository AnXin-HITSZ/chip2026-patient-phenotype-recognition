# CHIP2026 面向生物医学全文的特定患者表型概念识别

> CHIP 2026 评测一 · Patient-Specific Phenotype Concept Recognition in Full-Text Biomedical Literature
> 阿里天池赛事 ID：**532512** · 主办：大连理工大学（DUTIR-BioNLP）

从 PMC 生物医学文献**全文**中，自动识别表型概念并标准化到人类表型本体（HPO），并进一步建立**特定患者与表型之间的关联**。任务含两个子任务，最终以加权 F1（Score）排名。

| | |
|---|---|
| 天池平台 | https://tianchi.aliyun.com/competition/entrance/532512 |
| CHIP 官网评测页 | http://cips-chip.org.cn/2026/eval1 |
| 主办方数据仓库 | https://github.com/DUTIR-BioNLP/PatientPheX |
| 数据集 | PatientPheX（200 篇 PMC 全文；train 80 / dev-A 20 / test-B 100） |
| HPO 版本 | `hp/releases/2026-06-23`，仅评 Phenotypic abnormality（HP:0000118）分支 |
| 模型约束 | 参数 ≤ 10B；禁用外部人工标注表型数据集 |
| 报名截止 | 9 月 27 日（A 榜截止同日；B 榜 9/28–10/1；会议 10/30–11/1） |

## 两个子任务

- **子任务 1 — 全文表型概念识别**：不区分患者，识别全文所有表型概念及其位置，并标准化映射到 HPO。评测 mention-level 与 document-level 两个 F1。
- **子任务 2 — 特定患者表型概念识别**：给定患者，输出该患者相关表型的 HPO ID 集合。评测 micro-F1 与 macro-F1。

最终得分：`Score = 0.25 × (F1_men + F1_doc) + 0.25 × (F1_micro + F1_macro)`，四项等权。

## 文档目录

赛题资料已整理进 [docs/](docs/)。前五篇为**官方赛题的忠实转写**，后两篇为**参考补充**（已明确标注，非官方原文）：

| 文档 | 内容 |
|---|---|
| [01-任务概览](docs/01-任务概览.md) | 背景、任务内容、技术与资源约束 |
| [02-数据集说明](docs/02-数据集说明.md) | 数据规模、文件清单、JSONL 字段、样例、HPO 本体 |
| [03-评测指标](docs/03-评测指标.md) | 四类 F1 计算、总分公式、否定/复合/无 ID 判定 |
| [04-提交格式](docs/04-提交格式.md) | 提交文件格式、提交方式与次数限制 |
| [05-赛程与规则](docs/05-赛程与规则.md) | 日程、参赛规则、奖励、材料提交、组织者、官方基线 |
| [06-技术背景](docs/06-技术背景.md) | *（参考补充）* HPO 结构、PMC/BioC 全文格式、offset 约定 |
| [07-基线与打榜策略](docs/07-基线与打榜策略.md) | *（参考补充）* 基线方法、子任务解题思路、LLM+RAG、数据实测注意点、提交自检清单 |

## 目录结构

```
.
├── README.md
├── docs/                       # 赛题整理文档（见上表）
├── data/
│   └── PatientPheX-V1-A/       # A 榜数据包（需在天池登录后下载）
│       ├── PatientPheX-train.jsonl   # 训练集 80 篇，含答案
│       ├── PatientPheX-A.jsonl       # A 榜测试集 20 篇，无答案
│       ├── PatientPheX-B.jsonl       # B 榜测试集 100 篇（B 榜阶段释放）
│       ├── submit_pred_ex.jsonl      # 提交格式示例
│       ├── hp.obo                    # HPO 本体（hp/releases/2026-06-23）
│       └── README.md                 # 官方数据说明
└── scripts/
    └── inspect_data.py         # 数据统计 / offset 校验 / HPO 分支校验工具
```

> 数据文件需在天池平台**登录并报名后下载**（本仓库不分发数据）。当前 `data/` 下的 A 榜数据由参赛者自行下载放入。

## 数据速览

| 统计项 | 训练集 | 开发集(A) | 测试集(B) | 总共 |
|---|---|---|---|---|
| 文献数 | 80 | 20 | 100 | 200 |
| 患者数 | 209 | 53 | 244 | 506 |
| 表型实体提及数 | 6027 | 1563 | 7603 | 15193 |
| 唯一 HPO ID 数 | 1662 | 551 | 1863 | 2874 |
| 患者表型关联数 | 2307 | 563 | 2786 | 5656 |

以上统计已与本仓库实际发布数据核对一致，详见 [02-数据集说明](docs/02-数据集说明.md)。

## 快速开始

```bash
# 统计数据规模 + 章节分布
python scripts/inspect_data.py data/PatientPheX-V1-A/PatientPheX-train.jsonl

# 追加：offset 对齐校验 + HPO 分支/版本校验
python scripts/inspect_data.py data/PatientPheX-V1-A/PatientPheX-train.jsonl \
    --obo data/PatientPheX-V1-A/hp.obo --check-offset --check-hpo
```

`inspect_data.py` 仅依赖 Python 标准库，可复算数据规模、按 `section_type` 统计分布、校验每个实体/患者提及的 `(offset, length, text)` 是否与重建的全局全文对齐，并检查 `identifier` 是否落在指定版本 HPO 的 HP:0000118 分支内。

## 联系与致谢

- 组织单位：大连理工大学 —— 罗凌、王健、孙媛媛、林鸿飞；联系人：朱旋律（xlzhu@mail.dlut.edu.cn）
- 材料提交邮箱：dutir_bionlp@163.com

> 本仓库中的赛题文字均转写自 CHIP 官网评测页与主办方 GitHub 仓库，如与官方最新表述冲突，以官方为准。
