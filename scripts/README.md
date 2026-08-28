# scripts/ 工具与日常工作流

本目录是 CHIP2026「特定患者表型概念识别」的**纯标准库**工具集（无需第三方依赖）。
围绕一个核心闭环：**改进方法 → 在 dev 上预测 → 本地打分 → 记录 → 对比**。

## 脚本清单

| 脚本 | 作用 | 输入 → 输出 |
|---|---|---|
| [inspect_data.py](inspect_data.py) | 数据统计 / offset 校验 / HPO 分支校验 | jsonl (+obo) → 终端报告 |
| [split_data.py](split_data.py) | 训练集切成 train/dev（确定性、零重叠） | train.jsonl → `data/split/` 三文件 |
| [baseline_dict.py](baseline_dict.py) | 词典匹配基线（阶段 0，零深度学习） | 题目版 jsonl + obo → 预测 jsonl |
| [evaluate.py](evaluate.py) | 复现四个 F1 + Score，可自动记录实验 | gold + pred → 终端分数 (+记录) |
| [explog.py](explog.py) | 实验记录系统底座（被下面三者复用） | — |
| [log_online.py](log_online.py) | 补录天池 A/B 榜线上成绩 | 命令行数字 → 记录 |
| [report.py](report.py) | 生成人类可读表格 | 记录 → `results/实验记录.md` |

---

## 一次性准备（只做一次）

```bash
# 从 80 篇训练集切出 train(64) / dev(16)，并生成 dev 题目版（清空答案）
python scripts/split_data.py data/PatientPheX-V1-A/PatientPheX-train.jsonl
```

产出 `data/split/`：
- `train.jsonl` —— 64 篇，带答案，**训练**用。
- `dev.jsonl` —— 16 篇，带答案，作打分**尺子**（`--gold`）。
- `dev_input.jsonl` —— 同样 16 篇但**答案已清空**，作模型**输入**（防止偷看答案）。

---

## 日常工作流（每次做一个新方法，重复这套）

### 第 1 步：生成 dev 预测

用**题目版** `dev_input.jsonl` 作输入（模型看不到答案，公平）：

```bash
python scripts/baseline_dict.py \
    --input data/split/dev_input.jsonl \
    --obo   data/PatientPheX-V1-A/hp.obo \
    --out   pred_dev.jsonl
```

> 换成你自己的新方法脚本时，只要保证输出是**同样的提交格式 jsonl** 即可。

### 第 2 步：本地打分 + 自动记录

用**答案版** `dev.jsonl` 作 `--gold`。加 `--tag` 就会自动把本地四项 F1 记入
`results/experiments.jsonl`（时间自动取当前本地时间，无需手填）：

```bash
python scripts/evaluate.py \
    --gold data/split/dev.jsonl \
    --pred pred_dev.jsonl \
    --tag  sapbert_v1 \
    --note "接入 SapBERT 提召回"
```

- `--tag` 是实验主键。**同名 tag 再次记录会合并覆盖**（不会产生重复行）。
- `--note` 一句话写清这次改了什么（未来的你会感谢现在的你）。
- 想回填历史实验可加 `--ts "2026-08-28 16:30"` 手动指定时间。

### 第 3 步（可选）：提交天池后补录线上成绩

线上分数是花「提交次数」买来的，务必记录。数字照抄天池成绩单那一行：

```bash
python scripts/log_online.py \
    --tag sapbert_v1 --board A \
    --score 0.61 --men 0.68 --doc 0.74 --mic 0.55 --mac 0.50 \
    --submit-ts "2026-08-29 10:00:00"
```

`--tag` 必须与第 2 步一致，才能挂到同一条实验上。

### 第 4 步：刷新报告

```bash
python scripts/report.py       # → results/实验记录.md
```

打开 [../results/实验记录.md](../results/实验记录.md) 看表格：本地/线上分数并排，
「较上条」列显示相对上一次实验的本地 Score 升降（🟢升/🔴降/⚪平）。

---

## 关键纪律（踩过的坑）

1. **dev 才能本地打分；A/B 榜测试集本地无法打分**（官方不给答案，`entities`/`association`
   本就为空）。对 A 榜的预测只能提交到天池，由平台算分。
2. **别把 dev 预测和 A 榜预测混用同一文件名**。建议：dev 用 `pred_dev.jsonl`，
   A 榜提交用 `submit_A.jsonl`。evaluate.py 已内置防呆：当 pred 与 gold 的
   pmc_id 完全无交集时会中止并提示「可能用错文件」。
3. **train 与 dev 零重叠**：模型只在 train 上训练/调参，dev 仅用于打分，
   否则分数虚高（数据泄漏）。
4. **评测口径已验证与官方一致**：官方四项 F1 代入 Score 公式与天池显示逐位吻合，
   可放心用本地 dev Score 指导迭代。可随时 `python scripts/evaluate.py --selftest`
   用手算例复验评测脚本本身。

---

## 生成 A 榜提交文件（要提交时）

```bash
python scripts/baseline_dict.py \
    --input data/PatientPheX-V1-A/PatientPheX-A.jsonl \
    --obo   data/PatientPheX-V1-A/hp.obo \
    --out   submit_A.jsonl
# 然后把 submit_A.jsonl 上传天池（此步不打分）
```

> `pred_*.jsonl` / `submit*.jsonl` 已在 `.gitignore` 中忽略（可重新生成的产物）；
> `results/` 会进版本库（实验记录需长期留存）。
