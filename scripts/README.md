# scripts/ 工具与日常工作流

本目录是 CHIP2026「特定患者表型概念识别」的工具集。核心闭环：
**改进方法 → 在 dev 上预测 → 本地打分 → 记录 → 对比**。

## 目录结构（按角色分类）

```
scripts/
  core/            共享核心库（被其它脚本 import；也各自兼 CLI）
    baseline_dict.py   词典匹配基线 + 全局文本重建/就近归属等工具
    evaluate.py        复现四项 F1 + Score，可 --tag 记录实验
    explog.py          实验记录系统底座（纯库，被 evaluate/log_online/report 复用）
  s1_identify/     子任务1：表型识别（NER，需 GPU）
    build_concept_index.py  一次性：把 HPO 概念编码成向量索引
    baseline_sapbert.py     词典命中 + SapBERT 向量检索补召回
  s2_associate/    子任务2：患者-表型归属
    associate_llm.py   Qwen3-8B 门控式归属（--gate-only 推荐；需 GPU）
    assoc_gate.py      离线组合两份预测=就近∩LLM保留集（纯 Python，无 GPU）
  prep/            数据准备与检查
    split_data.py      训练集切成 train/dev（确定性、零重叠）
    inspect_data.py    数据统计 / offset 校验 / HPO 分支校验
  track/           评测记录与报表
    log_online.py      补录天池 A/B 榜线上成绩
    report.py          渲染 results/实验记录.md
  diag/            失分诊断
    diag_assoc.py      子任务2 归属失分三方归因（gold/就近/LLM）
  gpu_check.py     环境自检（验证 CUDA 可用）
  run_A.sh         A/B 榜一键推理链路（SapBERT 识别 → 门控式归属 → 自检）
```

> **import 约定**：非 core 脚本用
> `sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "core"))`
> 后按裸名 import core 里的模块。**新增脚本若要复用 core，照抄这一行**。
> 所有命令仍从**项目根目录**运行（路径相对根，不要 `cd scripts/`）。

---

## 一次性准备（只做一次）

```bash
# 从 80 篇训练集切出 train(64) / dev(16)，并生成题目版（清空答案）
python scripts/prep/split_data.py data/PatientPheX-V1-A/PatientPheX-train.jsonl
```

产出 `data/split/`：
- `train.jsonl` —— 64 篇，带答案，**训练**用。
- `dev.jsonl` —— 16 篇，带答案，作打分**尺子**（`--gold`）。
- `dev_input.jsonl` —— 同样 16 篇但**答案已清空**，作模型**输入**（防止偷看答案）。

SapBERT 识别还需先建概念向量索引（一次性，**需整卡 GPU**）：

```bash
python scripts/s1_identify/build_concept_index.py \
    --obo data/PatientPheX-V1-A/hp.obo --out-dir outputs/sapbert --fp16
```

---

## 日常工作流（每次做一个新方法，重复这套）

### 第 1 步：生成 dev 预测

用**题目版** `dev_input.jsonl` 作输入（模型看不到答案，公平）。以词典基线为例：

```bash
python scripts/core/baseline_dict.py \
    --input data/split/dev_input.jsonl \
    --obo   data/PatientPheX-V1-A/hp.obo \
    --out   pred_dev.jsonl
```

> 换成你自己的新方法脚本时，只要保证输出是**同样的提交格式 jsonl** 即可。
> 完整两子任务链路见文末「生成 A 榜提交文件」。

### 第 2 步：本地打分 + 自动记录

用**答案版** `dev.jsonl` 作 `--gold`。加 `--tag` 就会自动把本地四项 F1 记入
`results/experiments.jsonl`：

```bash
python scripts/core/evaluate.py \
    --gold data/split/dev.jsonl \
    --pred pred_dev.jsonl \
    --tag  sapbert_v1 \
    --note "接入 SapBERT 提召回"
```

- `--tag` 是实验主键。**同名 tag 再次记录会合并覆盖**（不会产生重复行）。
- `--note` 一句话写清这次改了什么。
- 想回填历史实验可加 `--ts "2026-08-28 16:30"` 手动指定时间。

### 第 3 步（可选）：提交天池后补录线上成绩

数字照抄天池成绩单那一行：

```bash
python scripts/track/log_online.py \
    --tag sapbert_v1 --board A \
    --score 0.61 --men 0.68 --doc 0.74 --mic 0.55 --mac 0.50 \
    --submit-ts "2026-08-29 10:00:00"
```

`--tag` 必须与第 2 步一致，才能挂到同一条实验上。

### 第 4 步：刷新报告

```bash
python scripts/track/report.py       # → results/实验记录.md
```

打开 [../results/实验记录.md](../results/实验记录.md) 看表格：本地/线上分数并排，
「较上条」列显示相对上一次实验的本地 Score 升降（🟢升/🔴降/⚪平）。

---

## 关键纪律（踩过的坑）

1. **dev 才能本地打分；A/B 榜测试集本地无法打分**（官方不给答案，`entities`/`association`
   本就为空）。对 A 榜的预测只能提交到天池，由平台算分。
2. **别把 dev 预测和 A 榜预测混用同一文件名**。dev 用 `pred_dev*.jsonl`，A 榜用
   `pred_A*.jsonl`。evaluate.py 已内置防呆：pred 与 gold 的 pmc_id 完全无交集时中止提示。
3. **train 与 dev 零重叠**：模型只在 train 上训练/调参，dev 仅用于打分，否则分数虚高。
4. **评测口径已验证与官方一致**：可随时 `python scripts/core/evaluate.py --selftest`
   用手算例复验评测脚本本身。
5. **全程 GPU**：识别（SapBERT）与归属（Qwen3-8B）需整卡开机；纯组合/打分/记录/诊断
   （assoc_gate、evaluate、track/*、diag/*）不吃 GPU，无卡模式即可。

---

## 生成 A 榜提交文件（要提交时）

一键链路（**整卡开机**；SapBERT 识别 → Qwen3-8B 门控式归属 → 提交前自检）：

```bash
bash scripts/run_A.sh                 # 默认跑 A 榜，产出 pred_A_gate.jsonl
```

或手动分两步：

```bash
# 子任务1：识别
python scripts/s1_identify/baseline_sapbert.py \
    --input data/PatientPheX-V1-A/PatientPheX-A.jsonl \
    --obo   data/PatientPheX-V1-A/hp.obo \
    --index-dir outputs/sapbert --out pred_A_sapbert.jsonl \
    --sim-threshold 0.95 --fp16

# 子任务2：门控式归属
python scripts/s2_associate/associate_llm.py \
    --input data/PatientPheX-V1-A/PatientPheX-A.jsonl \
    --pred  pred_A_sapbert.jsonl --out pred_A_gate.jsonl --gate-only
# 然后把 pred_A_gate.jsonl 上传天池（此步不打分）
```

> `pred_*.jsonl` / `submit*.jsonl` 已在 `.gitignore` 中忽略（可重新生成的产物）；
> `results/` 会进版本库（实验记录需长期留存）。
