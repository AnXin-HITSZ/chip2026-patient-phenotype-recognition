#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
train_gate.py —— 第 4 步：训练「患者-表型 门控分类器」（PyTorch 逻辑回归 / GPU）。

它做什么：
  读第 3 步 build_gate_dataset.py 产出的 gate_train.jsonl / gate_dev.jsonl
  （每行 = 一个实体的【原始特征 + label】），做「喂模型前的预处理」，用 PyTorch
  训练一个逻辑回归（= 单层神经网络），在 dev 上看效果，保存模型 + 预处理器。

为什么预处理放这里、而不放第 3 步：
  独热 / 标准化 / 缺失填充 因模型而异（LR 必须做，树模型不用），是「喂模型前」的事。
  第 3 步只吐原始信号，本步按模型需要加工。三处新手必踩的坑都在预处理里：
    坑① 类别特征(section_type)必须独热，不能编号(会造假的大小顺序)。
    坑② 缺失值(retrieved_score/hpo_depth/dist 可能是 None)不能填 0，要填中性值 + isnull 标志。
    坑③ 标准化的 mean/std **只能在 train 上算**，再套用到 dev；用 dev 统计量 = 泄露。
  Preprocessor 已把这三坑处理好（本步 GPU 无关，纯标准库，本地可 --dry-preprocess 验证）。

职责边界（本步不做）：
  * 不把 keep 决策接回子任务2（就近∩keep-set 的应用留给第 5 步 gate_apply.py）。
    本步只在「实体留/弃」这个二分类任务上评估 + 扫阈值看保留率，作训练是否学到东西的信号。
  * 阈值最终怎么挑(保 macro)由第 5/6 步接子任务2 后定；本步先把候选阈值表打出来。

GPU 纪律：训练用 GPU，检测不到 CUDA 直接退出（--dry-preprocess 除外，那步不碰 torch）。

用法：
    # 本地无卡：只验证预处理（形状/特征名/是否泄露），不训练
    python scripts/s2_associate/train_gate.py \
        --train data/split/gate_train.jsonl --dev data/split/gate_dev.jsonl \
        --dry-preprocess

    # 云上整卡：训练 + dev 评估 + 保存
    python scripts/s2_associate/train_gate.py \
        --train data/split/gate_train.jsonl --dev data/split/gate_dev.jsonl \
        --out-dir outputs/gate --epochs 300 --lr 0.05
"""
import argparse
import json
import math
import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except AttributeError:
    pass

# 复用 evaluate 的 P/R/F1（纯标准库，本地可导入）
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "core"))
from evaluate import prf  # noqa: E402


# ======================= 数据读取 =======================

def load_dataset(path):
    """读 gate_*.jsonl，返回 rows(list of dict)。每行含原始特征 + label。"""
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    if not rows:
        print("❌ 空数据集: %s" % path)
        sys.exit(2)
    return rows


# ======================= 预处理器（已写好，三坑都处理了，无需改） =======================
# 特征列分三类（顺序固定，保存后 gate_apply 必须一致）：
#   数值列 NUM_COLS：标准化(减均值除标准差)。其中 MISSABLE 列可能为 None。
#   二元列 BIN_COLS：0/1 标志，原样(不标准化)。
#   类别列 section_type：独热。
#   额外：对每个 MISSABLE 列生成一个 <col>_isnull 标志(坑②)。

NUM_COLS = ["retrieved_score", "mention_char_len", "mention_word_len",
            "hpo_depth", "rel_position", "dist_to_nearest_anchor_norm",
            "patient_count"]
MISSABLE = ["retrieved_score", "hpo_depth", "dist_to_nearest_anchor_norm"]
BIN_COLS = ["is_dict_hit", "has_negation_cue", "has_normal_cue"]
UNK_SECTION = "__UNK__"


class Preprocessor:
    """把原始特征 dict 列表转成数值矩阵。fit 只在 train 上做(坑③防泄露)。"""

    def __init__(self):
        self.sections = None      # 独热类别表(含 UNK)
        self.fill = {}            # 数值列缺失填充值 = train 非缺失均值(坑②)
        self.mean = {}            # 标准化均值(= fill，见推导)
        self.std = {}             # 标准化标准差(std=0 -> 1 防除零)
        self.feature_names = None # 最终每一维的名字(供解释 w / gate_apply 对齐)

    def fit(self, rows):
        # 数值列：非缺失均值作 fill 与 mean；再在填充后序列上算 std
        for c in NUM_COLS:
            vals = [r.get(c) for r in rows if r.get(c) is not None]
            m = sum(vals) / len(vals) if vals else 0.0
            self.fill[c] = m
            self.mean[c] = m       # 填充值=均值 => 填充后整列均值仍=m
            filled = [(r.get(c) if r.get(c) is not None else m) for r in rows]
            var = sum((x - m) ** 2 for x in filled) / len(filled) if filled else 0.0
            self.std[c] = math.sqrt(var) or 1.0
        # 类别列：train 见过的 section_type(已大写) + UNK 兜 dev 新类
        secs = sorted({(r.get("section_type") or UNK_SECTION) for r in rows})
        if UNK_SECTION not in secs:
            secs.append(UNK_SECTION)
        self.sections = secs
        # 组装最终特征名(顺序固定)
        names = list(NUM_COLS) + list(BIN_COLS)
        names += ["%s_isnull" % c for c in MISSABLE]
        names += ["section=%s" % s for s in self.sections]
        self.feature_names = names
        return self

    def transform(self, rows):
        """返回 (X: list[list[float]], y: list[int])。"""
        X, y = [], []
        for r in rows:
            vec = []
            for c in NUM_COLS:                       # 标准化(缺失填中性值)
                v = r.get(c)
                v = self.fill[c] if v is None else v
                vec.append((v - self.mean[c]) / self.std[c])
            for c in BIN_COLS:                        # 二元原样
                vec.append(float(r.get(c) or 0))
            for c in MISSABLE:                        # 缺失标志(坑②)
                vec.append(1.0 if r.get(c) is None else 0.0)
            st = r.get("section_type") or UNK_SECTION # 独热(坑①)
            if st not in self.sections:
                st = UNK_SECTION
            for s in self.sections:
                vec.append(1.0 if s == st else 0.0)
            X.append(vec)
            y.append(int(r.get("label", 0)))
        return X, y

    def to_dict(self):
        return {"sections": self.sections, "fill": self.fill, "mean": self.mean,
                "std": self.std, "feature_names": self.feature_names,
                "num_cols": NUM_COLS, "bin_cols": BIN_COLS, "missable": MISSABLE}


# ======================= 模型定义（你来写） =======================

def build_model(in_dim):
    """【TODO 你来写】构造逻辑回归模型 = 单层神经网络。

    要点：
      * import torch.nn as nn（torch 只能在函数内导入，模块级本地没有）
      * 逻辑回归就是一个线性层：nn.Linear(in_dim, 1)。
      * forward 返回**logits(未过 sigmoid)**，形状建议 squeeze 成一维 [N]，
        因为我们用 nn.BCEWithLogitsLoss（它内部含 sigmoid，数值更稳；
        千万别自己再 sigmoid 一次，会算两遍）。

    可以这样写(供参考，你补全)：
        import torch.nn as nn
        class GateLR(nn.Module):
            def __init__(self, d):
                super().__init__()
                self.linear = nn.Linear(d, 1)
            def forward(self, x):
                return self.linear(x).squeeze(-1)   # [N,1] -> [N]
        return GateLR(in_dim)
    """
    import torch.nn as nn
    class GateLR(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.linear = nn.Linear(d, 1)
        def forward(self, x):
            return self.linear(x).squeeze(-1)
    return GateLR(in_dim)


def train_model(model, X_train, y_train, device, epochs, lr, pos_weight=None):
    """【TODO 你来写核心训练循环】这是深度学习的心脏：前向 → loss → 反向 → 更新。

    骨架已把张量、优化器、损失函数、设备都备好(见下)。你只需在 for 循环里写那 5 行。

    参考(全批量梯度下降，数据小，最清晰；想要 mini-batch 可后续改)：
        import torch
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        for ep in range(epochs):
            optimizer.zero_grad()            # 1. 清空上一轮梯度
            logits = model(X_train)          # 2. 前向：得到 [N] logits
            loss = criterion(logits, y_train)# 3. 算损失(y 要 float)
            loss.backward()                  # 4. 反向：自动求梯度
            optimizer.step()                 # 5. 用梯度更新参数
            if (ep + 1) % 50 == 0:
                print("  epoch %d/%d  loss=%.4f" % (ep + 1, epochs, loss.item()))
        return model

    注：X_train 已是 [N,D] float32、y_train 已是 [N] float32、都在 device 上(见 main)。
    """
    import torch
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for ep in range(epochs):
        optimizer.zero_grad()
        logits = model(X_train)
        loss = criterion(logits, y_train)
        loss.backward()
        optimizer.step()
        if (ep + 1) % 50 == 0:
            print("  epoch %d/%d  loss=%.4f" % (ep + 1, epochs, loss.item()))
    return model


# ======================= dev 评估 + 阈值扫描（已写好） =======================

def predict_proba(model, X, device):
    """前向得 keep 概率(过 sigmoid)。仅推理，no_grad。"""
    import torch
    model.eval()
    with torch.no_grad():
        logits = model(X.to(device))
        probs = torch.sigmoid(logits).cpu().tolist()
    return probs


def sweep_thresholds(probs, y_true, thresholds):
    """在「实体留/弃」二分类上扫阈值：每个阈值算 P/R/F1 + 保留率。
       注意：这是**二分类**指标，不是最终子任务2 的 F1(那要接就近路由，第 5 步做)。
       但它能告诉你：分类器学到东西没有、阈值调高调低时保留率怎么变。"""
    print("\n  阈值   保留率   P(留)    R(留)    F1(留)")
    print("  " + "-" * 44)
    n = len(y_true)
    for th in thresholds:
        tp = fp = fn = 0
        kept = 0
        for p, y in zip(probs, y_true):
            pred = 1 if p >= th else 0
            kept += pred
            if pred == 1 and y == 1:
                tp += 1
            elif pred == 1 and y == 0:
                fp += 1
            elif pred == 0 and y == 1:
                fn += 1
        P, R, F1 = prf(tp, fp, fn)
        print("  %.2f   %5.1f%%   %.4f   %.4f   %.4f"
              % (th, 100.0 * kept / n if n else 0, P, R, F1))


# ======================= 主流程（已写好） =======================

def main():
    ap = argparse.ArgumentParser(description="训练门控分类器(PyTorch 逻辑回归)")
    ap.add_argument("--train", required=True, help="gate_train.jsonl(第 3 步产出)")
    ap.add_argument("--dev", required=True, help="gate_dev.jsonl(第 3 步产出)")
    ap.add_argument("--out-dir", default=os.path.join("outputs", "gate"),
                    help="保存模型 + 预处理器的目录")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--pos-weight", type=float, default=0.0,
                    help="正类权重(>0 启用，缓解类别不平衡；58/42 不严重，默认不开)")
    ap.add_argument("--dry-preprocess", action="store_true",
                    help="只跑预处理并打印(形状/特征名/统计)，不导入 torch、不训练；本地验证用")
    args = ap.parse_args()

    train_rows = load_dataset(args.train)
    dev_rows = load_dataset(args.dev)
    print("train %d 条，dev %d 条" % (len(train_rows), len(dev_rows)))

    # 预处理：fit 只在 train(坑③)，再 transform 两份
    pre = Preprocessor().fit(train_rows)
    Xtr, ytr = pre.transform(train_rows)
    Xdv, ydv = pre.transform(dev_rows)
    d = len(pre.feature_names)
    pos = sum(ytr)
    print("特征维度 d=%d；train 正样本占比 %.1f%%（%d/%d）"
          % (d, 100.0 * pos / len(ytr) if ytr else 0, pos, len(ytr)))

    if args.dry_preprocess:
        print("\n[--dry-preprocess] 特征名(共 %d 维)：" % d)
        for i, name in enumerate(pre.feature_names):
            print("  %2d. %s" % (i, name))
        print("\ntrain 第一条特征向量：")
        print("  ", [round(x, 3) for x in Xtr[0]])
        print("\n未训练(需 torch+GPU)。预处理逻辑确认后，去掉 --dry-preprocess 上云训练。")
        return

    # ↓↓↓ 以下需 torch + GPU ↓↓↓
    import torch

    if not torch.cuda.is_available():
        print("❌ 未检测到 GPU（torch.cuda.is_available()=False）。本项目训练全程用 GPU，已中止。")
        print("   先跑 `python scripts/gpu_check.py`；只想验证预处理用 --dry-preprocess。")
        sys.exit(2)
    device = torch.device("cuda:0")
    print("使用 GPU :", torch.cuda.get_device_name(0))

    # 张量化：X [N,D] float32；y [N] float32(BCEWithLogitsLoss 要 float)
    X_train = torch.tensor(Xtr, dtype=torch.float32, device=device)
    y_train = torch.tensor(ytr, dtype=torch.float32, device=device)
    X_dev = torch.tensor(Xdv, dtype=torch.float32)
    pw = torch.tensor([args.pos_weight], device=device) if args.pos_weight > 0 else None

    # —— 你写的两块：建模 + 训练 ——
    model = build_model(d).to(device)
    print("开始训练：epochs=%d lr=%g%s"
          % (args.epochs, args.lr, "  pos_weight=%.2f" % args.pos_weight if pw is not None else ""))
    model = train_model(model, X_train, y_train, device, args.epochs, args.lr, pos_weight=pw)

    # dev 评估：二分类阈值扫描
    probs = predict_proba(model, X_dev, device)
    print("\n===== dev「实体留/弃」二分类·阈值扫描 =====")
    sweep_thresholds(probs, ydv, [0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    print("  (最终阈值挑选要接子任务2就近路由，见第 5 步 gate_apply.py)")

    # 保存模型 + 预处理器 + dev 概率(供第 5 步扫阈值接子任务2)
    os.makedirs(args.out_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(args.out_dir, "gate_lr.pt"))
    with open(os.path.join(args.out_dir, "gate_preproc.json"), "w", encoding="utf-8") as f:
        json.dump(pre.to_dict(), f, ensure_ascii=False, indent=2)
    with open(os.path.join(args.out_dir, "dev_probs.jsonl"), "w", encoding="utf-8") as f:
        for r, p in zip(dev_rows, probs):
            f.write(json.dumps({"pmc_id": r.get("pmc_id"), "offset": r.get("offset"),
                                "identifier": r.get("identifier"), "keep_prob": p,
                                "label": r.get("label")}, ensure_ascii=False) + "\n")
    print("\n已保存到 %s：gate_lr.pt / gate_preproc.json / dev_probs.jsonl" % args.out_dir)
    print("下一步：第 5 步 gate_apply.py 加载模型 -> keep 概率 -> 就近路由 ∩ keep 集 -> 打子任务2分。")


if __name__ == "__main__":
    main()
