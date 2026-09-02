#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
gate_apply.py —— 第 5 步：把训练好的门控分类器接回子任务 2，兑现整件事的初衷。

它做什么（一句话）：
  加载第 4 步训练好的 LR（gate_lr.pt + gate_preproc.json）→ 给每个 SapBERT 识别出的实体
  打一个 keep 概率 → **先按阈值筛掉噪声实体**，再把留下的实体交给原来的「就近路由」
  associate() 挂给最近患者 → 得到新的 association → 打子任务 2 分。

为什么这样就对了（职责分离，看 baseline_dict.associate 源码后得到的关键洞察）：
  associate(doc, entities) 是「对传进来的每个实体，各自独立挂到最近患者」。所以：
      「就近路由 ∩ keep 集」  ==  先用 keep 概率过滤实体列表，再把幸存者喂给原样的 associate()
  * 就近路由 决定「挂给哪个患者」——完全不动。
  * 门控     决定「该不该挂给任何人」——就是这一步的 keep 阈值过滤。
  两个职责从不混淆：我们只在「喂进 associate 之前」拦一道，routing 逻辑一行都不改。

两条铁律（错一点，模型就吃到与训练时不一致的输入）：
  铁律①（特征一致）：特征必须用 build_gate_dataset.extract_features **同一份代码**算，
                      绝不在这里重写一遍 feat_*。直接 import 复用。
  铁律②（预处理一致）：标准化的 mean/std、独热的 section 表、缺失填充值，必须复用训练时
                      保存进 gate_preproc.json 的**同一套**（Preprocessor.from_dict 复原），
                      绝不在 apply 阶段重新在测试数据上 fit —— 那等于 train/apply 分布错位
                      （同一个实体在两处被标准化成不同的数）。

子任务 1（识别）完全不动：输出的 entities 原样透传 SapBERT 预测，menF1/docF1 不受影响。
门控只改 association（子任务 2）。这正是「只做归属、不做识别」的边界。

GPU 纪律（本步的特例说明）：
  本步是「特征构造 + 一个单层线性模型的前向」——[N,24]@[24,1] 的矩阵乘，几千行、CPU 瞬间完成，
  是**不吃 GPU 的活**。按项目约定，这类活可在 AutoDL 无卡模式（~0.1 元/时）跑，不算违反
  「全程 GPU」纪律（那条针对 SapBERT/Qwen3-8B 这种真吃卡的模型）。故本步不硬退：有 GPU 就用，
  没有就用 CPU，并打印用了哪个。整条子任务2门控链路（第3步建数据→第5步应用→打分）都能在
  无卡模式下跑完，只有第 4 步一次性训练需要整卡。

数据流（按 pmc_id join）：
    --input  题目版 jsonl（出 patient[]/full_text[]：做患者锚点、章节、上下文，与建数据集同源）
    --pred   SapBERT 预测 jsonl（出 entities[]：每个实体打一个 keep 概率）
    --obo    hp.obo（算 HPO 深度特征，与建数据集同源）
    --model-dir  第 4 步产物目录（gate_lr.pt + gate_preproc.json）
    --gold   带答案 jsonl（可选：传了就打子任务 2 分 / 扫阈值）
    --out    输出预测 jsonl（entities 原样 + 门控后重算的 association）

用法：
    # 扫阈值：在 dev 上对每个 keep 阈值跑一遍完整子任务 2，看 micF1/macF1 怎么变（需 --gold）
    #   阈值 0.0 那行 = 全留 = 复现「就近强挂」基线，作对照
    python scripts/s2_associate/gate_apply.py \
        --input data/split/dev_input.jsonl --pred pred_dev_sapbert.jsonl \
        --obo data/PatientPheX-V1-A/hp.obo --model-dir outputs/gate \
        --gold data/split/dev.jsonl \
        --scan-thresholds 0.0,0.3,0.4,0.5,0.6,0.7,0.8

    # 定好阈值后：用单一阈值生成正式预测文件（A 榜同样这样跑）
    python scripts/s2_associate/gate_apply.py \
        --input data/split/dev_input.jsonl --pred pred_dev_sapbert.jsonl \
        --obo data/PatientPheX-V1-A/hp.obo --model-dir outputs/gate \
        --threshold 0.5 --out pred_dev_gated.jsonl
"""
import argparse
import json
import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except AttributeError:
    pass

# ---- 复用三处已验证代码（import 约定见 scripts/README.md）----
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "core"))   # baseline_dict / evaluate
sys.path.insert(0, os.path.join(_HERE, "..", "prep"))   # build_gate_dataset（特征代码）
sys.path.insert(0, _HERE)                               # train_gate（Preprocessor / build_model）

from baseline_dict import (  # noqa: E402
    parse_obo, build_global, patient_anchors, associate,
)
from evaluate import load_jsonl, evaluate  # noqa: E402
import build_gate_dataset as bgd  # noqa: E402  铁律①：特征用建数据集的同一份代码
import train_gate as tg           # noqa: E402  铁律②：预处理器与模型结构复用训练侧


# ============================ 打分（子任务 2）============================

def score_docs(gold_docs, out_docs):
    """用 evaluate 打完整四指标，返回我们关心的 micF1/macF1/Score。
       门控只改 association，故 menF1/docF1 不变；真正会动的是 micF1/macF1。"""
    m = evaluate(gold_docs, out_docs, verbose=False)
    return m["f1_micro"], m["f1_macro"], m["score"]


# ============================ 主流程 ============================

def main():
    ap = argparse.ArgumentParser(description="门控应用：keep 过滤 + 就近路由 -> 子任务2预测")
    ap.add_argument("--input", required=True, help="题目版 jsonl（patient/full_text）")
    ap.add_argument("--pred", required=True, help="SapBERT 预测 jsonl（entities）")
    ap.add_argument("--obo", required=True, help="hp.obo 路径（算 HPO 深度）")
    ap.add_argument("--model-dir", default=os.path.join("outputs", "gate"),
                    help="第 4 步产物目录（gate_lr.pt + gate_preproc.json）")
    ap.add_argument("--gold", help="带答案 jsonl（传了就打子任务2分/扫阈值）")
    ap.add_argument("--out", help="输出预测 jsonl（单一阈值模式写出）")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="单一阈值模式：keep_prob>=该值才留（默认 0.5，最终值应按扫描结果定）")
    ap.add_argument("--scan-thresholds",
                    help="逗号分隔多个阈值；扫描模式，需 --gold。建议含 0.0（=就近强挂基线对照）")
    ap.add_argument("--context-window", type=int, default=120,
                    help="否定/正常线索窗口字符数（必须与建数据集时一致，默认 120）")
    ap.add_argument("--limit-docs", type=int, default=0, help="只处理前 N 篇（冒烟用）；0=全量")
    args = ap.parse_args()

    # ---- 读三份数据，按 pmc_id join ----
    inputs = load_jsonl(args.input)
    if args.limit_docs:
        inputs = inputs[:args.limit_docs]
    preds = {d["pmc_id"]: d for d in load_jsonl(args.pred)}
    print("输入 %d 篇；预测覆盖 %d 篇" % (len(inputs), len(preds)))

    # ---- 复原预处理器 + 加载模型（铁律②：不重算，复用保存的 json）----
    import torch
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("推理设备:", torch.cuda.get_device_name(0) if device.type == "cuda"
          else "CPU（单层 LR 前向，不吃 GPU，无卡模式即可）")

    with open(os.path.join(args.model_dir, "gate_preproc.json"), encoding="utf-8") as f:
        pre = tg.Preprocessor.from_dict(json.load(f))
    d = len(pre.feature_names)
    model = tg.build_model(d)
    state = torch.load(os.path.join(args.model_dir, "gate_lr.pt"), map_location=device)
    model.load_state_dict(state)
    model.to(device)
    print("已加载门控 LR：输入维度 d=%d（与训练一致）" % d)

    # ---- 深度表（特征需要，和建数据集同源）----
    print("解析 hp.obo 建深度表 ...")
    _, is_a = parse_obo(args.obo)
    depth_map = bgd.build_depth_map(is_a)
    print("  ROOT 子树深度表节点数: %d" % len(depth_map))

    # ---- 逐篇抽特征（铁律①：extract_features 同一份代码），一次性算所有 keep 概率 ----
    #   per_doc[i] = (doc, entities, n_ent)；feats 摊平成一个大列表，只做一次前向。
    per_doc = []
    all_feats = []
    for d0 in inputs:
        pmc = d0["pmc_id"]
        entities = (preds.get(pmc) or {}).get("entities", [])
        G = build_global(d0.get("full_text", []))
        ctx = {
            "global_text": G,
            "global_len": len(G),
            "sec_idx": bgd.section_index(d0.get("full_text", [])),
            "anchors": patient_anchors(d0),
            "patient_count": len(d0.get("patient", [])),
            "depth_map": depth_map,
            "window": args.context_window,
        }
        for e in entities:
            all_feats.append(bgd.extract_features(e, ctx))
        per_doc.append((d0, entities, len(entities)))

    n_ent = len(all_feats)
    print("待打分实体总数: %d" % n_ent)

    # transform 用训练侧保存的 mean/std/sections；y 无标签(默认0)，忽略即可
    X, _ = pre.transform(all_feats)
    X_t = torch.tensor(X, dtype=torch.float32)
    probs = tg.predict_proba(model, X_t, device) if n_ent else []

    # keep 概率摊回每篇（transform 保序：probs[k] 对应第 k 个实体）
    doc_probs = []
    cur = 0
    for (d0, entities, k) in per_doc:
        doc_probs.append(probs[cur:cur + k])
        cur += k

    # ---- 按阈值组装预测：keep 过滤 -> 原样 associate 就近路由 ----
    def assemble(threshold):
        """返回 (out_docs, n_kept)。entities 原样透传(子任务1不动)，association 用过滤后实体重算。"""
        out_docs = []
        n_kept = 0
        for (d0, entities, _k), pr in zip(per_doc, doc_probs):
            kept = [e for e, p in zip(entities, pr) if p >= threshold]
            n_kept += len(kept)
            assoc = associate(d0, kept)          # 就近路由，只喂幸存实体
            out_docs.append({
                "pmc_id": d0["pmc_id"], "pmid": d0.get("pmid"),
                "entities": entities,            # 原样：子任务1 menF1/docF1 不变
                "association": assoc,            # 门控后重算：子任务2
            })
        return out_docs, n_kept

    # ---- 扫描模式：每个阈值跑一遍完整子任务 2 ----
    if args.scan_thresholds:
        if not args.gold:
            ap.error("--scan-thresholds 需要 --gold 才能打分")
        gold_docs = load_jsonl(args.gold)
        ths = [float(x) for x in args.scan_thresholds.split(",") if x.strip()]

        print("\n===== keep 阈值扫描（接就近路由，打完整子任务 2）=====")
        print("  阈值   留存实体   micF1    macF1    Score    Δmac(vs全留)")
        print("  " + "-" * 58)
        base_mac = None          # 阈值最低那行(通常 0.0=全留=就近强挂基线)作对照基准
        best = None              # 记录 Score 最优的阈值
        for th in sorted(ths):
            out_docs, n_kept = assemble(th)
            mic, mac, sc = score_docs(gold_docs, out_docs)
            if base_mac is None:
                base_mac = mac
            dmac = mac - base_mac
            print("  %.2f   %6d    %.4f   %.4f   %.4f   %+.4f"
                  % (th, n_kept, mic, mac, sc, dmac))
            # 按 Score 挑：竞赛 micF1 与 macF1 等权，Score=0.25*(...)+0.25*(micF1+macF1)，
            # 故最大化 Score 就是最大化 micF1+macF1 之和，而非任一单指标。
            if best is None or sc > best[1]:
                best = (th, sc, mic, mac, n_kept)
        print("  ---- 对照：就近强挂基线 dev micF1≈0.4310 / macF1≈0.4030 ----")
        print("  >>> Score 最优阈值 = %.2f：Score=%.4f  micF1=%.4f  macF1=%.4f（留 %d 实体）<<<"
              % (best[0], best[1], best[2], best[3], best[4]))
        print("确定阈值后，用 --threshold %.2f --out <文件> 生成正式预测。" % best[0])
        return

    # ---- 单一阈值模式：生成预测文件 ----
    if not args.out:
        ap.error("单一阈值模式需要 --out 指定输出文件（或用 --scan-thresholds 扫描）")
    out_docs, n_kept = assemble(args.threshold)
    with open(args.out, "w", encoding="utf-8") as f:
        for od in out_docs:
            f.write(json.dumps(od, ensure_ascii=False) + "\n")
    print("\n已写出: %s" % args.out)
    print("  阈值 %.2f 下：%d 个实体中留 %d 个喂就近路由（弃 %d 个疑似噪声）"
          % (args.threshold, n_ent, n_kept, n_ent - n_kept))
    if args.gold:
        gold_docs = load_jsonl(args.gold)
        mic, mac, sc = score_docs(gold_docs, out_docs)
        print("  子任务2：micF1=%.4f  macF1=%.4f  Score=%.4f" % (mic, mac, sc))
    print("接下来可用 evaluate.py 记录实验：")
    print("  python scripts/core/evaluate.py --gold %s --pred %s \\"
          % (args.gold or "<dev.jsonl>", args.out))
    print("      --tag gate_v1 --note \"门控砍过度关联 阈值%.2f\"" % args.threshold)


if __name__ == "__main__":
    main()
