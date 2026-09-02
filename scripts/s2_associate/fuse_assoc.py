#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
fuse_assoc.py —— 融合两个「子任务2 门控」的产物，看 ∩/∪ 能否超过单独任一个。

背景（为什么做融合）：
  我们有两个**正交**的 keep/drop 判断器，各自把 SapBERT 过度召回的表型砍一部分：
    * LR 门控 (gate_apply.py)     ：看 24 维统计特征（分数/章节/深度/距锚点…）
    * LLM 门控 (associate_llm --gate-only)：看 Qwen3-8B 的语义判断（属于谁/NONE）
  两者依据不同、错误也不同。dev 上各自：LR Score 0.5719、LLM Score 0.5783。
  融合两个独立判断，很可能比单独任一个强 —— 这就是本脚本要验证的。

融合在哪一层做（关键）：
  两份预测的 entities 完全相同（都原样透传 SapBERT 识别），就近路由也是同一套
  associate()，差异**只**来自各自的 keep 决策。而 evaluate 子任务2 是把每个患者的
  phenotype 用 split_ids 拆成 token 集合再比对。所以最干净、且与评估口径 100% 对齐的
  融合层 = 在 (pmc_id, patient_id) -> {phenotype token} 集合上做布尔运算：
    * intersect（∩，两个门控都留才留）：更保守，精准砍 FP，冲 precision/micF1
    * union    （∪，任一门控留就留）：更宽松，补召回，冲 recall
  哪个好不先验判断，dev 上各打一次分，直接看谁超过 0.5783。

它不做什么：
  * 不碰识别（entities 原样，两份必须一致，脚本会校验）。只重算 association。
  * 不训练、不吃 GPU。纯标准库 + evaluate。无卡模式即可跑。

数据流（按 pmc_id join 两份预测）：
    --pred-a  门控A 预测 jsonl（如 pred_dev_gated.jsonl，LR 门控）
    --pred-b  门控B 预测 jsonl（如 pred_dev_gate.jsonl，LLM 门控）
    --gold    带答案 jsonl（可选：传了就把 A / B / 融合 一起打分对比）
    --out     融合预测 jsonl（可选：--mode 非 both 时写出）
    --mode    intersect | union | both（默认 both：两种都算并对比）

用法：
    # dev 上对比 LR / LLM / ∩ / ∪ 四者（需 --gold）
    python scripts/s2_associate/fuse_assoc.py \
        --pred-a pred_dev_gated.jsonl --pred-b pred_dev_gate.jsonl \
        --gold data/split/dev.jsonl --mode both

    # 选定融合方式后生成正式预测（A 榜同样这样跑，用 A 榜两份门控产物）
    python scripts/s2_associate/fuse_assoc.py \
        --pred-a pred_A_gated.jsonl --pred-b pred_A_gate.jsonl \
        --mode intersect --out pred_A_fused.jsonl
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

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "core"))
from evaluate import load_jsonl, evaluate, split_ids  # noqa: E402


# ============================ 融合核心 ============================

def assoc_token_map(doc):
    """把一篇的 association 变成 {patient_id: set(phenotype token)}（split_ids 拆复合、去重）。
       对齐 evaluate.patient_pheno_map 的 token 口径，融合在这一层做才与打分一致。"""
    out = {}
    for a in doc.get("association", []):
        s = out.setdefault(a["patient_id"], set())
        for ph in a.get("phenotype", []):
            for tok in split_ids(ph):
                s.add(tok)
    return out


def fuse_doc(doc_a, doc_b, mode):
    """融合单篇的 association。entities 取 A（两份应相同）。返回融合后的 association 列表。
       患者集合取两份并集（缺失当空集），逐患者对 token 集合做 ∩ 或 ∪。"""
    ma = assoc_token_map(doc_a)
    mb = assoc_token_map(doc_b)
    pids = set(ma) | set(mb)
    assoc = []
    for pid in pids:
        sa = ma.get(pid, set())
        sb = mb.get(pid, set())
        toks = (sa & sb) if mode == "intersect" else (sa | sb)
        assoc.append({"patient_id": pid, "phenotype": sorted(toks)})
    return assoc


def build_fused(docs_a, b_by_pmc, mode):
    """按 pmc_id join，产出融合预测 docs（entities 原样透传 A）。"""
    out_docs = []
    for da in docs_a:
        pmc = da["pmc_id"]
        db = b_by_pmc.get(pmc, {"association": []})
        out_docs.append({
            "pmc_id": pmc, "pmid": da.get("pmid"),
            "entities": da.get("entities", []),      # 原样：子任务1 不动
            "association": fuse_doc(da, db, mode),    # 融合后的子任务2
        })
    return out_docs


# ============================ 一致性校验 ============================

def check_entities_aligned(docs_a, b_by_pmc):
    """两份预测的 entities 必须一致（融合前提）。不一致时告警，返回是否对齐。
       只比 (offset,length,identifier) 集合，忽略字段顺序/附加字段。"""
    def ent_key(doc):
        return {(e.get("offset"), e.get("length"), e.get("identifier"))
                for e in doc.get("entities", [])}
    n_mismatch = 0
    for da in docs_a:
        db = b_by_pmc.get(da["pmc_id"])
        if db is None:
            n_mismatch += 1
            continue
        if ent_key(da) != ent_key(db):
            n_mismatch += 1
    return n_mismatch


# ============================ 主流程 ============================

def score_line(name, gold_docs, out_docs):
    m = evaluate(gold_docs, out_docs, verbose=False)
    print("  %-14s micF1=%.4f  macF1=%.4f  Score=%.4f"
          % (name, m["f1_micro"], m["f1_macro"], m["score"]))
    return m


def main():
    ap = argparse.ArgumentParser(description="融合两个门控产物（∩/∪），dev 上对比择优")
    ap.add_argument("--pred-a", required=True, help="门控A 预测 jsonl（如 LR 门控 pred_dev_gated.jsonl）")
    ap.add_argument("--pred-b", required=True, help="门控B 预测 jsonl（如 LLM 门控 pred_dev_gate.jsonl）")
    ap.add_argument("--gold", help="带答案 jsonl（传了就打分对比 A/B/融合）")
    ap.add_argument("--out", help="融合预测 jsonl（--mode 非 both 时写出）")
    ap.add_argument("--mode", choices=["intersect", "union", "both"], default="both",
                    help="intersect=∩(都留才留)；union=∪(任一留就留)；both=两种都算并对比")
    args = ap.parse_args()

    docs_a = load_jsonl(args.pred_a)
    docs_b = load_jsonl(args.pred_b)
    b_by_pmc = {d["pmc_id"]: d for d in docs_b}
    print("门控A %d 篇（%s）" % (len(docs_a), args.pred_a))
    print("门控B %d 篇（%s）" % (len(docs_b), args.pred_b))

    # 一致性校验：entities 必须一致，否则融合无意义
    n_mismatch = check_entities_aligned(docs_a, b_by_pmc)
    if n_mismatch:
        print("⚠ %d 篇的 entities 在两份预测里不一致（或缺篇）。融合仍按 A 的 entities 进行，"
              "但请先确认两份是同一 SapBERT 识别的产物。" % n_mismatch)
    else:
        print("✓ 两份 entities 完全一致（同一识别产物，融合前提满足）")

    modes = ["intersect", "union"] if args.mode == "both" else [args.mode]

    gold_docs = load_jsonl(args.gold) if args.gold else None
    if gold_docs is not None:
        print("\n===== dev 打分对比（子任务2）=====")
        score_line("门控A(单独)", gold_docs, docs_a)
        score_line("门控B(单独)", gold_docs, docs_b)
        print("  " + "-" * 52)

    fused_by_mode = {}
    for mode in modes:
        fused = build_fused(docs_a, b_by_pmc, mode)
        fused_by_mode[mode] = fused
        if gold_docs is not None:
            tag = "融合 ∩(%s)" % mode if mode == "intersect" else "融合 ∪(%s)" % mode
            score_line(tag, gold_docs, fused)

    if gold_docs is not None:
        print("  " + "-" * 52)
        print("  对照：LLM 单独 dev Score≈0.5783 / A 榜线上 0.6006（当前最佳）")
        print("  → 若某融合 Score 明显超 0.5783，值得上 A 榜；否则维持 LLM 单独。")

    # 写出（单一 mode 时）
    if args.out:
        if len(modes) != 1:
            ap.error("--out 需要指定单一 --mode（intersect 或 union），both 模式不写出")
        with open(args.out, "w", encoding="utf-8") as f:
            for od in fused_by_mode[modes[0]]:
                f.write(json.dumps(od, ensure_ascii=False) + "\n")
        print("\n已写出融合预测（%s）: %s" % (modes[0], args.out))


if __name__ == "__main__":
    main()
