#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
assoc_gate.py —— 门控式归属：LLM 只做 keep/drop，路由仍交给就近（纯 Python，不碰 GPU）。

诊断结论（见 diag_assoc.py 输出）：Qwen3-8B 全权归属在 dev 上
    micF1 0.4310→0.4514（+0.0204，好）  但  macF1 0.4030→0.3369（-0.0661，崩）。
拆开看，LLM 其实同时干了两件相反难度的事：
    · 门控(gate)：每个表型「留下 / 判 NONE 丢弃」——二分类，砍掉 ~42% 疾病背景 FP，
      这是 micro 上涨的来源，LLM 干得好。
    · 路由(route)：留下的表型「挂给哪个患者」——多患者(占 81%)时的 3+ 路选择，
      LLM 把小患者/非主患者的表型误挂给 proband，16 个小患者被打成 0 分 → macro 崩。
macro 逐 gold 患者等权，对小患者被清零极敏感，所以路由的错主导了净失分。

本脚本把两件事解耦：**保留 LLM 的门控，把路由还给就近**。
    gate-only 关联 = 就近关联(pred_base) ∩ LLM 保留集(pred_llm 里被挂给任何人的 token)
就近按 offset 就近分配，绝不会把 A 患者的表型抢给 proband，所以不会制造 macro 崩；
同时 LLM 判 NONE 丢掉的背景表型依旧被砍掉，micro 的收益应当保住。

这是纯离线组合，输入是两份**已存在**的预测文件，**不需要 GPU、无卡模式即可跑**。
若 gate-only 在 dev 上 micF1 与 macF1 双双 ≥ 基线，就证明「解耦」路线成立，
再把它做进 associate_llm.py 的正式 --gate-only 模式并在 A 榜验证。

数据流（按 pmc_id join）：
    --base  pred_dev_sapbert.jsonl   就近基线预测（出 entities 与 就近 association）
    --llm   pred_dev_llm.jsonl        LLM 全权归属预测（出 LLM 保留集）
    --out   pred_dev_gate.jsonl       门控式预测（entities 原样，association 为交集）
    [--gold data/split/dev.jsonl]     传了就顺便打分

口径与 evaluate.py 一致：token 级用 split_ids 拆复合/去重，输出单 token 列表（evaluate 会再拆，等价）。

用法（无卡模式即可，项目根目录）：
    python scripts/assoc_gate.py \
        --base pred_dev_sapbert.jsonl \
        --llm  pred_dev_llm.jsonl \
        --out  pred_dev_gate.jsonl \
        --gold data/split/dev.jsonl
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate import load_jsonl, split_ids, evaluate  # noqa: E402


def kept_tokens_of(doc):
    """LLM 预测里「被挂给任何患者」的表型 token 集合（=没被判 NONE 的）。"""
    kept = set()
    for a in doc.get("association", []):
        for ph in a.get("phenotype", []):
            for t in split_ids(ph):
                kept.add(t)
    return kept


def base_assoc_tokens(doc):
    """就近预测里每个患者的表型 token 集合：{patient_id: set(token)}。"""
    out = {}
    for a in doc.get("association", []):
        s = out.setdefault(a["patient_id"], set())
        for ph in a.get("phenotype", []):
            for t in split_ids(ph):
                s.add(t)
    return out


def main():
    ap = argparse.ArgumentParser(description="门控式归属：LLM 门控 × 就近路由（离线组合，无 GPU）")
    ap.add_argument("--base", required=True, help="就近基线预测 jsonl（出 entities 与就近 association）")
    ap.add_argument("--llm", required=True, help="LLM 全权归属预测 jsonl（出 LLM 保留集）")
    ap.add_argument("--out", required=True, help="输出门控式预测 jsonl")
    ap.add_argument("--gold", help="金标准 jsonl；传了就顺便打分")
    args = ap.parse_args()

    base_docs = load_jsonl(args.base)
    llm_docs = load_jsonl(args.llm)
    llm_by_pmc = {d["pmc_id"]: d for d in llm_docs}

    out_docs = []
    n_join = n_nollm = 0
    n_tok_base = n_tok_gate = 0     # 就近总 token 数 / 门控后保留 token 数（看砍了多少）
    for d in base_docs:
        pmc = d["pmc_id"]
        base_tok = base_assoc_tokens(d)
        n_tok_base += sum(len(s) for s in base_tok.values())

        llm_doc = llm_by_pmc.get(pmc)
        if llm_doc is None:
            # LLM 没覆盖这篇 → 保守保留就近原样（绝不比基线差）
            kept = None
            n_nollm += 1
        else:
            kept = kept_tokens_of(llm_doc)
            n_join += 1

        assoc = []
        for pid, toks in base_tok.items():
            new = toks if kept is None else (toks & kept)   # 就近路由 ∩ LLM 门控
            n_tok_gate += len(new)
            assoc.append({"patient_id": pid, "phenotype": sorted(new)})

        out_docs.append({
            "pmc_id": pmc, "pmid": d.get("pmid"),
            "entities": d.get("entities", []),      # 识别层原样，子任务1 指标不变
            "association": assoc,
        })

    with open(args.out, "w", encoding="utf-8") as f:
        for d in out_docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    print("已写出: %s" % args.out)
    print("  join 到 LLM 的篇数 %d；LLM 未覆盖(保留就近) %d 篇" % (n_join, n_nollm))
    dropped = n_tok_base - n_tok_gate
    if n_tok_base:
        print("  就近关联 token %d → 门控保留 %d（LLM 判 NONE 砍掉 %d，占 %.1f%%）"
              % (n_tok_base, n_tok_gate, dropped, 100.0 * dropped / n_tok_base))

    if args.gold:
        gold_docs = load_jsonl(args.gold)
        print("\n===== 门控式打分（对照 sapbert_v1: micF1=0.4310 macF1=0.4030；"
              "LLM全权: micF1=0.4514 macF1=0.3369）=====")
        evaluate(gold_docs, out_docs, verbose=True)


if __name__ == "__main__":
    main()
