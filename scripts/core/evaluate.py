#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
evaluate.py —— 本地复现 CHIP2026 评测一的四个 F1 与总分 Score。

严格对齐 docs/03-评测指标.md：
  子任务1  F1_men  (mention 级：offset+length+HPO_ID 全严格相等)
           F1_doc  (document 级：文档内去重 HPO 集合)
  子任务2  F1_micro(全局累加 患者-表型 对)
           F1_macro(按患者等权平均；真实与预测均空 → P=R=F1=1)
  Score = 0.25*(F1_men + F1_doc) + 0.25*(F1_micro + F1_macro)

特殊实体处理（对 gold 与 pred 施以相同规范化，保证内部一致、可比）：
  - 否定 note=="NO"：不作为正例。gold 的 NO 不计 FN；pred 的 NO 从评价中剔除；
    pred 把否定预测成普通表型 → 自然计 FP（因其不在 gold 正例集中）。
  - 复合 identifier 含 ';'：按分号拆成独立评价单元（mention/doc/子任务2 均拆）。
  - 无 ID '-1'：位置正确即算对（id 用字面量 '-1' 参与匹配）。
  - note 的其它取值（如 'D'）按「非 NO」即正例处理。

用法：
    python scripts/core/evaluate.py --gold data/split/dev.jsonl --pred submit.jsonl
    python scripts/core/evaluate.py --selftest        # 用 docs/07 §6 手算例校验本脚本

注意：本脚本为「自研本地评测」，用于开发期相对比较；与官方线上评测可能存在
     未公开的边界差异（尤其 -1 在 doc 级、复合在子任务2 是否拆分）。以官方为准。
"""
import argparse
import json
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except AttributeError:
    pass


# ----------------------------- 基础工具 -----------------------------

def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def split_ids(identifier):
    """按分号拆分复合 ID；保留 '-1' 与原文文本；去空白。"""
    if identifier is None:
        return []
    return [t.strip() for t in str(identifier).split(";") if t.strip()]


def is_negated(entity):
    return entity.get("note") == "NO"


def prf(tp, fp, fn):
    """由 TP/FP/FN 计算 P/R/F1；分母为 0 时该项取 0。"""
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f1


def index_by_pmc(docs):
    return {d["pmc_id"]: d for d in docs}


# --------------------------- 子任务 1：mention 级 ---------------------------

def mention_units(doc):
    """返回该文档正例 mention 单元集合：{(offset,length,id)}，剔除 NO，复合拆分。"""
    units = set()
    for e in doc.get("entities", []):
        if is_negated(e):
            continue
        for hid in split_ids(e.get("identifier")):
            units.add((e.get("offset"), e.get("length"), hid))
    return units


def eval_mention(gold_docs, pred_docs):
    gold = index_by_pmc(gold_docs)
    pred = index_by_pmc(pred_docs)
    tp = fp = fn = 0
    for pmc, g in gold.items():
        gset = mention_units(g)
        pset = mention_units(pred.get(pmc, {"entities": []}))
        tp += len(gset & pset)
        fp += len(pset - gset)
        fn += len(gset - pset)
    return prf(tp, fp, fn), (tp, fp, fn)


# --------------------------- 子任务 1：document 级 ---------------------------

def doc_id_set(doc):
    """该文档去重 HPO 集合（剔除 NO，复合拆分；'-1' 作为字面量计入）。"""
    ids = set()
    for e in doc.get("entities", []):
        if is_negated(e):
            continue
        for hid in split_ids(e.get("identifier")):
            ids.add(hid)
    return ids


def eval_document(gold_docs, pred_docs):
    gold = index_by_pmc(gold_docs)
    pred = index_by_pmc(pred_docs)
    tp = fp = fn = 0
    for pmc, g in gold.items():
        gd = doc_id_set(g)
        pd = doc_id_set(pred.get(pmc, {"entities": []}))
        tp += len(gd & pd)
        fp += len(pd - gd)
        fn += len(gd - pd)
    return prf(tp, fp, fn), (tp, fp, fn)


# --------------------------- 子任务 2：患者-表型 ---------------------------

def patient_pheno_map(doc):
    """{(pmc_id, patient_id): set(phenotype tokens)}；复合拆分、去重，保留原文/-1 文本。"""
    out = {}
    pmc = doc["pmc_id"]
    for a in doc.get("association", []):
        key = (pmc, a["patient_id"])
        s = out.setdefault(key, set())
        for ph in a.get("phenotype", []):
            for tok in split_ids(ph):
                s.add(tok)
    return out


def eval_subtask2(gold_docs, pred_docs):
    gmap = {}
    for d in gold_docs:
        gmap.update(patient_pheno_map(d))
    pmap = {}
    for d in pred_docs:
        pmap.update(patient_pheno_map(d))

    # micro：over 所有 gold 与 pred 键的并集，多出的 pred 患者其表型计 FP
    micro_tp = micro_fp = micro_fn = 0
    all_keys = set(gmap) | set(pmap)
    for k in all_keys:
        y = gmap.get(k, set())
        yh = pmap.get(k, set())
        micro_tp += len(y & yh)
        micro_fp += len(yh - y)
        micro_fn += len(y - yh)
    micro = prf(micro_tp, micro_fp, micro_fn)

    # macro：K = gold 患者总数，逐患者算 F1 再等权平均；双空 → 1
    f1s = []
    ps = []
    rs = []
    for k in gmap:
        y = gmap[k]
        yh = pmap.get(k, set())
        if not y and not yh:
            ps.append(1.0); rs.append(1.0); f1s.append(1.0)
            continue
        tp = len(y & yh)
        p = tp / len(yh) if yh else 0.0
        r = tp / len(y) if y else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        ps.append(p); rs.append(r); f1s.append(f1)
    K = len(gmap)
    macro = (
        sum(ps) / K if K else 0.0,
        sum(rs) / K if K else 0.0,
        sum(f1s) / K if K else 0.0,
    )
    return micro, macro, (micro_tp, micro_fp, micro_fn), K


# ------------------------------- 汇总 -------------------------------

def evaluate(gold_docs, pred_docs, verbose=True):
    (pm, rm, f1_men), men_c = eval_mention(gold_docs, pred_docs)
    (pd, rd, f1_doc), doc_c = eval_document(gold_docs, pred_docs)
    (mi_p, mi_r, f1_micro), (ma_p, ma_r, f1_macro), mi_c, K = eval_subtask2(gold_docs, pred_docs)

    score = 0.25 * (f1_men + f1_doc) + 0.25 * (f1_micro + f1_macro)

    if verbose:
        print("========== 子任务 1：全文表型概念识别 ==========")
        print("  [mention 级]  P=%.4f  R=%.4f  F1=%.4f   (TP=%d FP=%d FN=%d)"
              % (pm, rm, f1_men, *men_c))
        print("  [document 级] P=%.4f  R=%.4f  F1=%.4f   (TP=%d FP=%d FN=%d)"
              % (pd, rd, f1_doc, *doc_c))
        print("========== 子任务 2：特定患者表型概念识别 ==========")
        print("  [micro]       P=%.4f  R=%.4f  F1=%.4f   (TP=%d FP=%d FN=%d)"
              % (mi_p, mi_r, f1_micro, *mi_c))
        print("  [macro]       P=%.4f  R=%.4f  F1=%.4f   (患者数 K=%d)"
              % (ma_p, ma_r, f1_macro, K))
        print("==================================================")
        print("  F1_men=%.4f  F1_doc=%.4f  F1_micro=%.4f  F1_macro=%.4f"
              % (f1_men, f1_doc, f1_micro, f1_macro))
        print("  >>> Score = 0.25*(F1_men+F1_doc) + 0.25*(F1_micro+F1_macro) = %.4f <<<"
              % score)

    return {
        "f1_men": f1_men, "f1_doc": f1_doc,
        "f1_micro": f1_micro, "f1_macro": f1_macro,
        "score": score,
    }


# ------------------------------- 自检 -------------------------------

def selftest():
    """用 docs/07 §6 的手算例校验：期望 F1_men=F1_doc≈0.667, micro=0.75, macro≈0.733, Score≈0.704。"""
    B = 5  # 任意 length
    gold = [{
        "pmc_id": "TEST", "pmid": "0",
        "patient": [{"patient_id": "P1"}, {"patient_id": "P2"}],
        "full_text": [],
        "entities": [
            {"identifier": "HP:1", "offset": 10, "length": B, "note": None},
            {"identifier": "HP:2", "offset": 20, "length": B, "note": "NO"},
            {"identifier": "HP:3;HP:4", "offset": 30, "length": B, "note": None},
        ],
        "association": [
            {"patient_id": "P1", "phenotype": ["HP:1", "HP:3", "HP:4"]},
            {"patient_id": "P2", "phenotype": ["HP:5"]},
        ],
    }]
    pred = [{
        "pmc_id": "TEST", "pmid": "0",
        "entities": [
            {"identifier": "HP:1", "offset": 10, "length": B, "note": None},
            {"identifier": "HP:2", "offset": 20, "length": B, "note": None},  # 否定被当普通 → FP
            {"identifier": "HP:3", "offset": 30, "length": B, "note": None},  # 复合漏 HP:4
        ],
        "association": [
            {"patient_id": "P1", "phenotype": ["HP:1", "HP:3"]},
            {"patient_id": "P2", "phenotype": ["HP:5", "HP:6"]},
        ],
    }]

    r = evaluate(gold, pred, verbose=True)
    exp = {"f1_men": 2 / 3, "f1_doc": 2 / 3, "f1_micro": 0.75,
           "f1_macro": (0.8 + 2 / 3) / 2}
    exp["score"] = 0.25 * (exp["f1_men"] + exp["f1_doc"]) + \
                   0.25 * (exp["f1_micro"] + exp["f1_macro"])
    print("\n--- 自检对比（期望 vs 实得）---")
    ok = True
    for k in ("f1_men", "f1_doc", "f1_micro", "f1_macro", "score"):
        diff = abs(r[k] - exp[k])
        flag = "OK " if diff < 1e-6 else "!!!"
        if diff >= 1e-6:
            ok = False
        print("  %-9s 期望=%.6f  实得=%.6f  %s" % (k, exp[k], r[k], flag))
    print("自检结果:", "全部通过 ✅" if ok else "存在偏差 ❌")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description="本地复现四个 F1 与 Score")
    ap.add_argument("--gold", help="金标准 jsonl（含答案，如 dev.jsonl）")
    ap.add_argument("--pred", help="预测 jsonl（submit 文件）")
    ap.add_argument("--selftest", action="store_true", help="运行手算例自检")
    ap.add_argument("--tag", help="实验名（主键），提供则把本地分数记入 results/experiments.jsonl")
    ap.add_argument("--note", help="一句话说明这次实验改了什么")
    ap.add_argument("--ts", help="记录时间字符串（如 2026-08-28 17:00）；不传则自动取当前本地时间")
    args = ap.parse_args()

    if args.selftest:
        sys.exit(selftest())

    if not (args.gold and args.pred):
        ap.error("需要同时提供 --gold 与 --pred（或用 --selftest）")

    gold_docs = load_jsonl(args.gold)
    pred_docs = load_jsonl(args.pred)

    gset = {d["pmc_id"] for d in gold_docs}
    pset = {d["pmc_id"] for d in pred_docs}
    if gset != pset:
        miss = gset - pset
        extra = pset - gset
        print("⚠️ pmc_id 集合不一致：漏 %d 篇，多 %d 篇" % (len(miss), len(extra)))
        if miss:
            print("   漏预测:", sorted(miss)[:10], "..." if len(miss) > 10 else "")
        if extra:
            print("   多预测:", sorted(extra)[:10], "..." if len(extra) > 10 else "")
        if not (gset & pset):
            print("⚠️ pmc_id 完全无交集——很可能 --gold 与 --pred 用错了文件"
                  "（如拿 dev 答案给 A 榜预测打分）。分数无意义，已中止记录。")
            return

    metrics = evaluate(gold_docs, pred_docs, verbose=True)

    if args.tag:
        try:
            import explog
        except ImportError:
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            import explog
        rec = {
            "ts_local": args.ts or explog.now_str(),
            "tag": args.tag,
            "note": args.note,
            "split": args.gold,
            "pred": args.pred,
            "local": {
                "f1_men": round(metrics["f1_men"], 4),
                "f1_doc": round(metrics["f1_doc"], 4),
                "f1_micro": round(metrics["f1_micro"], 4),
                "f1_macro": round(metrics["f1_macro"], 4),
                "score": round(metrics["score"], 4),
            },
        }
        explog.upsert(rec)
        print("\n📝 已记录到 %s（tag=%s）" % (explog.STORE, args.tag))


if __name__ == "__main__":
    main()
