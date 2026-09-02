#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
diag_assoc.py —— 子任务2「患者-表型归属」的失分归因诊断（纯 Python，不碰 GPU）。

背景：associate_llm.py（Qwen3-8B 归属）在 dev 上 micF1 涨了（0.4310→0.4514）但
      macF1 崩了（0.4030→0.3369），Score 净降 0.0114。macro 逐 gold 患者等权，
      dev 里 81% 是多患者篇、34% 患者仅 ≤2 表型——极易被「判 NONE 过头」或
      「多患者挂错人」打成 0 分。这两种病改法相反，必须先定位再动手。

它做什么：三方逐患者对比
    --gold    金标准 jsonl（出 association，作真值 y）
    --base    就近基线预测 jsonl（出 association，如 pred_dev_sapbert.jsonl）→ 集合 g
    --llm     LLM 归属预测 jsonl（associate_llm.py 的 --out）              → 集合 l
  对每个 gold 患者 (pmc_id, patient_id) 算 per-patient F1，把 base 与 llm 摆一起，
  归因 llm 相对 gold 的失分：miss=y-l（漏挂，伤 R）、extra=l-y（多挂，伤 P）；
  再看 llm 相对 base 的每患者 delta，按患者表型数分桶，并列出 delta 最负的样本明细。

口径与 evaluate.py 完全一致（patient_pheno_map / split_ids / prf），所以脚本复现的
macro/micro 应与 evaluate 打分逐位吻合——先看开头「自洽校验」两行确认。

用法（无卡模式即可，项目根目录）：
    python scripts/diag_assoc.py \
        --gold data/split/dev.jsonl \
        --base pred_dev_sapbert.jsonl \
        --llm  pred_dev_llm.jsonl
"""
import argparse
import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except AttributeError:
    pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate import load_jsonl, patient_pheno_map, split_ids, prf  # noqa: E402


def f1_of(y, yh):
    """单患者 F1，口径同 evaluate.eval_subtask2 的 macro 分支（双空→1）。"""
    if not y and not yh:
        return 1.0, 1.0, 1.0
    tp = len(y & yh)
    p = tp / len(yh) if yh else 0.0
    r = tp / len(y) if y else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f1


def macro_micro(gmap, pmap):
    """复现 evaluate 的 macro（逐 gold 患者）与 micro（gold∪pred 键并集）。"""
    # macro
    f1s = []
    for k in gmap:
        _, _, f1 = f1_of(gmap[k], pmap.get(k, set()))
        f1s.append(f1)
    macro = sum(f1s) / len(gmap) if gmap else 0.0
    # micro
    tp = fp = fn = 0
    for k in set(gmap) | set(pmap):
        y = gmap.get(k, set())
        yh = pmap.get(k, set())
        tp += len(y & yh)
        fp += len(yh - y)
        fn += len(y - yh)
    _, _, micro = prf(tp, fp, fn)
    return macro, micro


def collect(docs):
    """{(pmc,pid): set(tokens)}，合并全体文档。"""
    m = {}
    for d in docs:
        m.update(patient_pheno_map(d))
    return m


def bucket(n):
    if n <= 2:
        return "≤2 "
    if n <= 5:
        return "3-5 "
    return ">5  "


def miss_destination(gmap, bmap, lmap):
    """量化每个 gold 患者的 miss 表型(y-l)去哪了，区分三种病因（改法相反）：
       to_other = 该表型被 LLM 判给了**同文档的别的患者** → 挂错人(主患者偏向)，收结构。
       to_none  = 该表型没进任何 LLM 患者，但 base 把它挂对了该患者 → LLM 判了 NONE(弃权过头)，放宽措辞。
       to_lost  = base 也没挂到该患者 → 多半识别层就没有，非归属锅，两法都救不了。"""
    from collections import defaultdict
    llm_by_pmc = defaultdict(dict)
    for (pmc, pid), s in lmap.items():
        llm_by_pmc[pmc][pid] = s
    to_other = to_none = to_lost = 0
    for (pmc, pid), y in gmap.items():
        l = lmap.get((pmc, pid), set())
        b = bmap.get((pmc, pid), set())
        others = set()
        for pid2, s2 in llm_by_pmc[pmc].items():
            if pid2 != pid:
                others |= s2
        for t in (y - l):
            if t in others:
                to_other += 1
            elif t in b:
                to_none += 1
            else:
                to_lost += 1
    return to_other, to_none, to_lost


def main():
    ap = argparse.ArgumentParser(description="子任务2 归属失分归因诊断")
    ap.add_argument("--gold", required=True, help="金标准 jsonl")
    ap.add_argument("--base", required=True, help="就近基线预测 jsonl（如 pred_dev_sapbert.jsonl）")
    ap.add_argument("--llm", required=True, help="LLM 归属预测 jsonl（associate_llm 的 --out）")
    ap.add_argument("--top", type=int, default=12, help="列出 delta 最负的前 N 个患者明细")
    args = ap.parse_args()

    gmap = collect(load_jsonl(args.gold))
    bmap = collect(load_jsonl(args.base))
    lmap = collect(load_jsonl(args.llm))

    # ---------- 自洽校验：复现 evaluate 的 macro/micro ----------
    b_mac, b_mic = macro_micro(gmap, bmap)
    l_mac, l_mic = macro_micro(gmap, lmap)
    print("自洽校验（应与 evaluate 打分吻合）：")
    print("  就近 base : micF1=%.4f  macF1=%.4f" % (b_mic, b_mac))
    print("  LLM       : micF1=%.4f  macF1=%.4f" % (l_mic, l_mac))
    print("  变化      : micF1 %+.4f   macF1 %+.4f" % (l_mic - b_mic, l_mac - b_mac))

    # ---------- 逐 gold 患者：base vs llm ----------
    rows = []
    for k in gmap:
        y = gmap[k]
        _, _, fb = f1_of(y, bmap.get(k, set()))
        _, lr, fl = f1_of(y, lmap.get(k, set()))
        rows.append((k, len(y), fb, fl, fl - fb))

    better = sum(1 for r in rows if r[4] > 1e-9)
    worse = sum(1 for r in rows if r[4] < -1e-9)
    same = len(rows) - better - worse
    print("\n逐 gold 患者 base→LLM（K=%d）：变好 %d，变差 %d，持平 %d"
          % (len(rows), better, worse, same))

    # ---------- 按患者表型数分桶看 delta ----------
    print("\n按 gold 表型数分桶（macro 等权单元）：")
    print("  桶     患者数   base_macF1   LLM_macF1   ΔmacF1   （该桶对总 macro 的拖累）")
    order = ["≤2 ", "3-5 ", ">5  "]
    agg = {b: [] for b in order}
    for (k, n, fb, fl, dl) in rows:
        agg[bucket(n)].append((fb, fl))
    K = len(rows)
    for b in order:
        lst = agg[b]
        if not lst:
            continue
        nb = len(lst)
        mb = sum(x[0] for x in lst) / nb
        ml = sum(x[1] for x in lst) / nb
        # 该桶对总 macro 的贡献差 = Σ(fl-fb)/K
        contrib = sum(x[1] - x[0] for x in lst) / K
        print("  %s   %4d     %.4f       %.4f     %+.4f      %+.4f"
              % (b, nb, mb, ml, ml - mb, contrib))

    # ---------- miss / extra 归因（LLM 相对 gold）----------
    tot_miss = tot_extra = 0
    zeroed = 0          # base>0 但 llm=0 的患者（被 LLM 打成零分）
    rescued = 0         # base=0 但 llm>0 的患者（被 LLM 救活）
    for (k, n, fb, fl, dl) in rows:
        y = gmap[k]
        l = lmap.get(k, set())
        tot_miss += len(y - l)
        tot_extra += len(l - y)
        if fb > 1e-9 and fl <= 1e-9:
            zeroed += 1
        if fb <= 1e-9 and fl > 1e-9:
            rescued += 1
    print("\nLLM 相对 gold 的 token 级归因：")
    print("  漏挂 miss(伤R)=%d   多挂 extra(伤P)=%d" % (tot_miss, tot_extra))
    print("  被 LLM 打成 0 分的患者(base>0→llm=0)=%d   被救活的患者(base=0→llm>0)=%d"
          % (zeroed, rescued))

    # ---------- miss 去向拆解：挂错人 vs 判NONE vs 识别本就缺 ----------
    to_other, to_none, to_lost = miss_destination(gmap, bmap, lmap)
    print("\n  miss=%d 的去向拆解（决定改法方向）：" % tot_miss)
    print("    挂给了同文档别的患者 to_other=%d  → LLM 挂错人(主患者偏向)，需在结构上强制每患者认领"
          % to_other)
    print("    LLM判NONE但base挂对了  to_none =%d  → LLM 弃权过头，需放宽弃权措辞" % to_none)
    print("    base也没挂到(识别层缺) to_lost =%d  → 非归属问题，归属改不动" % to_lost)

    # ---------- delta 最负 top-N 明细 ----------
    rows.sort(key=lambda r: r[4])
    print("\ndelta 最负的前 %d 个患者（看 gold / base / llm 判断是 漏挂 还是 挂错人）：" % args.top)
    for (k, n, fb, fl, dl) in rows[:args.top]:
        y = gmap[k]
        g = bmap.get(k, set())
        l = lmap.get(k, set())
        print("  %s  gold=%d base_f1=%.2f llm_f1=%.2f Δ=%+.2f" % (str(k), n, fb, fl, dl))
        print("     gold : %s" % (sorted(y) or "∅"))
        print("     base : %s" % (sorted(g) or "∅"))
        print("     llm  : %s   [miss=%s extra=%s]"
              % (sorted(l) or "∅", sorted(y - l) or "∅", sorted(l - y) or "∅"))

    print("\n判读指引：")
    print("  · miss 主导（漏挂多、zeroed 高）→ LLM 判 NONE 过头，放宽弃权措辞 / 缩小上下文噪声。")
    print("  · extra 主导（多挂多）→ LLM 挂错人或没砍够背景，收紧措辞 / 强化否定与他人过滤。")
    print("  · ≤2 桶拖累最大 → 小患者是 macro 命门，优先保这些患者不被判空。")


if __name__ == "__main__":
    main()
