#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
log_online.py —— 把天池 A/B 榜的线上成绩补录到某次实验上。

线上分数是花「提交次数」买来的宝贵数据，必须记录，用来积累
「本地 dev 涨幅 ≈ 线上涨幅」的经验。

用法（数字照抄天池成绩单那一行）:
    python scripts/log_online.py --tag baseline_dict --board A \
        --score 0.5878 --men 0.6446 --doc 0.7121 --mic 0.5138 --mac 0.4807 \
        --submit-ts "2026-08-28 17:03:40" --note "词典基线首次提交"

--tag 必须与本地 evaluate.py 记录时用的 tag 一致，才能挂到同一条实验上。
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
import explog


def main():
    ap = argparse.ArgumentParser(description="补录线上 A/B 榜成绩")
    ap.add_argument("--tag", required=True, help="实验名，须与本地记录的 tag 一致")
    ap.add_argument("--board", default="A", choices=["A", "B"], help="A 榜 / B 榜")
    ap.add_argument("--score", type=float, required=True)
    ap.add_argument("--men", type=float, help="task1_menF1")
    ap.add_argument("--doc", type=float, help="task1_docF1")
    ap.add_argument("--mic", type=float, help="task2_micF1")
    ap.add_argument("--mac", type=float, help="task2_macF1")
    ap.add_argument("--submit-ts", dest="submit_ts", help="天池显示的提交时间")
    ap.add_argument("--note", help="可选：补充说明（若该 tag 尚无 note）")
    args = ap.parse_args()

    online = {
        "board": args.board,
        "score": args.score,
        "task1_menF1": args.men,
        "task1_docF1": args.doc,
        "task2_micF1": args.mic,
        "task2_macF1": args.mac,
        "submit_ts": args.submit_ts,
    }
    rec = {"tag": args.tag, "note": args.note, "online": online}
    merged = explog.upsert(rec)
    print("已把 %s 榜成绩记入 tag=%s" % (args.board, args.tag))
    loc = merged.get("local", {})
    if loc.get("score") is not None:
        print("  本地 dev score = %.4f   线上 %s榜 score = %.4f   差异 %+.4f"
              % (loc["score"], args.board, args.score, args.score - loc["score"]))
    else:
        print("  （该 tag 尚无本地分数；先用 evaluate.py --tag %s 记录本地成绩）" % args.tag)


if __name__ == "__main__":
    main()
