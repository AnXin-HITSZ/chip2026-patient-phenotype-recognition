#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
report.py —— 把 experiments.jsonl 渲染成人类可读的实验记录表格 results/实验记录.md。

用法:
    python scripts/track/report.py

输出为 Markdown 表格：按时间排序，每条实验一行，
本地四项 F1 + Score，A/B 榜线上成绩，以及线上-本地得分差。
"""
import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except AttributeError:
    pass

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "core"))
import explog

MARKDOWN_PATH = os.path.join(explog.RESULTS_DIR, "实验记录.md")


def fmt(x, nd=4):
    return "--" if x is None else ("%.*f" % (nd, x))


def fmt_signed(x, nd=4):
    """带正负号的差值；None -> '--'，用于「相对上一条的提升」列。"""
    if x is None:
        return "--"
    arrow = "🟢" if x > 1e-9 else ("🔴" if x < -1e-9 else "⚪")
    return "%s %+.*f" % (arrow, nd, x)


def render(records):
    lines = [
        "# CHIP2026 实验记录",
        "",
        "> 由 `scripts/track/report.py` 自动生成，源数据：`results/experiments.jsonl`。",
        "> 本地分数来自 `evaluate.py --tag`，线上分数来自 `log_online.py`。",
        "> 「较上条」= 本行本地 Score 减去上一行本地 Score（🟢升/🔴降/⚪平）。",
        "",
        "| 时间 | 实验 tag | 说明 | 本地F1men | 本地F1doc | 本地F1mic | 本地F1mac | **本地Score** | 较上条 | 榜单 | **线上Score** | 差(线上-本地) |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    prev_local = None
    for r in records:
        ts = r.get("ts_local") or "-"
        tag = r.get("tag") or "-"
        note = (r.get("note") or "-").replace("|", "\\|")
        loc = r.get("local") or {}
        ol = r.get("online") or {}
        cur_local = loc.get("score")
        delta_prev = (cur_local - prev_local) if (cur_local is not None and prev_local is not None) else None
        lines.append(
            "| %s | %s | %s | %s | %s | %s | %s | **%s** | %s | %s | **%s** | %s |"
            % (
                ts, tag, note,
                fmt(loc.get("f1_men")), fmt(loc.get("f1_doc")),
                fmt(loc.get("f1_micro")), fmt(loc.get("f1_macro")),
                fmt(cur_local),
                fmt_signed(delta_prev),
                ol.get("board") or "-",
                fmt(ol.get("score")),
                fmt(ol.get("score") - cur_local) if ol.get("score") is not None and cur_local is not None else "--",
            )
        )
        if cur_local is not None:
            prev_local = cur_local
    return "\n".join(lines) + "\n"


def main():
    records = explog.load_all()
    records.sort(key=lambda r: r.get("ts_local") or "")
    md = render(records)
    explog.ensure_dir()
    with open(MARKDOWN_PATH, "w", encoding="utf-8") as f:
        f.write(md)
    print("已生成 %s（共 %d 条实验记录）" % (MARKDOWN_PATH, len(records)))


if __name__ == "__main__":
    main()