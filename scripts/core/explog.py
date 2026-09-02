#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
explog.py —— 实验记录系统的共享底座（被 evaluate.py / log_online.py / report.py 复用）。

唯一真相源: results/experiments.jsonl —— 每行一条 JSON 记录，只追加(append)，不改写。
一条记录代表「一次实验」，用 tag 作为主键（同 tag 再次记录会合并/更新，见 upsert）。

记录字段（都可为空）:
  ts_local   记录时间（由调用方传入的时间字符串；本环境脚本内不取系统时间）
  tag        实验名，主键，如 "baseline_dict"
  note       一句话说明这次改了什么
  split      本地评测所用 gold 文件（如 data/split/dev.jsonl）
  pred       预测文件
  local      本地四项 F1 + score  {f1_men,f1_doc,f1_micro,f1_macro,score}
  online     线上成绩 {board:"A"/"B", score, task1_menF1, task1_docF1,
             task2_micF1, task2_macF1, submit_ts}

设计取舍: 不引第三方库；时间默认由 now_str() 自动取本地时间，
         也可由调用方用 --ts 显式覆盖（回填历史实验时用）。
"""
import json
import os
from datetime import datetime

RESULTS_DIR = "results"
STORE = os.path.join(RESULTS_DIR, "experiments.jsonl")


def now_str():
    """当前本地时间，格式 'YYYY-MM-DD HH:MM'。用于自动填充实验记录时间。"""
    return datetime.now().strftime("%Y-%m-%d %H:%M")


def ensure_dir():
    os.makedirs(RESULTS_DIR, exist_ok=True)


def load_all():
    if not os.path.exists(STORE):
        return []
    with open(STORE, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _write_all(records):
    ensure_dir()
    with open(STORE, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def upsert(record):
    """按 tag 合并：已存在则深合并（新值覆盖非空字段），否则追加。返回合并后的记录。"""
    records = load_all()
    tag = record.get("tag")
    idx = next((i for i, r in enumerate(records) if r.get("tag") == tag), None)
    if idx is None:
        records.append(record)
        merged = record
    else:
        merged = _deep_merge(records[idx], record)
        records[idx] = merged
    _write_all(records)
    return merged


def _deep_merge(base, new):
    """把 new 里的非空值合并进 base（dict 递归，其余直接覆盖）。"""
    out = dict(base)
    for k, v in new.items():
        if v is None:
            continue
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out
