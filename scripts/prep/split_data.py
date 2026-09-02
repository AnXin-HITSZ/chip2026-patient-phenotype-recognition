#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
split_data.py —— 把带答案的训练集切成 train / dev 两份。

用途：官方评测是在线提交（A 榜每天仅 5 次），无法靠提交调模型。
     我们从 80 篇训练集里划出一份 dev（带答案）用于本地打分，
     用 evaluate.py 复现四个 F1，做到「改一版 → 本地立刻看到分数涨没涨」。

划分方式：确定性划分（不随机，保证可复现）。默认把「下标 % k == 0」的
     文献放入 dev，其余进 train。k=5 时得到 dev 16 篇 / train 64 篇。

产出 data/split/ 五个文件：
     train.jsonl / dev.jsonl        —— 带答案（训练 / 打分尺子）
     train_input.jsonl / dev_input.jsonl —— 题目版（答案清空，同构测试集输入）
     其中 train_input 供「SapBERT 在 train 上识别」，为门控分类器提供
     与上线同分布的训练实体（详见 make_input_version 注释）。

用法：
    python scripts/prep/split_data.py data/PatientPheX-V1-A/PatientPheX-train.jsonl
    python scripts/prep/split_data.py <train.jsonl> --dev-every 5 --outdir data/split
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


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def dump_jsonl(docs, path):
    with open(path, "w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")


def make_input_version(docs):
    """生成「题目版」：清空 entities/association，模拟测试集输入（防止偷看答案）。
       用于把 train/dev 喂给识别/归属脚本时，与真实 A/B 榜输入同构。"""
    out = []
    for d in docs:
        d2 = dict(d)
        d2["entities"] = []
        d2["association"] = []
        out.append(d2)
    return out


def main():
    ap = argparse.ArgumentParser(description="切分训练集为 train/dev")
    ap.add_argument("train", help="PatientPheX-train.jsonl 路径")
    ap.add_argument("--dev-every", type=int, default=5,
                    help="每隔 k 篇取 1 篇进 dev（下标%%k==0），默认 5")
    ap.add_argument("--outdir", default=None,
                    help="输出目录，默认与输入同目录下的 split/")
    args = ap.parse_args()

    docs = load_jsonl(args.train)
    # 默认固定输出到 data/split/（相对项目根目录），与文档示例一致，避免路径困惑
    outdir = args.outdir or os.path.join("data", "split")
    os.makedirs(outdir, exist_ok=True)

    dev = [d for i, d in enumerate(docs) if i % args.dev_every == 0]
    train = [d for i, d in enumerate(docs) if i % args.dev_every != 0]

    train_path = os.path.join(outdir, "train.jsonl")
    dev_path = os.path.join(outdir, "dev.jsonl")
    dump_jsonl(train, train_path)
    dump_jsonl(dev, dev_path)

    # 题目版（清空答案）：train_input 供「SapBERT 在 train 上识别」用——门控分类器必须
    # 训练在 SapBERT 的真实预测实体上（含过召回/错映射/背景FP），才与上线时的输入同分布；
    # 若训练在 gold 实体上，分类器只见过「全对」的世界，学不会滤掉 FP。dev_input 同理供打分。
    train_input = make_input_version(train)
    dev_input = make_input_version(dev)
    train_input_path = os.path.join(outdir, "train_input.jsonl")
    dev_input_path = os.path.join(outdir, "dev_input.jsonl")
    dump_jsonl(train_input, train_input_path)
    dump_jsonl(dev_input, dev_input_path)

    print("总篇数        : %d" % len(docs))
    print("train(含答案) : %d 篇 -> %s" % (len(train), train_path))
    print("dev(含答案)   : %d 篇 -> %s" % (len(dev), dev_path))
    print("train(题目版) : %d 篇 -> %s  (答案已清空，供 SapBERT 识别→门控训练)"
          % (len(train_input), train_input_path))
    print("dev(题目版)   : %d 篇 -> %s  (答案已清空，喂给模型/基线打分)"
          % (len(dev_input), dev_input_path))


if __name__ == "__main__":
    main()
