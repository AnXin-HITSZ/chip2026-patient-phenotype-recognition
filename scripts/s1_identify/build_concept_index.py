#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
build_concept_index.py —— 用 SapBERT 把 HPO 概念库（HP:0000118 子树）编码成向量并缓存。

这是「SapBERT 提召回」方案的第 1 步，也是这张 4090 第一次真正干活。
产物（概念向量库）编码一次即可反复复用，后续检索脚本直接加载，不必重编。

为什么要它：
  词典基线 precision≈0.80 但 recall≈0.56——只认识收录过的字符串，漏掉同义词/变体。
  SapBERT 把生物医学术语编码成向量，同义词在向量空间里彼此靠近。于是「文本里
  词典没收录的表达」可以通过「向量最近邻找到对应 HPO 概念」被补回来。
  第一步就是先把 19087 个 HPO 概念（每个概念的 name + 所有同义词）编码成向量库。

已核实的 SapBERT 官方用法（用错会掉性能）：
  - 模型 id : cambridgeltl/SapBERT-from-PubMedBERT-fulltext（英文生物医学，对口本赛题）
  - 池化    : CLS token，即 last_hidden_state[:, 0, :]（不是 mean pooling！）
  - 维度    : 768
  - 相似度  : cosine —— 本脚本对向量做 L2 归一化后存盘，检索时「内积 == cosine」

GPU 纪律：本项目约定全程用 GPU。检测不到 CUDA 直接报错退出，绝不静默退回 CPU，
         以免你误以为在用 GPU。

用法（在项目根目录、GPU 正常开机下运行）：
    # 先冒烟测试：只编码前 500 条，验证流程跑通（几秒）
    python scripts/s1_identify/build_concept_index.py --obo data/PatientPheX-V1-A/hp.obo --limit 500

    # 确认无误后全量编码（4090 约 1-2 分钟），产物存到 outputs/sapbert/
    python scripts/s1_identify/build_concept_index.py --obo data/PatientPheX-V1-A/hp.obo

产物（默认 outputs/sapbert/，已被 .gitignore 忽略）：
    concept_emb.pt      [M, 768] float32，已 L2 归一化的概念向量
    concept_meta.jsonl  M 行，每行 {"hpo_id":..., "name":...}，与向量逐行对应
    manifest.json       元信息（模型/池化/维度/条数），检索脚本据此核对
"""
import argparse
import json
import os
import sys
import time

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except AttributeError:
    pass

# 复用词典基线里已验证的 OBO 解析与子树抽取（与 evaluate.py import explog 同一手法）
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "core"))
from baseline_dict import parse_obo, descendants, ROOT  # noqa: E402

MODEL_NAME = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"


def collect_concept_names(terms, is_a):
    """收集 HP:0000118 子树内每个非 obsolete 概念的 name + 同义词。

    返回 (entries, n_concept)：
      entries   = [(hpo_id, name_string), ...]  一个概念的每个别名各占一条
      n_concept = 参与的概念数（去重前的概念个数）
    同一概念内按小写去重（避免 name 与某同义词字面重复）。不同概念间不去重
    （不同 HPO 概念可能共享同名别名，检索时靠向量+就近逻辑处理，这里如实保留）。
    """
    keep = descendants(ROOT, is_a)
    entries = []
    n_concept = 0
    for hid in sorted(keep):
        t = terms.get(hid)
        if not t or t["obsolete"]:
            continue
        n_concept += 1
        names = []
        if t["name"]:
            names.append(t["name"])
        names.extend(t["syns"])
        seen = set()
        for nm in names:
            nm2 = " ".join(nm.split())        # 折叠多余空白
            key = nm2.lower()
            if not nm2 or key in seen:
                continue
            seen.add(key)
            entries.append((hid, nm2))
    return entries, n_concept


def main():
    ap = argparse.ArgumentParser(description="用 SapBERT 编码 HPO 概念库并缓存")
    ap.add_argument("--obo", required=True, help="hp.obo 路径")
    ap.add_argument("--out-dir", default=os.path.join("outputs", "sapbert"),
                    help="产物目录（默认 outputs/sapbert，已被 .gitignore 忽略）")
    ap.add_argument("--model", default=MODEL_NAME, help="HuggingFace 模型 id")
    ap.add_argument("--batch-size", type=int, default=256, help="编码 batch 大小（4090 可用 256+）")
    ap.add_argument("--max-length", type=int, default=32, help="分词最大长度（HPO 名字通常很短）")
    ap.add_argument("--limit", type=int, default=0, help="只编码前 N 条做冒烟测试；0=全量")
    ap.add_argument("--fp16", action="store_true", help="半精度编码：更快更省显存（结果几乎不变）")
    ap.add_argument("--force", action="store_true", help="覆盖已有缓存重新编码")
    args = ap.parse_args()

    # torch/transformers 是重依赖，放到参数解析后再 import，--help 才不会卡
    import torch
    from transformers import AutoModel, AutoTokenizer

    # —— GPU 硬检查：本项目全程用 GPU，检测不到就报错退出 ——
    if not torch.cuda.is_available():
        print("❌ 未检测到 GPU（torch.cuda.is_available()=False）。")
        print("   本项目约定全程用 GPU，已中止。先跑 `python scripts/gpu_check.py` 排查。")
        sys.exit(2)
    device = torch.device("cuda:0")
    print("使用 GPU :", torch.cuda.get_device_name(0))

    emb_path = os.path.join(args.out_dir, "concept_emb.pt")
    meta_path = os.path.join(args.out_dir, "concept_meta.jsonl")
    manifest_path = os.path.join(args.out_dir, "manifest.json")
    if os.path.exists(emb_path) and not args.force:
        print("已存在缓存 %s（加 --force 可覆盖重建）。跳过编码。" % emb_path)
        return

    print("解析 hp.obo ...")
    terms, is_a = parse_obo(args.obo)
    entries, n_concept = collect_concept_names(terms, is_a)
    if args.limit:
        entries = entries[:args.limit]
        print("  [冒烟测试] 只取前 %d 条" % args.limit)
    names = [e[1] for e in entries]
    print("  概念数(非 obsolete)       : %d" % n_concept)
    print("  名字条目数(name+同义词去重): %d" % len(entries))

    print("加载 SapBERT: %s ..." % args.model)
    print("  (首次运行会从 HuggingFace 下载约 400MB，请确保已开学术加速或设 HF 镜像)")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).to(device).eval()  # .to("cuda") 把权重搬上卡
    if args.fp16:
        model = model.half()

    bs = args.batch_size
    embs = []
    t0 = time.perf_counter()
    # torch.no_grad(): 只做前向推理、不建反向图，省显存又提速（推理必备）
    with torch.no_grad():
        for i in range(0, len(names), bs):
            batch = names[i:i + bs]
            # 直接调用 tokenizer（__call__）是官方现推荐写法，等价于旧的
            # batch_encode_plus，且兼容新版 transformers（旧方法名已移除）
            toks = tok(
                batch, padding=True, truncation=True,
                max_length=args.max_length, return_tensors="pt")
            toks = {k: v.to(device) for k, v in toks.items()}  # 输入也要搬上卡
            cls = model(**toks)[0][:, 0, :]                    # [0]=last_hidden_state; [:,0,:]=CLS
            cls = torch.nn.functional.normalize(cls, p=2, dim=1)  # L2 归一化 → 内积即 cosine
            embs.append(cls.float().cpu())                     # 搬回 CPU 累积，释放显存
            if (i // bs) % 20 == 0:
                done = min(i + bs, len(names))
                mem = torch.cuda.memory_allocated(device) / 1024**2
                print("  编码 %6d / %6d   显存 %.0f MB" % (done, len(names), mem))
    emb = torch.cat(embs, 0)  # [M, 768] float32, on CPU
    dt = time.perf_counter() - t0
    print("编码完成: 形状 %s，耗时 %.1f 秒（约 %.0f 条/秒）"
          % (tuple(emb.shape), dt, len(names) / dt if dt > 0 else 0))

    os.makedirs(args.out_dir, exist_ok=True)
    torch.save(emb, emb_path)
    with open(meta_path, "w", encoding="utf-8") as f:
        for hid, nm in entries:
            f.write(json.dumps({"hpo_id": hid, "name": nm}, ensure_ascii=False) + "\n")
    manifest = {
        "model": args.model,
        "pooling": "cls",
        "normalized": True,
        "similarity": "cosine (dot product of L2-normalized vectors)",
        "dim": int(emb.shape[1]),
        "num_entries": int(emb.shape[0]),
        "num_concepts": n_concept,
        "max_length": args.max_length,
        "fp16_encode": bool(args.fp16),
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print("已保存:")
    print("  向量  : %s   (%.0f MB)" % (emb_path, os.path.getsize(emb_path) / 1024**2))
    print("  元数据: %s" % meta_path)
    print("  清单  : %s" % manifest_path)
    print("下一步：写检索脚本 baseline_sapbert.py，在词典未命中区用这个向量库补召回。")


if __name__ == "__main__":
    main()
