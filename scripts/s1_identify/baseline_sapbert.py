#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
baseline_sapbert.py —— 「SapBERT 提召回」方案第 2 步：在词典基线之上补召回。

策略（稳健优先，宁缺毋滥，防假阳性爆炸）：
  1) 词典先跑（复用 baseline_dict 的 build_dictionary + match_entities），
     命中区守住高精确率那部分，原样保留；
  2) 只在「词典未覆盖的词元区域」生成 1..K gram 候选 span
     （跳过纯数字 / ≤2 字符 / 首字符非字母，减少噪声）；
  3) 批量用 SapBERT 编码候选，在 GPU 上暴力检索概念库（第 1 步 build_concept_index
     产出的 outputs/sapbert/）取 top-1，score = cosine 相似度；
  4) score ≥ 阈值才收（默认 0.90，可 --sim-threshold 调，或 --scan-thresholds 自动扫）；
  5) 候选按 score 降序贪心去重：不与词典命中、不与已选候选重叠；
  6) 合并成 entities，子任务 2 沿用 baseline 的就近归属。

关键约束（对齐 evaluate.py 的 mention 级：offset+length+HPO_ID 全严格相等）：
  候选 span 的边界严格落在与词典同一套 TOKEN_RE=[A-Za-z0-9]+ 的词元边界上，
  所以 SapBERT 补的实体 offset/length 与金标准口径一致，能真正算 TP。

GPU 纪律：全程用 GPU，检测不到 CUDA 直接报错退出，绝不静默退回 CPU。

用法（项目根目录、GPU 正常开机、已跑过 build_concept_index.py）：
  # 单一阈值：生成 dev 预测
  python scripts/s1_identify/baseline_sapbert.py \
      --input data/split/dev_input.jsonl \
      --obo   data/PatientPheX-V1-A/hp.obo \
      --out   pred_dev_sapbert.jsonl \
      --sim-threshold 0.90

  # 自动扫阈值：用 dev 答案找最优阈值（需 --gold）
  python scripts/s1_identify/baseline_sapbert.py \
      --input data/split/dev_input.jsonl \
      --obo   data/PatientPheX-V1-A/hp.obo \
      --gold  data/split/dev.jsonl \
      --scan-thresholds 0.82,0.85,0.88,0.90,0.92,0.95
"""
import argparse
import json
import os
import re
import sys
import time

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except AttributeError:
    pass

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "core"))
from baseline_dict import (  # noqa: E402
    parse_obo, build_dictionary, build_global, match_entities, associate,
    TOKEN_RE,
)

MODEL_NAME = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
CAND_MAX_NGRAM = 4       # 候选最多几个词元；补召回不追长短语，控假阳性
DIGIT_RE = re.compile(r"^\d+$")


# ----------------------- 概念库加载 -----------------------

def load_concept_index(index_dir):
    """加载第 1 步产出的概念向量库。返回 (emb[M,768] float32 已L2归一化, meta[M], manifest)。"""
    emb_path = os.path.join(index_dir, "concept_emb.pt")
    meta_path = os.path.join(index_dir, "concept_meta.jsonl")
    manifest_path = os.path.join(index_dir, "manifest.json")
    for p in (emb_path, meta_path):
        if not os.path.exists(p):
            print("❌ 找不到 %s。请先运行 build_concept_index.py 生成概念库。" % p)
            sys.exit(2)
    import torch
    emb = torch.load(emb_path)              # [M, 768]
    if emb.dtype != torch.float32:
        emb = emb.float()
    meta = []
    with open(meta_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                meta.append(json.loads(line))
    manifest = {}
    if os.path.exists(manifest_path):
        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)
    if emb.shape[0] != len(meta):
        print("❌ 概念库损坏：向量 %d 条 vs 元数据 %d 条 不一致。请 --force 重建。"
              % (emb.shape[0], len(meta)))
        sys.exit(2)
    return emb, meta, manifest


# ----------------------- 候选生成 -----------------------

def occupied_intervals(dict_entities):
    """词典命中占据的字符区间 [(start,end), ...]。"""
    return [(e["offset"], e["offset"] + e["length"]) for e in dict_entities]


def token_occupied(tok_start, tok_end, occ):
    """该词元是否与任一词典命中区间重叠。"""
    for a, b in occ:
        if tok_start < b and tok_end > a:
            return True
    return False


def gen_candidates(global_text, dict_entities, max_ngram=CAND_MAX_NGRAM):
    """在词典未命中的词元区域生成 1..max_ngram gram 候选。
       返回 [(start, end, text)]，边界为 TOKEN_RE 词元边界（与词典口径一致）。"""
    occ = occupied_intervals(dict_entities)
    toks = [(m.start(), m.end()) for m in TOKEN_RE.finditer(global_text)]
    free = [not token_occupied(s, e, occ) for (s, e) in toks]
    n = len(toks)
    cands = []
    for i in range(n):
        if not free[i]:
            continue
        for span in range(1, max_ngram + 1):
            j = i + span - 1
            if j >= n or not free[j]:
                break
            start, end = toks[i][0], toks[j][1]
            text = global_text[start:end]
            stripped = text.strip()
            if len(stripped) <= 2:                 # 太短，噪声
                continue
            if DIGIT_RE.match(stripped):           # 纯数字
                continue
            if not stripped[0].isalpha():          # 首字符非字母
                continue
            cands.append((start, end, text))
    return cands


def gen_dict_extensions(global_text, dict_entities, max_ngram=CAND_MAX_NGRAM):
    """A3 修复：对每个词典命中，向左/右扩「自由词元」生成超集候选。

    背景：词典贪心短匹配（如 diabetes）占住词元后，gen_candidates 遇占位即 break，
    生成不出跨越它的更长表型（diabetes mellitus）。这里专门为每个词典命中补出它的超集，
    交 SapBERT 打分；build_entities 里若某超集 score≥阈值，就顶替被它完全包含的短匹配。

    只扩到「自由词元」（不被任何词典命中占据）→ 自然不会跨越*别的*词典命中，
    故每个超集恰好包含唯一一个词典命中（它的来源），顶替时不误伤别的命中。

    返回 [(start, end, text, src_iv)]，src_iv=(hs,he) 是被扩展词典命中的字符区间（可被顶替）。
    """
    occ = occupied_intervals(dict_entities)
    toks = [(m.start(), m.end()) for m in TOKEN_RE.finditer(global_text)]
    free = [not token_occupied(s, e, occ) for (s, e) in toks]
    n = len(toks)
    exts = []
    for de in dict_entities:
        hs, he = de["offset"], de["offset"] + de["length"]
        core = [i for i, (s, e) in enumerate(toks) if s < he and e > hs]  # 命中覆盖的词元
        if not core:
            continue                                   # 命中不落词元边界（罕见），跳过
        lc, rc = core[0], core[-1]
        core_ntok = rc - lc + 1
        if core_ntok >= max_ngram:
            continue                                   # 命中本身已达上限，无扩展空间
        budget = max_ngram - core_ntok                 # 还能加几个词元（左右合计）
        max_l = 0                                      # 左侧连续可扩的自由词元数
        while lc - max_l - 1 >= 0 and free[lc - max_l - 1] and max_l < budget:
            max_l += 1
        max_r = 0                                      # 右侧连续可扩的自由词元数
        while rc + max_r + 1 < n and free[rc + max_r + 1] and max_r < budget:
            max_r += 1
        src_iv = (hs, he)
        for l in range(0, max_l + 1):
            for r in range(0, max_r + 1):
                if l + r == 0:
                    continue                           # 不扩=词典命中本身，跳过
                if core_ntok + l + r > max_ngram:
                    continue                           # 左右合计超上限
                start, end = toks[lc - l][0], toks[rc + r][1]
                text = global_text[start:end]
                stripped = text.strip()
                if len(stripped) <= 2:                 # 与 gen_candidates 同一套卫生过滤
                    continue
                if DIGIT_RE.match(stripped):
                    continue
                if not stripped[0].isalpha():
                    continue
                exts.append((start, end, text, src_iv))
    return exts


# ----------------------- SapBERT 检索 -----------------------

def retrieve_scores(texts, tok, model, concept_emb, meta, device, bs, max_len):
    """对唯一候选文本批量编码 + GPU 暴力检索 top-1。
       返回 dict: text -> (hpo_id, name, score)。"""
    import torch
    uniq = list(dict.fromkeys(texts))     # 去重且保序，相同文本只编码一次
    out = {}
    with torch.no_grad():
        for i in range(0, len(uniq), bs):
            batch = uniq[i:i + bs]
            toks = tok(batch, padding=True, truncation=True,
                       max_length=max_len, return_tensors="pt")
            toks = {k: v.to(device) for k, v in toks.items()}
            cls = model(**toks)[0][:, 0, :]                        # CLS 池化
            cls = torch.nn.functional.normalize(cls.float(), p=2, dim=1)  # L2 归一化
            sims = cls @ concept_emb.t()                          # [B, M] 内积=cosine
            top_sim, top_idx = sims.max(dim=1)                    # top-1
            top_sim = top_sim.cpu().tolist()
            top_idx = top_idx.cpu().tolist()
            for t, s, idx in zip(batch, top_sim, top_idx):
                out[t] = (meta[idx]["hpo_id"], meta[idx]["name"], float(s))
    return out


# ----------------------- 合并（贪心去重） -----------------------

def build_entities(global_text, dict_entities, cand_spans, text2hit, threshold,
                   ext_spans=None):
    """词典实体 + 阈值以上的 SapBERT 候选（按 score 降序贪心，不重叠）。

    ext_spans（A3 词典边界扩展候选，[(start,end,text,src_iv)]）若给：某扩展候选 score≥阈值
    且只压住自己来源的那个词典命中（不碰别的已选区间）时，顶替该短匹配（长表型替短匹配）。
    ext_spans=None（默认）时行为与原版逐字节一致。
    """
    # 先收所有过阈值候选：ext 位=None 普通候选（不可碰词典）；=src_iv 扩展候选（可顶替其来源命中）
    scored = []
    for (start, end, text) in cand_spans:
        hit = text2hit.get(text)
        if hit is None:
            continue
        hpo_id, name, score = hit
        if score >= threshold:
            scored.append((score, start, end, text, hpo_id, name, None))
    for (start, end, text, src_iv) in (ext_spans or []):
        hit = text2hit.get(text)
        if hit is None:
            continue
        hpo_id, name, score = hit
        if score >= threshold:
            scored.append((score, start, end, text, hpo_id, name, src_iv))
    scored.sort(key=lambda x: (-x[0], x[1], -(x[2] - x[1])))   # 高分优先，靠前优先，长者优先

    selected = occupied_intervals(dict_entities)[:]           # 已占区间（含词典）
    displaced = set()                                         # 被超集顶替掉的词典命中区间
    new_ents = []
    for score, start, end, text, hpo_id, name, src_iv in scored:
        if src_iv is None:
            if token_occupied(start, end, selected):          # 普通候选：碰任何已占即跳过（原逻辑）
                continue
        else:
            others = [iv for iv in selected if iv != src_iv]  # 扩展候选：可压住来源命中，不碰其余
            if token_occupied(start, end, others):
                continue
            if src_iv in selected:
                selected.remove(src_iv)                       # 顶替：摘掉短匹配
                displaced.add(src_iv)
        selected.append((start, end))
        new_ents.append({
            "identifier": hpo_id,
            "type": "Phenotype",
            "offset": start,
            "length": end - start,
            "text": text,
            "note": None,
            "retrieved_name": name,        # 非标准字段，评估不读，便于开发期核对
            "retrieved_score": round(score, 4),
        })
    kept_dict = [de for de in dict_entities
                 if (de["offset"], de["offset"] + de["length"]) not in displaced]
    return kept_dict + new_ents, len(new_ents)


# ----------------------- 主流程 -----------------------

def main():
    ap = argparse.ArgumentParser(description="SapBERT 提召回：在词典基线上补召回")
    ap.add_argument("--input", required=True, help="题目版 jsonl")
    ap.add_argument("--obo", required=True, help="hp.obo 路径")
    ap.add_argument("--index-dir", default=os.path.join("outputs", "sapbert"),
                    help="概念库目录（build_concept_index.py 的产物）")
    ap.add_argument("--out", help="输出预测 jsonl（单一阈值模式必填）")
    ap.add_argument("--sim-threshold", type=float, default=0.90, help="cosine 相似度阈值")
    ap.add_argument("--scan-thresholds", help="逗号分隔的多个阈值；扫描模式，需 --gold")
    ap.add_argument("--gold", help="金标准 jsonl（扫描模式打分用）")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--max-length", type=int, default=32)
    ap.add_argument("--limit-docs", type=int, default=0, help="只处理前 N 篇（冒烟用）；0=全量")
    ap.add_argument("--fp16", action="store_true", help="半精度编码，更快省显存")
    ap.add_argument("--extend-dict", action="store_true",
                    help="A3 修复：对每个词典命中补出向左右扩自由词元的超集候选，"
                         "score≥阈值则顶替短匹配（默认关，与原基线一致）")
    ap.add_argument("--cand-max-ngram", type=int, default=CAND_MAX_NGRAM,
                    help="A1：候选最多几个词元（默认 %d）；提高可召回超 4 词元的长表型" % CAND_MAX_NGRAM)
    args = ap.parse_args()

    import torch
    from transformers import AutoModel, AutoTokenizer

    if not torch.cuda.is_available():
        print("❌ 未检测到 GPU（torch.cuda.is_available()=False）。本项目全程用 GPU，已中止。")
        print("   先跑 `python scripts/gpu_check.py` 排查。")
        sys.exit(2)
    device = torch.device("cuda:0")
    print("使用 GPU :", torch.cuda.get_device_name(0))

    # 词典
    print("解析 hp.obo + 建词典 ...")
    terms, is_a = parse_obo(args.obo)
    phrase2id, keep, n_terms = build_dictionary(terms, is_a)
    print("  词典短语条目数: %d" % len(phrase2id))

    # 概念库
    print("加载概念库 %s ..." % args.index_dir)
    concept_emb, meta, manifest = load_concept_index(args.index_dir)
    concept_emb = concept_emb.to(device)
    print("  概念向量: %s，维度 %d，条数 %d"
          % (tuple(concept_emb.shape), concept_emb.shape[1], len(meta)))

    # 模型
    print("加载 SapBERT: %s ..." % MODEL_NAME)
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device).eval()
    if args.fp16:
        model = model.half()

    # 读数据
    docs = [json.loads(l) for l in open(args.input, encoding="utf-8") if l.strip()]
    if args.limit_docs:
        docs = docs[:args.limit_docs]
    print("处理 %d 篇文献 ..." % len(docs))

    # 逐篇：词典匹配 + 候选生成（缓存，供单/多阈值复用）
    per_doc = []          # [(doc, global_text, dict_entities, cand_spans, ext_spans)]
    all_cand_texts = []
    t0 = time.perf_counter()
    for d in docs:
        G = build_global(d.get("full_text", []))
        dict_ents = match_entities(G, phrase2id)
        cands = gen_candidates(G, dict_ents, max_ngram=args.cand_max_ngram)
        exts = (gen_dict_extensions(G, dict_ents, max_ngram=args.cand_max_ngram)
                if args.extend_dict else [])
        per_doc.append((d, G, dict_ents, cands, exts))
        all_cand_texts.extend(c[2] for c in cands)
        all_cand_texts.extend(e[2] for e in exts)
    n_dict = sum(len(x[2]) for x in per_doc)
    n_ext = sum(len(x[4]) for x in per_doc)
    print("  词典实体总数: %d，候选 span 总数: %d（唯一 %d）%s"
          % (n_dict, len(all_cand_texts), len(set(all_cand_texts)),
             ("，其中词典扩展 %d" % n_ext) if args.extend_dict else ""))

    # 批量检索所有唯一候选
    print("SapBERT 编码 + 检索候选 ...")
    tr0 = time.perf_counter()
    text2hit = retrieve_scores(all_cand_texts, tok, model, concept_emb, meta,
                               device, args.batch_size, args.max_length)
    print("  检索完成，唯一候选 %d 条，耗时 %.1f 秒" % (len(text2hit), time.perf_counter() - tr0))

    # 分数分布速览（帮助定阈值）
    scores = sorted((v[2] for v in text2hit.values()), reverse=True)
    if scores:
        def pct(p):
            return scores[min(len(scores) - 1, int(len(scores) * p))]
        print("  候选 top-1 相似度分布: max=%.3f  p10=%.3f  p25=%.3f  中位=%.3f  min=%.3f"
              % (scores[0], pct(0.10), pct(0.25), pct(0.50), scores[-1]))

    def assemble(threshold):
        """按阈值组装所有文档的预测。返回 (out_docs, n_new_total)。"""
        out_docs = []
        n_new = 0
        for (d, G, dict_ents, cands, exts) in per_doc:
            ents, k = build_entities(G, dict_ents, cands, text2hit, threshold,
                                     ext_spans=exts)
            n_new += k
            assoc = associate(d, ents)
            out_docs.append({
                "pmc_id": d["pmc_id"], "pmid": d.get("pmid"),
                "entities": ents, "association": assoc,
            })
        return out_docs, n_new

    # ---- 扫描模式：多阈值打分选最优 ----
    if args.scan_thresholds:
        if not args.gold:
            ap.error("--scan-thresholds 需要同时提供 --gold 才能打分")
        from evaluate import load_jsonl, evaluate
        gold_docs = load_jsonl(args.gold)
        ths = [float(x) for x in args.scan_thresholds.split(",") if x.strip()]
        print("\n===== 阈值扫描（越高越保守）=====")
        print("  阈值   净新增   F1_men   F1_doc   F1_mic   F1_mac   Score")
        best = None
        for th in ths:
            out_docs, n_new = assemble(th)
            m = evaluate(gold_docs, out_docs, verbose=False)
            print("  %.2f   %6d   %.4f   %.4f   %.4f   %.4f   %.4f"
                  % (th, n_new, m["f1_men"], m["f1_doc"], m["f1_micro"],
                     m["f1_macro"], m["score"]))
            if best is None or m["score"] > best[1]:
                best = (th, m["score"], n_new)
        print("  ---- 词典基线对照 Score=0.5427 ----")
        print("  >>> 最优阈值 = %.2f，Score = %.4f（净新增 %d 实体）<<<"
              % (best[0], best[1], best[2]))
        print("确定阈值后用单一阈值模式加 --out 生成正式预测。")
        return

    # ---- 单一阈值模式：生成预测文件 ----
    if not args.out:
        ap.error("单一阈值模式需要 --out 指定输出文件")
    out_docs, n_new = assemble(args.sim_threshold)
    with open(args.out, "w", encoding="utf-8") as f:
        for d in out_docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    n_ents = sum(len(d["entities"]) for d in out_docs)     # 真实体总数（顶替后，已扣被摘短匹配）
    print("\n已写出: %s" % args.out)
    print("  阈值 %.2f 下，SapBERT 补 %d，实体总数 %d（词典命中 %d，扩展开=%s）"
          % (args.sim_threshold, n_new, n_ents, n_dict, args.extend_dict))
    print("  全程耗时 %.1f 秒" % (time.perf_counter() - t0))
    print("接下来打分：")
    print("  python scripts/core/evaluate.py --gold data/split/dev.jsonl --pred %s \\" % args.out)
    print("      --tag sapbert_v1 --note \"SapBERT补召回 阈值%.2f\"" % args.sim_threshold)


if __name__ == "__main__":
    main()
