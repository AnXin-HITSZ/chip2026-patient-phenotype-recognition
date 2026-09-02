#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
build_gate_dataset.py —— 为「患者-表型 门控分类器」构造训练/验证数据集。

它做什么：
  把 SapBERT 预测出的**每一个实体**抽成一条样本 = 【原始特征向量 + 标签】。
    * 特征：只用「题目版输入 + SapBERT 自己的预测 + 公开本体 hp.obo」能算出的数字，
            绝不碰 gold（见文末「不泄露标签」铁律）。
    * 标签：该实体的 HPO id（复合按 ';' 拆）是否落在**本篇 gold 关联的并集**里。
            落在里面 -> 1（该留给某个患者）；不在 -> 0（背景/方法/阴性/召回噪声，该弃）。

它不做什么（重要，职责分离）：
  * 不做独热编码 / 不做标准化 / 不做缺失值填充。这些「喂模型前的预处理」因
    逻辑回归(LR) 与树模型(HGBC) 要求不同，留到第 4 步 train_gate.py 里按模型分叉。
    本脚本只吐**原始信号**（section_type 保留字符串、retrieved_score 缺失就写 null）。
  * 不训练、不吃 GPU。纯标准库，本地 / AutoDL 无卡模式即可跑。

数据流（按 pmc_id join 三个文件）：
    --input  题目版 jsonl（出 patient[] 与 full_text[]：做患者锚点、章节、上下文）
    --pred   SapBERT 预测 jsonl（出 entities[]：每个实体 = 一条样本）
    --gold   带答案 jsonl（出 association[]：**只用来打标签**，特征绝不碰它）
    --obo    hp.obo（算 HPO 本体深度）
    --out    输出数据集 jsonl（每行一条样本：元信息 + 原始特征 + label）

标签口径与 evaluate.py 完全一致：split_ids() 拆复合、association→phenotype 取并集。

用法：
    # 先只做「标签自检」，确���标签逻辑对、正样本占比 ~58%（不需要特征填完）
    python scripts/prep/build_gate_dataset.py \
        --input data/split/train_input.jsonl \
        --pred  pred_train_sapbert.jsonl \
        --gold  data/split/train.jsonl \
        --obo   data/PatientPheX-V1-A/hp.obo \
        --out   data/split/gate_train.jsonl --label-only

    # 特征 feat_* 填完后，去掉 --label-only，产出完整数据集
    python scripts/prep/build_gate_dataset.py \
        --input data/split/train_input.jsonl --pred pred_train_sapbert.jsonl \
        --gold data/split/train.jsonl --obo data/PatientPheX-V1-A/hp.obo \
        --out data/split/gate_train.jsonl
    # dev 同样跑一份（第 4 步扫阈值/验证用）：--input dev_input --pred pred_dev_sapbert
    #   --gold dev.jsonl --out data/split/gate_dev.jsonl
"""
import argparse
import json
import os
import re
import sys
from collections import deque

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except AttributeError:
    pass

# 复用 core 里已验证的工具（import 约定见 scripts/README.md）
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "core"))
from baseline_dict import (  # noqa: E402
    ROOT, TOKEN_RE, parse_obo, build_global, patient_anchors,
)
from evaluate import split_ids  # noqa: E402


# ============================ 基础设施（已写好，无需改动） ============================

def section_index(full_text):
    """把 full_text 段落变成按 offset 排序的区间表 [(start, end, section_type)]。
       （与 associate_llm.section_index 同口径，复制进来避免 prep 反向依赖 s2_associate。）"""
    idx = []
    for s in full_text or []:
        o = s.get("offset", 0)
        t = s.get("text", "")
        idx.append((o, o + len(t), s.get("section_type") or s.get("type") or "?"))
    idx.sort(key=lambda x: x[0])
    return idx


def section_of(offset, sec_idx):
    """查 offset 落在哪个章节（段不重叠，线性够快）。找不到返回 '?'。"""
    for a, b, st in sec_idx:
        if a <= offset < b:
            return st
    return "?"


def build_depth_map(is_a, root=ROOT):
    """从 ROOT=HP:0000118 出发 BFS，返回 {HPO_id: 最短深度}。root 深度=0。
       只覆盖 ROOT 子树内的节点；子树外/obsolete 的 id 不在表里（查不到）。"""
    children = {}
    for c, parents in is_a.items():
        for p in parents:
            children.setdefault(p, set()).add(c)
    depth = {root: 0}
    dq = deque([root])
    while dq:
        n = dq.popleft()
        for c in children.get(n, ()):
            if c not in depth:
                depth[c] = depth[n] + 1
                dq.append(c)
    return depth


def gold_union(gold_doc):
    """本篇 gold 关联的所有 HPO token 并集（复合拆分、去重）。
       口径对齐 evaluate.patient_pheno_map：只从 association→phenotype 取，是子任务2 的正例集。
       这是**唯一**用到 gold 的地方，且只用于 label，绝不进特征。"""
    u = set()
    for a in (gold_doc or {}).get("association", []):
        for ph in a.get("phenotype", []):
            for tok in split_ids(ph):
                u.add(tok)
    return u


def make_label(entity, gold_toks):
    """标签：实体的 HPO id（复合拆分）只要有**任一** token 落在 gold 并集里 -> 1，否则 0。
       为何「任一即 1」：门控是对「整个实体」留/弃的决策（实体是一个不可拆的 mention），
       复合实体两个 token 共享同一套特征。绝大多数实体是单 id，此近似影响极小。"""
    toks = split_ids(entity.get("identifier"))
    return 1 if any(t in gold_toks for t in toks) else 0


# ============================ 特征提取（你来填 feat_*） ============================
# 规则回顾：
#   1. 只能用 entity 自身 + ctx（题目版输入派生）+ 公开本体，绝不碰 gold。
#   2. 返回**原始值**：类别就返回字符串，缺失就返回 None（不要在这里编码/填充/归一化）。
#   3. 每填好一个，用 --limit-docs 2 冒烟看这一维输出是否合理。
#
# ctx 字典（每篇共享，遍历时传入）字段：
#   ctx["global_text"]   -> str，重建的全局全文（可切上下文窗口）
#   ctx["global_len"]    -> int，全文长度（做归一化分母）
#   ctx["sec_idx"]       -> 章节区间表，配 section_of(offset, sec_idx) 用
#   ctx["anchors"]       -> [(patient_id, [offset,...])]，患者提及锚点
#   ctx["patient_count"] -> int，本篇患者数
#   ctx["depth_map"]     -> {HPO_id: depth}
#   ctx["window"]        -> int，上下文窗口字符数（--context-window）


# 否定/正常线索词表（模块级编译一次）。用 \b 词边界避免 'not' 误配 'notable'、
# 'normal' 误配 'abnormal'（'abnormal' 里 normal 前无词边界，\bnormal\b 不会命中）。
NEG_RE = re.compile(
    r"\b(?:no|not|without|absent|denies|denied|deny|negative|neither|nor|none|"
    r"ruled?\s+out|rules\s+out|cannot)\b")
NORMAL_RE = re.compile(
    r"\b(?:normal|unremarkable|within\s+normal\s+limits|wnl|unaffected)\b")


def feat_mention_len(e):
    """【范例·已填】mention 的字符长度与词数。极短(缩写)常是误召回。
       返回 (char_len:int, word_len:int)。照这个模式填其余 feat_*。"""
    text = e.get("text") or ""
    char_len = len(text)
    word_len = len(TOKEN_RE.findall(text))
    return char_len, word_len


def feat_score(e):
    """【TODO】SapBERT 检索可信度 + 是否词典命中。
       信号：retrieved_score 低=召回勉强=更可能是噪声；词典命中通常高精度。
       关键洞察：retrieved_score 只有 SapBERT 补召回的实体才有，词典命中的实体**没有**
                 这个字段。所以「缺失」本身就是信号 == 词典命中。
       提示：score = e.get("retrieved_score")；有值 -> is_dict_hit=0；无值 -> score=None、
             is_dict_hit=1。返回原始值即可，缺失填 None（不要填 0，0 是「最不相似」语义全错）。
       返回 (score: float|None, is_dict_hit: int 0/1)。"""
    score = e.get("retrieved_score")            # 词典命中的实体没有此字段 -> None
    is_dict_hit = 0 if score is not None else 1  # 缺失即词典命中；缺失本身就是信号
    return score, is_dict_hit


def feat_section(off, sec_idx):
    """【TODO】实体所在章节类型（原始字符串，独热留到 train 阶段做）。
       信号：CASE/RESULTS/ABSTRACT 多为患者特异(倾向留)；INTRO/DISCUSSION/METHODS
             多为疾病背景(倾向弃)。
       提示：用 section_of(off, sec_idx)。建议 .upper() 统一大小写，减少同义类别碎片。
       返回 section_type: str。"""
    return section_of(off, sec_idx).upper()


def feat_rel_position(off, global_len):
    """【TODO】实体在全文的相对位置 offset/全文长度，0~1。
       信号：背景铺垫常在开头(intro)或结尾(讨论)。
       提示：注意 global_len 可能为 0，用 max(1, global_len) 防除零。
       返回 float。"""
    return float(off / global_len) if global_len > 0 else 0.0


def feat_dist_to_anchor(off, anchors, global_len):
    """【TODO】实体到「最近患者提及」的归一化字符距离。
       信号：离任何患者都远 -> 不太像在讲某个具体患者(倾向弃)。
       提示：anchors=[(pid,[offsets])]；对所有 offset 求 min(abs(off-o))，再 /max(1,global_len)。
             本篇没有任何 anchor(无患者/无 offset) 时该返回什么？由你决定(建议 1.0=最远 或 None)，
             并在 train 阶段与缺失处理保持一致。
       返回 float|None。"""
    # 收集所有患者的所有 mention offset
    all_offs = [o for _, offs in anchors for o in offs]
    if not all_offs:
        return None                          # 本篇无患者/无锚点 -> 缺失(None)，交 train 阶段处理
    nearest = min(abs(off - o) for o in all_offs)
    return float(nearest) / max(1, global_len)  # 归一化到 ~[0,1]（篇很长时理论可略超，无妨）


def feat_negation_normal(global_text, off, length, window):
    """【TODO】上下文窗口内是否出现「否定」/「正常」线索。
       信号：'no fever'/'denied'/'ruled out' = 阴性；'liver was normal'/'unremarkable'
             = 有描述但正常 —— 两类都**不该**当异常表型挂给患者。
       提示：取 ctx 文本片段 global_text[max(0,off-window) : off+length+window]，.lower()。
             否定词表建议：no / not / without / absent / denied / negative / ruled out / rule out。
             正常词表建议：normal / unremarkable / within normal limits / wnl。
             用子串或分词匹配都行(注意 'not' 会误配 'notable'，可按词边界匹配更稳)。
       返回 (has_negation: int 0/1, has_normal: int 0/1)。"""
    left = max(0, off - window)
    right = off + length + window
    ctx_text = global_text[left:right].lower()
    has_negation = 1 if NEG_RE.search(ctx_text) else 0
    has_normal = 1 if NORMAL_RE.search(ctx_text) else 0
    return has_negation, has_normal


def feat_hpo_depth(identifier, depth_map):
    """【TODO】实体 HPO 在本体里的层级深度（从 HP:0000118 起 BFS）。
       信号：浅=泛类词(如「神经系统异常」大类，背景常客)；深=具体症状(更像真实表型)。
       提示：split_ids(identifier) 可能是复合，逐个查 depth_map。复合取 min 还是 max 由你定
             (min=最泛的那个更保守)；全都查不到(子树外/obsolete)返回什么也由你定(建议 None 或 -1)，
             与 train 阶段缺失处理保持一致。
       返回 int|None。"""
    depths = [depth_map[t] for t in split_ids(identifier) if t in depth_map]
    if not depths:
        return None          # 子树外/obsolete/'-1' 等查不到 -> 缺失(None)，交 train 阶段处理
    return min(depths)       # 复合取最浅(最泛)的那个，保守：更可能是背景大类


def extract_features(e, ctx):
    """把一个实体抽成特征 dict。逐维调用上面的 feat_*。
       未实现的 feat_* 会在此 raise，提示你「下一个填这里」——逐个填、逐个冒烟。"""
    off = e.get("offset", 0)
    length = e.get("length", 0)

    char_len, word_len = feat_mention_len(e)
    score, is_dict_hit = feat_score(e)
    section = feat_section(off, ctx["sec_idx"])
    rel_pos = feat_rel_position(off, ctx["global_len"])
    dist = feat_dist_to_anchor(off, ctx["anchors"], ctx["global_len"])
    has_neg, has_normal = feat_negation_normal(
        ctx["global_text"], off, length, ctx["window"])
    depth = feat_hpo_depth(e.get("identifier"), ctx["depth_map"])

    return {
        "section_type": section,
        "retrieved_score": score,
        "is_dict_hit": is_dict_hit,
        "mention_char_len": char_len,
        "mention_word_len": word_len,
        "hpo_depth": depth,
        "rel_position": rel_pos,
        "dist_to_nearest_anchor_norm": dist,
        "patient_count": ctx["patient_count"],
        "has_negation_cue": has_neg,
        "has_normal_cue": has_normal,
    }


# ============================ 主流程（已写好） ============================

def load_jsonl(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def main():
    ap = argparse.ArgumentParser(description="构造门控分类器数据集（特征+标签）")
    ap.add_argument("--input", required=True, help="题目版 jsonl（出 patient/full_text）")
    ap.add_argument("--pred", required=True, help="SapBERT 预测 jsonl（出 entities）")
    ap.add_argument("--gold", required=True, help="带答案 jsonl（只用于打标签）")
    ap.add_argument("--obo", required=True, help="hp.obo 路径（算 HPO 深度）")
    ap.add_argument("--out", required=True, help="输出数据集 jsonl")
    ap.add_argument("--context-window", type=int, default=120,
                    help="否定/正常线索的上下文窗口字符数（与 associate_llm 一致，默认 120）")
    ap.add_argument("--limit-docs", type=int, default=0, help="只处理前 N 篇（冒烟用）；0=全量")
    ap.add_argument("--label-only", action="store_true",
                    help="只做标签自检（不抽特征）：验证 label 逻辑与正样本占比，特征没填也能跑")
    args = ap.parse_args()

    inputs = load_jsonl(args.input)
    preds = {d["pmc_id"]: d for d in load_jsonl(args.pred)}
    golds = {d["pmc_id"]: d for d in load_jsonl(args.gold)}
    if args.limit_docs:
        inputs = inputs[:args.limit_docs]
    print("输入 %d 篇；预测覆盖 %d 篇；gold 覆盖 %d 篇" % (len(inputs), len(preds), len(golds)))

    # 标签深度表（--label-only 时用不到深度，但 parse_obo 很快，统一先建）
    print("解析 hp.obo 建深度表 ...")
    _, is_a = parse_obo(args.obo)
    depth_map = build_depth_map(is_a)
    print("  ROOT 子树深度表节点数: %d" % len(depth_map))

    samples = []
    n_pos = n_neg = 0
    n_ent_total = 0
    for d in inputs:
        pmc = d["pmc_id"]
        entities = (preds.get(pmc) or {}).get("entities", [])
        gtoks = gold_union(golds.get(pmc))
        n_ent_total += len(entities)

        G = build_global(d.get("full_text", []))
        ctx = {
            "global_text": G,
            "global_len": len(G),
            "sec_idx": section_index(d.get("full_text", [])),
            "anchors": patient_anchors(d),
            "patient_count": len(d.get("patient", [])),
            "depth_map": depth_map,
            "window": args.context_window,
        }

        for e in entities:
            label = make_label(e, gtoks)
            n_pos += label
            n_neg += (1 - label)
            if args.label_only:
                continue
            feats = extract_features(e, ctx)
            row = {
                "pmc_id": pmc,
                "offset": e.get("offset"),
                "identifier": e.get("identifier"),
                "text": e.get("text"),
                "label": label,
            }
            row.update(feats)
            samples.append(row)

    total = n_pos + n_neg
    print("\n===== 标签自检 =====")
    print("  实体(样本)总数: %d" % total)
    if total:
        print("  正样本(该留 1): %d  (%.1f%%)" % (n_pos, 100.0 * n_pos / total))
        print("  负样本(该弃 0): %d  (%.1f%%)" % (n_neg, 100.0 * n_neg / total))
        print("  （预期正样本占比 ~58%%；偏离太多先查标签/join 是否对齐）")

    if args.label_only:
        print("\n[--label-only] 未写出数据集。标签逻辑确认后，实现 feat_* 再去掉该开关。")
        return

    with open(args.out, "w", encoding="utf-8") as f:
        for r in samples:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("\n已写出数据集: %s（%d 条样本，每条 %d 维原始特征）"
          % (args.out, len(samples), len(samples[0]) - 5 if samples else 0))
    print("下一步：第 4 步 train_gate.py 读它 -> 预处理(独热/标准化/缺失) -> PyTorch 训练 LR。")


if __name__ == "__main__":
    main()
