#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
baseline_dict.py —— 阶段 0「词典匹配」基线（零深度学习）。

思路（对应 docs/07 推荐组合里「字典打底」那一步）：
  1) 解析 hp.obo，取 HP:0000118 子树，收集每个概念的 name + 同义词，建
     "小写短语 -> HPO ID" 词典；
  2) 用 docs/06 的全局 offset 约定重建每篇文章的全局文本；
  3) 在全局文本上做「最长优先、不重叠」的词典匹配，产出 entities
     （offset/length 为全局坐标，text 取原文切片，note=null）；
  4) 子任务 2：把每个命中表型按 offset「就近」挂到最近的患者 mention，
     按 patient_id 去重，作为 association。

这是最简基线：不处理否定/复合/缩写，同名歧义按词典任取一个 ID。
目的是「跑通格式 + 拿到第一个可对比的分数」，之后逐块替换升级。

用法：
    # 在 dev 题目版上生成预测，再用 evaluate.py 打分
    python scripts/core/baseline_dict.py \
        --input data/split/dev_input.jsonl \
        --obo   data/PatientPheX-V1-A/hp.obo \
        --out   pred_dev.jsonl
    python scripts/core/evaluate.py --gold data/split/dev.jsonl --pred pred_dev.jsonl
"""
import argparse
import json
import re
import sys
from collections import deque

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except AttributeError:
    pass

ROOT = "HP:0000118"
MAX_NGRAM = 8          # 词典短语最多匹配的「词」数
MIN_LEN = 3            # 过短短语（如单字母缩写）不进词典，减少误命中
TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


# ----------------------------- OBO 解析 -----------------------------

def parse_obo(path):
    """返回 (terms, is_a)：
       terms[id] = {"name":..., "syns":[...], "obsolete":bool, "replaced_by":id|None, "alt_ids":[...]}
       is_a[id]  = set(parent_ids)
    """
    terms = {}
    is_a = {}
    cur = None
    with open(path, encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n")
            if line == "[Term]":
                cur = {"id": None, "name": None, "syns": [],
                       "obsolete": False, "replaced_by": None, "alt_ids": []}
            elif line.startswith("[") and line != "[Term]":
                cur = None
            elif cur is not None:
                if line.startswith("id: "):
                    cur["id"] = line[4:].strip()
                elif line.startswith("name: "):
                    cur["name"] = line[6:].strip()
                elif line.startswith("synonym: "):
                    m = re.search(r'"(.*?)"', line)
                    if m:
                        cur["syns"].append(m.group(1))
                elif line.startswith("is_a: "):
                    parent = line[6:].split("!")[0].strip()
                    if cur["id"]:
                        is_a.setdefault(cur["id"], set()).add(parent)
                    else:
                        cur.setdefault("_pending_isa", []).append(parent)
                elif line.startswith("alt_id: "):
                    cur["alt_ids"].append(line[8:].strip())
                elif line.startswith("is_obsolete: true"):
                    cur["obsolete"] = True
                elif line.startswith("replaced_by: "):
                    cur["replaced_by"] = line[13:].strip()
                # 块结束由下一个 [Term]/[ 处理；这里在 id 确定后落库
                if cur.get("id") and cur["id"] not in terms:
                    terms[cur["id"]] = cur
                    # 补挂在 id 之前出现的 is_a
                    for p in cur.pop("_pending_isa", []):
                        is_a.setdefault(cur["id"], set()).add(p)
    return terms, is_a


def descendants(root, is_a):
    """由 child->parent 的 is_a 反建 parent->children，BFS 取 root 全部后代（含自身）。"""
    children = {}
    for c, parents in is_a.items():
        for p in parents:
            children.setdefault(p, set()).add(c)
    seen = {root}
    dq = deque([root])
    while dq:
        n = dq.popleft()
        for c in children.get(n, ()):
            if c not in seen:
                seen.add(c)
                dq.append(c)
    return seen


# ----------------------------- 词典构建 -----------------------------

def norm_phrase(s):
    """短语规范化：取词元、小写、单空格连接（与匹配侧一致）。"""
    toks = TOKEN_RE.findall(s.lower())
    return " ".join(toks)


def build_dictionary(terms, is_a):
    """返回 phrase(str) -> HPO ID。仅 HP:0000118 子树、非 obsolete。
       同短语冲突时保留首个（基线不做消歧）。"""
    keep = descendants(ROOT, is_a)
    phrase2id = {}
    n_terms = 0
    for hid in keep:
        t = terms.get(hid)
        if not t or t["obsolete"]:
            continue
        n_terms += 1
        strings = []
        if t["name"]:
            strings.append(t["name"])
        strings.extend(t["syns"])
        for s in strings:
            p = norm_phrase(s)
            if len(p) < MIN_LEN or " " not in p and len(p) < 4:
                # 过滤过短/单短词，降低误命中（如 "a", "os"）
                if len(p) < 4:
                    continue
            if p and p not in phrase2id:
                phrase2id[p] = hid
    return phrase2id, keep, n_terms


# --------------------------- 全局文本重建 ---------------------------

def build_global(segs):
    if not segs:
        return ""
    end = max(s["offset"] + len(s["text"]) for s in segs)
    buf = [" "] * end
    for s in segs:
        o, t = s["offset"], s["text"]
        buf[o:o + len(t)] = list(t)
    return "".join(buf)


# ----------------------------- 词典匹配 -----------------------------

def match_entities(global_text, phrase2id):
    """最长优先、不重叠匹配。返回 entities 列表。"""
    tokens = [(m.group(0).lower(), m.start(), m.end())
              for m in TOKEN_RE.finditer(global_text)]
    ents = []
    i = 0
    n = len(tokens)
    while i < n:
        hit = None
        # 从最长 n-gram 往短试，取最长命中
        for span in range(min(MAX_NGRAM, n - i), 0, -1):
            phrase = " ".join(tokens[i + k][0] for k in range(span))
            hid = phrase2id.get(phrase)
            if hid is not None:
                start = tokens[i][1]
                end = tokens[i + span - 1][2]
                hit = (span, start, end, hid)
                break
        if hit:
            span, start, end, hid = hit
            ents.append({
                "identifier": hid,
                "type": "Phenotype",
                "offset": start,
                "length": end - start,
                "text": global_text[start:end],
                "note": None,
            })
            i += span
        else:
            i += 1
    return ents


# --------------------------- 子任务 2：就近归属 ---------------------------

def patient_anchors(doc):
    """返回 [(patient_id, [offset,...])]，用患者所有 mention 的 offset 作锚点。"""
    out = []
    for p in doc.get("patient", []):
        offs = [m["offset"] for m in p.get("mention", []) if "offset" in m]
        out.append((p["patient_id"], offs))
    return out


def associate(doc, entities, max_dist=None):
    """把每个命中表型按 offset 就近挂到最近患者，按 patient_id 去重。
       max_dist 非 None 时：表型到最近锚点距离 > max_dist（或无可用锚点走兜底）则判不挂(NONE)，
       用于砍掉离所有患者都很远的背景 FP（dev 分析：FP-N 距离中位 1469 ≫ TP 的 406）。"""
    anchors = patient_anchors(doc)
    assoc = {pid: [] for pid, _ in anchors}
    if not anchors:
        return []
    for e in entities:
        eo = e["offset"]
        best_pid, best_d = None, None
        for pid, offs in anchors:
            if not offs:
                continue
            d = min(abs(eo - o) for o in offs)
            if best_d is None or d < best_d:
                best_pid, best_d = pid, d
        if best_pid is None:
            # 无任何可用锚点：原兜底挂 anchors[0]；但开了距离阈值时视为不可信 → 砍
            if max_dist is not None:
                continue
            best_pid = anchors[0][0]
        elif max_dist is not None and best_d > max_dist:
            continue                    # 离最近锚点太远 → 判 NONE，不挂
        if e["identifier"] not in assoc[best_pid]:
            assoc[best_pid].append(e["identifier"])
    return [{"patient_id": pid, "phenotype": phs} for pid, phs in assoc.items()]


# ------------------------------- 主流程 -------------------------------

def main():
    ap = argparse.ArgumentParser(description="词典匹配基线，生成提交格式预测")
    ap.add_argument("--input", required=True, help="题目版 jsonl（entities/association 可空）")
    ap.add_argument("--obo", required=True, help="hp.obo 路径")
    ap.add_argument("--out", required=True, help="输出预测 jsonl")
    args = ap.parse_args()

    print("解析 hp.obo ...")
    terms, is_a = parse_obo(args.obo)
    phrase2id, keep, n_terms = build_dictionary(terms, is_a)
    print("  HP:0000118 子树 term 数: %d（非 obsolete 参与建词典: %d）" % (len(keep), n_terms))
    print("  词典短语条目数: %d" % len(phrase2id))

    docs = [json.loads(l) for l in open(args.input, encoding="utf-8") if l.strip()]
    print("处理 %d 篇文献 ..." % len(docs))

    out_docs = []
    n_ent = n_assoc = 0
    for d in docs:
        G = build_global(d.get("full_text", []))
        ents = match_entities(G, phrase2id)
        assoc = associate(d, ents)
        n_ent += len(ents)
        n_assoc += sum(len(a["phenotype"]) for a in assoc)
        out_docs.append({
            "pmc_id": d["pmc_id"],
            "pmid": d.get("pmid"),
            "entities": ents,
            "association": assoc,
        })

    with open(args.out, "w", encoding="utf-8") as f:
        for d in out_docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print("已写出: %s" % args.out)
    print("  预测实体总数: %d，患者-表型对: %d" % (n_ent, n_assoc))
    print("接下来用 evaluate.py 打分：")
    print("  python scripts/core/evaluate.py --gold <dev.jsonl> --pred %s" % args.out)


if __name__ == "__main__":
    main()
