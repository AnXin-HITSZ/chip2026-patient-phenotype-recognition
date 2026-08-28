#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PatientPheX 数据检查工具（CHIP2026 面向生物医学全文的特定患者表型概念识别）。

功能：
  1. 复算数据规模统计（文献/患者/表型实体/唯一 HPO/否定/复合/关联/患者相关 HPO）。
  2. 校验 offset：依据每个 section 的 offset+text 重建全局文本，检查 entities 与
     patient.mention 的 (offset, length, text) 是否与重建文本对齐。
  3. 校验 identifier 是否落在给定 HPO 版本的 Phenotypic abnormality (HP:0000118) 分支。

用法：
  python scripts/inspect_data.py data/PatientPheX-V1-A/PatientPheX-train.jsonl
  python scripts/inspect_data.py data/PatientPheX-V1-A/PatientPheX-train.jsonl \
      --obo data/PatientPheX-V1-A/hp.obo --check-offset --check-hpo

无第三方依赖，仅用标准库。
"""
import argparse
import json
import sys
from collections import Counter

# Windows 控制台默认可能是 GBK，强制 UTF-8 输出以正确显示中文。
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except AttributeError:  # Python < 3.7
    pass


def load_jsonl(path):
    docs = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                docs.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"[!] 第 {i} 行 JSON 解析失败: {e}", file=sys.stderr)
    return docs


def split_ids(identifier):
    """把 identifier 拆成独立 HPO ID 列表（复合表型按分号拆分，忽略 -1）。"""
    out = []
    for x in str(identifier).split(";"):
        x = x.strip()
        if x and x != "-1":
            out.append(x)
    return out


def stats(docs):
    n = len(docs)
    patients = sum(len(d.get("patient", [])) for d in docs)
    entities = sum(len(d.get("entities", [])) for d in docs)
    unique_hpo, patient_hpo = set(), set()
    neg = comp = noid = pairs = 0
    for d in docs:
        for e in d.get("entities", []):
            ident = str(e.get("identifier"))
            if e.get("note") == "NO":
                neg += 1
            if ";" in ident:
                comp += 1
            if "-1" in [x.strip() for x in ident.split(";")]:
                noid += 1
            unique_hpo.update(split_ids(ident))
        for a in d.get("association", []):
            ph = a.get("phenotype", [])
            pairs += len(ph)
            for x in ph:
                patient_hpo.update(split_ids(x))
    return {
        "文献数": n,
        "患者数": patients,
        "表型实体提及数": entities,
        "唯一 HPO ID 数": len(unique_hpo),
        "否定表型数": neg,
        "复合表型数": comp,
        "无 ID 表型数(-1)": noid,
        "患者表型关联数": pairs,
        "患者相关唯一 HPO ID 数": len(patient_hpo),
    }


def section_counts(docs):
    c = Counter()
    for d in docs:
        for s in d.get("full_text", []):
            c[s.get("section_type")] += 1
    return c


def build_global_text(doc):
    """依据每段的 offset 重建全局字符串（段间空隙用空格填充）。"""
    segs = sorted(doc.get("full_text", []), key=lambda s: s.get("offset", 0))
    end = 0
    for s in segs:
        end = max(end, s.get("offset", 0) + len(s.get("text", "")))
    buf = [" "] * end
    for s in segs:
        off = s.get("offset", 0)
        txt = s.get("text", "")
        for i, ch in enumerate(txt):
            if 0 <= off + i < end:
                buf[off + i] = ch
    return "".join(buf)


def check_offsets(docs, limit=20):
    """检查 entities 与 mention 的 (offset,length,text) 是否与重建全局文本对齐。"""
    problems = []
    for d in docs:
        gt = build_global_text(d)
        items = []
        for e in d.get("entities", []):
            items.append(("entity", e.get("offset"), e.get("length"), e.get("text")))
        for p in d.get("patient", []):
            for m in p.get("mention", []):
                items.append(("mention", m.get("offset"), m.get("length"), m.get("text")))
        for kind, off, length, text in items:
            if off is None or text is None:
                continue
            got = gt[off: off + (length if length is not None else len(text))]
            if got != text:
                problems.append((d.get("pmc_id"), kind, off, text, got))
    print(f"\n[offset 校验] 共检查 {len(docs)} 篇；发现 {len(problems)} 处不匹配")
    for pmc, kind, off, text, got in problems[:limit]:
        print(f"  - pmc={pmc} {kind}@{off} 期望={text!r} 实际={got!r}")
    if len(problems) > limit:
        print(f"  ... 其余 {len(problems) - limit} 处省略")
    return problems


def load_hp0000118_descendants(obo_path):
    """解析 obo，返回 HP:0000118 (Phenotypic abnormality) 分支的所有后代 ID 集合。"""
    parents = {}          # child -> set(parents) via is_a
    alt_to_main = {}      # alt_id / replaced_by -> main id
    all_terms = set()
    cur = None
    obsolete = set()
    with open(obo_path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line == "[Term]":
                cur = {"id": None, "is_a": set(), "alt": set(), "obs": False, "repl": None}
                continue
            if line.startswith("[") and cur is not None:
                # 结束一个 term 块（遇到其它 stanza）
                cur = None
                continue
            if cur is None:
                continue
            if line.startswith("id: "):
                cur["id"] = line[4:].strip()
                all_terms.add(cur["id"])
                parents.setdefault(cur["id"], set())
            elif line.startswith("is_a: "):
                pid = line[6:].split("!")[0].strip()
                cur["is_a"].add(pid)
                parents.setdefault(cur["id"], set()).add(pid)
            elif line.startswith("alt_id: "):
                alt_to_main[line[8:].strip()] = cur["id"]
            elif line.startswith("is_obsolete: true"):
                obsolete.add(cur["id"])
            elif line.startswith("replaced_by: "):
                alt_to_main[cur["id"]] = line[13:].strip()

    # 自顶向下 BFS：找 HP:0000118 的所有后代
    children = {}
    for c, ps in parents.items():
        for p in ps:
            children.setdefault(p, set()).add(c)
    root = "HP:0000118"
    desc = set()
    stack = [root]
    while stack:
        cur_id = stack.pop()
        for ch in children.get(cur_id, ()):
            if ch not in desc:
                desc.add(ch)
                stack.append(ch)
    desc.add(root)
    return desc, alt_to_main, obsolete


def check_hpo(docs, obo_path, limit=20):
    desc, alt_to_main, obsolete = load_hp0000118_descendants(obo_path)
    print(f"\n[HPO 校验] HP:0000118 分支后代数(含根)={len(desc)}，过时词条={len(obsolete)}")
    bad = []
    for d in docs:
        for e in d.get("entities", []):
            for hid in split_ids(e.get("identifier")):
                mapped = alt_to_main.get(hid, hid)
                if mapped not in desc:
                    bad.append((d.get("pmc_id"), hid, mapped in obsolete))
    print(f"  不在 HP:0000118 分支内的实体 ID 数：{len(bad)}")
    for pmc, hid, is_obs in bad[:limit]:
        print(f"  - pmc={pmc} id={hid} {'(过时)' if is_obs else ''}")
    if len(bad) > limit:
        print(f"  ... 其余 {len(bad) - limit} 处省略")
    return bad


def main():
    ap = argparse.ArgumentParser(description="PatientPheX 数据检查工具")
    ap.add_argument("jsonl", help="待检查的 jsonl 文件")
    ap.add_argument("--obo", help="hp.obo 路径（用于 --check-hpo）")
    ap.add_argument("--check-offset", action="store_true", help="校验 offset/length/text 对齐")
    ap.add_argument("--check-hpo", action="store_true", help="校验 identifier 是否在 HP:0000118 分支")
    args = ap.parse_args()

    docs = load_jsonl(args.jsonl)
    print(f"文件：{args.jsonl}")
    print("=" * 48)
    for k, v in stats(docs).items():
        print(f"  {k:<24}: {v}")

    print("\n[章节类型分布]")
    for name, cnt in section_counts(docs).most_common():
        print(f"  {str(name):<12}: {cnt}")

    if args.check_offset:
        check_offsets(docs)

    if args.check_hpo:
        if not args.obo:
            print("\n[!] --check-hpo 需要 --obo 指定 hp.obo 路径", file=sys.stderr)
        else:
            check_hpo(docs, args.obo)


if __name__ == "__main__":
    main()
