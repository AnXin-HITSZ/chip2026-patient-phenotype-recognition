#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
associate_llm.py —— 子任务 2「患者-表型归属」的 LLM 后处理器（官方基线：Qwen3-8B）。

它做什么（以及不做什么）：
  * 只重算 association（谁得哪些表型），**不碰识别**——entities 原样来自子任务 1 的预测。
  * 用本地 Qwen3-8B 逐篇判断：每个已识别的表型 mention「属于哪个患者 / 不属于任何人(NONE)」。

两种运行模式（--gate-only 决定 LLM 到底输出什么）：
  * 默认（全权归属）：LLM 既判 keep/drop，又判「挂给哪个患者」。诊断（见 assoc_gate.py）
    表明它把两件事捆一起——门控(砍~42%疾病背景FP)干得好，但路由(多患者篇把小患者表型
    误挂给proband)干砸，16个小患者清零致 macF1 0.4030→0.3369。已弃用，仅留作对照。
  * --gate-only（门控式，推荐，dev Score 0.5604→0.5783）：**LLM 只做 keep/drop，路由还给
    就近**。门控式关联 = 就近路由(associate()) ∩ LLM 保留集(没判NONE的token)。就近按
    offset 分配、绝不抢小患者的表型给 proband(消除 macro 崩)，同时 LLM 判 NONE 的背景
    表型照砍(保住 micro 收益)。LLM 未覆盖/无实体等退化场景原样回退就近，永不比基线差。

为什么要它（病根，已用 train 量化）：
  现基线 baseline_dict.associate() 把每个表型按 offset 就近**强挂**给最近患者（关联率 100%），
  但 train 上真正进入某患者 gold 关联的表型中位仅 ~58%——约 42% 是疾病背景/方法/别的队列/
  阴性描述，本不该挂给任何人。SapBERT 多召回的表型被 100% 强挂，全成了子任务 2 的 FP。
  LLM 能回答「NONE（不属于任何列出的个体）」，就砍掉这 ~42% 过度关联。

数据流（按 pmc_id join 一个文件，A 榜即子任务1 输出）：
    --input  题目版 jsonl（出 patient[] 与 full_text[]，用来做患者名册与上下文/章节）
    --pred   子任务1 预测 jsonl（出 entities[]；若带 association 也能被门控式重算）
    --out    输出 jsonl：entities 原样 + association
  输出格式对齐 evaluate.py 的 patient_pheno_map 读法。

鲁棒性：某篇 LLM 输出整体无法解析 → 该篇回退就近归属（--no-fallback 可关）。
        **永不比现基线更差。**

GPU 纪律：本项目全程用 GPU。检测不到 CUDA 直接报错退出，绝不静默退回 CPU。
         Qwen3-8B bf16 ≈16GB，4090 的 23.5GB 放得下，**需整卡开机（非无卡模式）**。
         （注：单靠 --gate-only 组合两份已存在文件不需要 GPU；但只要做 LLM 判定就必须整卡。）

用法（项目根目录、GPU 正常开机）：
    # 冒烟：先跑 2 篇，肉眼核对 prompt/输出
    python scripts/s2_associate/associate_llm.py \
        --input data/split/dev_input.jsonl --pred pred_dev_sapbert.jsonl \
        --out pred_dev_llm.jsonl --limit-docs 2 --show-prompt

    # dev 全量、门控式（最终路线）：直接出 dev 门控预测 + 顺便打子任务2分数
    python scripts/s2_associate/associate_llm.py \
        --input data/split/dev_input.jsonl --pred pred_dev_sapbert.jsonl \
        --out pred_dev_gate.jsonl --gate-only --gold data/split/dev.jsonl

    # A 榜（--pred 换子任务1 输出即可）：整卡跑 LLM，再交 evaluate 记录
    python scripts/s2_associate/associate_llm.py \
        --input data/split/test_input.jsonl --pred pred_test_sapbert.jsonl \
        --out pred_test_gate.jsonl --gate-only
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

# 复用词典基线里已验证的工具：全局文本重建 / 患者锚点 / 就近归属(兜底)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "core"))
from baseline_dict import build_global, associate  # noqa: E402

MODEL_NAME = "Qwen/Qwen3-8B"
# 解析 LLM 每行 "序号: 患者ID/NONE"。允许 ':' 或 '.' 或 ')' 作分隔，容忍前导空白/编号符。
# ID 字符类含 '.'/':'：患者 ID 常为 "OII.1"/"OII:9"/"OCSA108.01" 这类代:序号记法，
# 早期只含 '-' 会把 "OII:9" 截成 "OII"→非法→误判 NONE 丢弃（dev/A 榜多篇受影响，2026-09-04 修）。
LINE_RE = re.compile(r"^\s*[#\-\*]?\s*(\d+)\s*[:\.\)]\s*([A-Za-z0-9_\.\:\-]+)")


# ----------------------------- 章节定位 -----------------------------

def section_index(full_text):
    """把 full_text 段落变成按 offset 排序的区间表 [(start, end, section_type)]。
       每段覆盖 [offset, offset+len(text))；用来查某表型 mention 落在哪个章节。"""
    idx = []
    for s in full_text or []:
        o = s.get("offset", 0)
        t = s.get("text", "")
        idx.append((o, o + len(t), s.get("section_type") or s.get("type") or "?"))
    idx.sort(key=lambda x: x[0])
    return idx


def section_of(offset, sec_idx):
    """二分/线性查 offset 落在哪个章节（段不重叠，线性足够快，篇内段数~几十）。"""
    for a, b, st in sec_idx:
        if a <= offset < b:
            return st
    return "?"


# ----------------------------- 患者名册 -----------------------------

def patient_roster(doc):
    """返回 [(patient_id, 代表描述)]，代表描述取该患者最长的 mention 文本（信息量最大）。"""
    out = []
    for p in doc.get("patient", []):
        ms = p.get("mention", [])
        rep = ""
        if ms:
            rep = max((m.get("text", "") for m in ms), key=len)
        out.append((p["patient_id"], rep))
    return out


def full_roster(doc):
    """整篇分配口径用：[(pid, rep, [(text, offset), ...])]，rep=最长 mention，
       并带该患者**全部** mention 的原文+offset，供 LLM 在整篇里精确定位每个个体。"""
    out = []
    for p in doc.get("patient", []):
        ms = p.get("mention", [])
        rep = max((m.get("text", "") for m in ms), key=len) if ms else ""
        mm = [(" ".join((m.get("text", "") or "").split()), m.get("offset", -1)) for m in ms]
        out.append((p["patient_id"], " ".join(rep.split()), mm))
    return out


# ----------------------------- Prompt 组装 -----------------------------

def build_prompt(doc, entities, global_text, sec_idx, window):
    """为一篇文档组 prompt。返回 (prompt_str, valid_pids:set)。
       表型清单里每个实体给：序号 + 所在章节 + mention 文本 + ±window 字符上下文。"""
    roster = patient_roster(doc)
    valid_pids = {pid for pid, _ in roster}

    lines = []
    lines.append(
        "You are a clinical NLP expert. A biomedical case report mentions one or more "
        "individuals (patients / relatives). Below is the roster of individuals, then a "
        "numbered list of phenotype mentions found in the text (each with its section and "
        "surrounding context).")
    lines.append("")
    lines.append("For EACH numbered phenotype, decide which individual it describes, and "
                 "answer with that individual's ID. If the phenotype does NOT belong to any "
                 "listed individual — e.g. it is general disease background, a definition, a "
                 "method, another cohort/reference, or the text says it is normal/absent/ruled "
                 "out for the patient — answer NONE.")
    lines.append("")
    lines.append("Individuals:")
    for pid, rep in roster:
        rep = " ".join(rep.split())
        lines.append('  %s = "%s"' % (pid, rep[:120]))
    lines.append("")
    lines.append("Phenotype mentions:")
    for i, e in enumerate(entities, 1):
        off = e.get("offset", 0)
        length = e.get("length", 0)
        st = section_of(off, sec_idx)
        left = max(0, off - window)
        right = min(len(global_text), off + length + window)
        ctx = " ".join(global_text[left:right].split())
        men = " ".join((e.get("text") or "").split())
        lines.append('  %d. [%s] "%s" | context: ...%s...' % (i, st, men, ctx))
    lines.append("")
    lines.append("Output EXACTLY one line per phenotype, in the form:")
    lines.append("<number>: <individual ID or NONE>")
    lines.append("No explanations, no extra text. Example:")
    lines.append("1: %s" % (roster[0][0] if roster else "O1"))
    lines.append("2: NONE")
    return "\n".join(lines), valid_pids


def build_prompt_whole(doc, entities, global_text, window):
    """整篇分配口径的 prompt：喂**整篇 full_text** + 患者名册(含全部 mention 原文+offset)，
       替代 build_prompt 的「±window 上下文 + 最长 mention」。返回 (prompt_str, valid_pids)。
       诊断实测(_route_llm.py，dev)：整篇+名册offset 让 LLM 全权路由 macF1 由旧崩盘(0.337)
       翻正到 0.483、Score 0.5869→0.6011；路由准确率 0.84≫就近 0.63。"""
    roster = full_roster(doc)
    valid_pids = {pid for pid, _, _ in roster}

    L = []
    L.append("You are a clinical NLP expert reading a FULL biomedical case report. "
             "The article often describes several individuals (patients and/or relatives).")
    L.append("")
    L.append("=== FULL ARTICLE TEXT (char offsets start at 0) ===")
    L.append(global_text)
    L.append("=== END ARTICLE ===")
    L.append("")
    L.append("Individuals described in this article (ID = how we label them; the quoted "
             "phrases are how they are referred to in the text, with char offset):")
    for pid, rep, mm in roster:
        refs = "; ".join('"%s"@%d' % (t, o) for t, o in mm[:6] if t)
        L.append('  %s  (e.g. "%s")  refs: %s' % (pid, rep[:80], refs))
    L.append("")
    L.append("Below is a numbered list of phenotype mentions found in the article "
             "(each with a short surrounding context to locate it).")
    L.append("For EACH, decide which individual it describes and answer with that "
             "individual's ID. If it belongs to NONE of the listed individuals "
             "(general disease background, a definition, a method, another cohort/"
             "reference, or the text says it is normal/absent/ruled-out for the "
             "patient), answer NONE.")
    L.append("")
    L.append("Phenotype mentions:")
    for i, e in enumerate(entities, 1):
        off = e.get("offset", 0)
        length = e.get("length", 0)
        left = max(0, off - window)
        right = min(len(global_text), off + length + window)
        ctx = " ".join(global_text[left:right].split())
        men = " ".join((e.get("text") or "").split())
        L.append('  %d. "%s" @char%d | context: ...%s...' % (i, men, off, ctx))
    L.append("")
    L.append("Output EXACTLY one line per phenotype, in order, in the form:")
    L.append("<number>: <individual ID or NONE>")
    L.append("No explanations, no extra text.")
    return "\n".join(L), valid_pids


# ----------------------------- 输出解析 + 聚合 -----------------------------

def parse_assignments(resp, n_ent, valid_pids):
    """解析 LLM 回复为 {序号(1基): patient_id}。NONE / 未知ID / 缺行 都不入表（即判 NONE）。
       返回 (assign:dict, n_parsed:int)。n_parsed=解析到的合法行数（判 fallback 用）。"""
    assign = {}
    n_parsed = 0
    for raw in resp.splitlines():
        m = LINE_RE.match(raw)
        if not m:
            continue
        idx = int(m.group(1))
        val = m.group(2)
        if idx < 1 or idx > n_ent:
            continue
        n_parsed += 1
        if val in valid_pids:               # 只有命中真实患者ID才算归属；NONE/编造ID → 丢弃
            assign[idx] = val
    return assign, n_parsed


def assoc_from_assignments(entities, assign, valid_pids):
    """按 {序号: pid} 把实体 identifier 聚到各患者，去重。返回 association 列表。"""
    bucket = {pid: [] for pid in valid_pids}
    for i, e in enumerate(entities, 1):
        pid = assign.get(i)
        if pid is None:
            continue                        # 判 NONE：不挂给任何人
        ident = e.get("identifier")
        if ident is None:
            continue
        if ident not in bucket[pid]:        # 按 patient_id 去重（与 baseline 口径一致）
            bucket[pid].append(ident)
    return [{"patient_id": pid, "phenotype": phs} for pid, phs in bucket.items()]


def llm_routing(entities, assign, valid_pids):
    """LLM 的「保留集」按患者聚合：{patient_id: set(identifier)}，只含被 LLM 判给该患者的。
       门控式用它做 keep/drop；全权归属则直接当 association。"""
    bucket = {pid: set() for pid in valid_pids}
    for i, e in enumerate(entities, 1):
        pid = assign.get(i)
        if pid is None:
            continue                        # 判 NONE：不保留
        ident = e.get("identifier")
        if ident is not None:
            bucket[pid].add(ident)
    return {pid: sorted(s) for pid, s in bucket.items()}


def gate_assoc(base_assoc, llm_route):
    """门控式关联 = 就近路由 ∩ LLM 保留集。对每患者求交集；某个患者不在 llm_route 里
       （LLM 没判给它任何东西）→ 该患者为空，绝不误把别人表型塞给它。"""
    out = []
    for a in base_assoc:
        pid = a["patient_id"]
        kept = set(llm_route.get(pid, []))
        out.append({"patient_id": pid, "phenotype": sorted(set(a["phenotype"]) & kept)})
    return out


def filter_assoc(doc, entities, assign, max_dist=None):
    """真·门控（过滤口径）= LLM 只决定 keep/drop，留下的实体全部路由还给就近。
       与 gate_assoc（交集口径）的区别：彻底丢弃 LLM「挂给哪个患者」的意见，只用它
       「没判 NONE」这一条 keep 信号。因此不会出现「就近路由到 P1、LLM 却说 P2 →
       交集为空 → 该实体双杀蒸发」——这个双杀专砍多患者篇里的小患者、系统性压低 macF1。
       assign 里有键(=LLM 判给了某真实患者) 即 keep；判 NONE/编造ID/漏行 → drop。
       max_dist：非 None 时再叠一道距离阈值，砍掉离所有患者都很远的背景 FP（见 associate）。"""
    kept = [e for i, e in enumerate(entities, 1) if assign.get(i) is not None]
    return associate(doc, kept, max_dist=max_dist)


# ----------------------------- 主流程 -----------------------------

def main():
    ap = argparse.ArgumentParser(description="Qwen3-8B 患者-表型归属后处理器")
    ap.add_argument("--input", required=True, help="题目版 jsonl（出 patient/full_text）")
    ap.add_argument("--pred", required=True, help="子任务1 预测 jsonl（出 entities）")
    ap.add_argument("--out", required=True, help="输出预测 jsonl")
    ap.add_argument("--model", default=MODEL_NAME, help="HuggingFace 模型 id")
    ap.add_argument("--gold", help="金标准 jsonl；传了就顺便打分")
    ap.add_argument("--context-window", type=int, default=120, help="mention 左右各取多少字符上下文")
    ap.add_argument("--max-new-tokens", type=int, default=0, help="生成上限；0=按实体数自动(n*12+128)")
    ap.add_argument("--think", action="store_true", help="开 Qwen3 思考模式（更准但慢很多、更耗卡时）")
    ap.add_argument("--gate-only", action="store_true",
                    help="门控式（推荐）：LLM 只做 keep/drop，路由还给就近 = 就近∩LLM保留集")
    ap.add_argument("--whole-article", action="store_true",
                    help="整篇分配口径：喂整篇全文 + 患者名册(含mention offset)，LLM 全权判"
                         "挂给谁/NONE（不走就近）。诊断实测 dev 0.5869→0.6011、macF1 0.437→0.483、"
                         "路由 0.84≫就近 0.63。与 --gate-only 互斥。默认关。")
    ap.add_argument("--filter-mode", action="store_true",
                    help="真门控（过滤口径）：LLM 只 keep/drop，留下的全交就近路由；"
                         "丢弃 LLM 路由意见、消除双杀 bug。须与 --gate-only 同开")
    ap.add_argument("--route-max-dist", type=int, default=0,
                    help="方法1：过滤口径下，表型离最近患者锚点 > 此字符距离则不挂(砍背景FP)；"
                         "0=不启用（dev 扫描最优 ~800）。须与 --filter-mode 同开")
    ap.add_argument("--no-fallback", action="store_true", help="解析失败也不回退就近归属（默认回退）")
    ap.add_argument("--limit-docs", type=int, default=0, help="只处理前 N 篇（冒烟用）；0=全量")
    ap.add_argument("--show-prompt", action="store_true", help="打印首篇 prompt 与回复，便于核对")
    args = ap.parse_args()

    if args.filter_mode and not args.gate_only:
        print("❌ --filter-mode 须与 --gate-only 同开（它是门控式的一种口径）。已中止。")
        sys.exit(2)
    if args.route_max_dist and not args.filter_mode:
        print("❌ --route-max-dist 距离阈值只在过滤口径下生效，须与 --filter-mode 同开。已中止。")
        sys.exit(2)
    if args.whole_article and args.gate_only:
        print("❌ --whole-article 与 --gate-only 互斥（一个是整篇全权分配、一个是门控式）。已中止。")
        sys.exit(2)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # —— GPU 硬检查：本项目全程用 GPU，检测不到就报错退出 ——
    if not torch.cuda.is_available():
        print("❌ 未检测到 GPU（torch.cuda.is_available()=False）。本项目全程用 GPU，已中止。")
        print("   先跑 `python scripts/gpu_check.py` 排查；本脚本需整卡开机（非无卡模式）。")
        sys.exit(2)
    device = torch.device("cuda:0")
    print("使用 GPU :", torch.cuda.get_device_name(0))
    if args.whole_article:
        print("模式     : 整篇分配（喂整篇全文+患者名册offset，LLM 全权判挂谁/NONE，不走就近）")
    if args.gate_only:
        if args.filter_mode:
            print("模式     : 真门控/过滤口径（LLM 只 keep/drop，留下的全交就近路由）")
        else:
            print("模式     : 门控式/交集口径（就近 ∩ LLM 保留集）")

    # 读两个文件并按 pmc_id join
    inputs = [json.loads(l) for l in open(args.input, encoding="utf-8") if l.strip()]
    preds = [json.loads(l) for l in open(args.pred, encoding="utf-8") if l.strip()]
    pred_by_pmc = {d["pmc_id"]: d for d in preds}
    if args.limit_docs:
        inputs = inputs[:args.limit_docs]
    print("输入 %d 篇；预测文件覆盖 %d 篇" % (len(inputs), len(pred_by_pmc)))

    # 加载 Qwen3-8B
    print("加载 %s ...（首次会下载约 16GB，先 `source /etc/network_turbo` 开学术加速）" % args.model)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16).to(device).eval()
    print("  思考模式:", "开" if args.think else "关（默认，贪心解码，可复现且快）")

    out_docs = []
    n_llm = n_fallback = 0
    n_none = n_assigned = 0
    n_truncated = n_shortfall = 0     # 顶满 max_new_tokens 的篇数 / 解析行数<实体数 的篇数
    t0 = time.perf_counter()

    for di, d in enumerate(inputs):
        pmc = d["pmc_id"]
        pred = pred_by_pmc.get(pmc)
        entities = (pred or {}).get("entities", []) if pred else []
        G = build_global(d.get("full_text", []))
        sec_idx = section_index(d.get("full_text", []))

        # 无患者 → association 空；无实体 → 各患者空
        roster_pids = [p["patient_id"] for p in d.get("patient", [])]
        if not roster_pids:
            assoc = []
        elif not entities:
            assoc = [{"patient_id": pid, "phenotype": []} for pid in roster_pids]
        else:
            if args.whole_article:
                prompt, valid_pids = build_prompt_whole(
                    d, entities, G, args.context_window)
            else:
                prompt, valid_pids = build_prompt(
                    d, entities, G, sec_idx, args.context_window)
            # 每行 "<序号>: <患者ID或NONE>"，多患者时患者ID(如 OII.1)更长，按每行 ~12 token
            # + 128 余量估。贪心解码遇 EOS 自然停，故对已完整输出零成本，只兜住长文档不被截断。
            mnt = args.max_new_tokens or (len(entities) * 12 + 128)

            messages = [{"role": "user", "content": prompt}]
            text = tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=args.think)
            enc = tok([text], return_tensors="pt").to(device)
            with torch.no_grad():
                gen = model.generate(
                    **enc, max_new_tokens=mnt, do_sample=False,
                    pad_token_id=tok.eos_token_id)
            n_new_tok = gen.shape[1] - enc.input_ids.shape[1]
            # 顶满预算 = 没遇 EOS 自然停 = 输出被硬截断（后段序号缺失→误判 NONE）
            truncated = n_new_tok >= mnt
            resp = tok.decode(gen[0][enc.input_ids.shape[1]:], skip_special_tokens=True)
            # 思考模式下剥掉 <think>...</think>，只留最终答案
            if "</think>" in resp:
                resp = resp.split("</think>", 1)[1]

            if args.show_prompt and di == 0:
                print("\n----- 首篇 PROMPT -----\n%s" % prompt)
                print("\n----- 首篇 回复 -----\n%s\n" % resp.strip())

            assign, n_parsed = parse_assignments(resp, len(entities), valid_pids)
            if n_parsed == 0 and not args.no_fallback:
                # 整体没解析出任何合法行 → 回退就近归属，绝不比现基线差
                assoc = associate(d, entities)
                n_fallback += 1
            else:
                n_llm += 1
                n_assigned += len(assign)
                n_none += (len(entities) - len(assign))
                if truncated:
                    n_truncated += 1
                if n_parsed < len(entities):    # 有序号没被判到（截断/漏行）→ 那些实体默认 NONE
                    n_shortfall += 1
                if args.gate_only:
                    if args.filter_mode:
                        # 真门控（过滤口径）：LLM 只 keep/drop，留下的全交就近路由，
                        # 丢弃 LLM「挂给谁」的意见 → 消除双杀 bug（就近≠LLM 时不再蒸发）
                        assoc = filter_assoc(d, entities, assign,
                                             max_dist=(args.route_max_dist or None))
                    else:
                        # 交集口径：就近路由(path) ∩ LLM 保留集(keep)，多患者不被误路由
                        route = llm_routing(entities, assign, valid_pids)
                        base = associate(d, entities)
                        assoc = gate_assoc(base, route)
                else:
                    assoc = assoc_from_assignments(entities, assign, valid_pids)

        out_docs.append({
            "pmc_id": pmc, "pmid": d.get("pmid"),
            "entities": entities, "association": assoc,
        })
        if (di + 1) % 5 == 0 or di + 1 == len(inputs):
            print("  已处理 %d/%d 篇   用时 %.0f 秒" % (di + 1, len(inputs), time.perf_counter() - t0))

    with open(args.out, "w", encoding="utf-8") as f:
        for d in out_docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print("\n已写出: %s" % args.out)
    print("  LLM 判定 %d 篇，回退就近 %d 篇" % (n_llm, n_fallback))
    if n_truncated or n_shortfall:
        print("  ⚠ 输出被截断(顶满max_new_tokens) %d 篇；解析行数<实体数(有缺行) %d 篇"
              % (n_truncated, n_shortfall))
        print("    → 缺行的实体被默认判 NONE，会伤召回。可加大 --max-new-tokens 再跑。")
    else:
        print("  ✓ 无截断、无缺行（所有实体都拿到了判定）")
    if n_assigned + n_none:
        print("  表型归属：挂给患者 %d，判 NONE(丢弃) %d，NONE 占比 %.1f%%"
              % (n_assigned, n_none, 100.0 * n_none / (n_assigned + n_none)))
    print("  全程耗时 %.1f 秒" % (time.perf_counter() - t0))

    # 顺便打分
    if args.gold:
        from evaluate import load_jsonl, evaluate
        gold_docs = load_jsonl(args.gold)
        print("\n===== 打分（门控式对照 sapbert_v1: micF1=0.4310 macF1=0.4030）=====")
        evaluate(gold_docs, out_docs, verbose=True)
        print("\n如满意，正式记录：")
        print("  python scripts/core/evaluate.py --gold %s --pred %s \\" % (args.gold, args.out))
        print("      --tag llm_gate_v1 --note \"门控式归属：LLM只做keep/drop、路由还给就近\"")


if __name__ == "__main__":
    main()
