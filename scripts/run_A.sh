#!/usr/bin/env bash
# ============================================================================
# run_A.sh —— A 榜（/B 榜）一键推理链路：SapBERT 识别 → Qwen3-8B 门控式归属。
#
# 做什么：把跑榜的三件事串起来，全程 GPU（整卡开机），产出可直接提交的 jsonl。
#   1) 子任务1：baseline_sapbert.py 在题目版输入上识别表型（阈值 0.95，与 sapbert_v1 一致）
#   2) 子任务2：associate_llm.py --gate-only 门控式归属（LLM 只做 keep/drop，路由还给就近）
#   3) 提交前自检：篇数、pmc 顺序、entities 字段齐全
#
# 为什么要它：手敲三条命令+一堆环境变量易错；且学术加速代理常握手超时，
#            这里统一切到 hf-mirror 自建镜像，避开 xet CAS 主机。
#
# 用法（云端整卡开机、项目根目录或任意目录均可）：
#     bash scripts/run_A.sh                 # 默认跑 A 榜
#     bash scripts/run_A.sh <输入jsonl> <hp.obo> <前缀>   # B 榜/自定义
#   例：B 榜释放后
#     bash scripts/run_A.sh data/PatientPheX-V1-B/PatientPheX-B.jsonl \
#                           data/PatientPheX-V1-B/hp.obo  B
#
# 可调环境变量：
#     SIM_THRESHOLD=0.95   SapBERT cosine 阈值（默认 0.95，务必与线上基线一致）
#     INDEX_DIR=outputs/sapbert   概念向量索引目录（需已 build_concept_index）
#
# 产出：pred_<前缀>_sapbert.jsonl（识别结果）、pred_<前缀>_gate.jsonl（提交文件）
#
# 注意：本脚本不 commit、不提交天池——只产出文件，提交与否由你决定。
#      dev 复现（带打分）请单独跑 associate_llm.py --gate-only --gold（见其 docstring）。
# ============================================================================
set -euo pipefail

# —— 定位项目根：脚本在 scripts/ 下，根目录是其上一级 ——
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# —— 参数（默认 A 榜；B 榜/自定义可覆盖）——
INPUT="${1:-data/PatientPheX-V1-A/PatientPheX-A.jsonl}"
OBO="${2:-data/PatientPheX-V1-A/hp.obo}"
PREFIX="${3:-A}"
SIM_THRESHOLD="${SIM_THRESHOLD:-0.95}"
INDEX_DIR="${INDEX_DIR:-outputs/sapbert}"

PRED_SAPBERT="pred_${PREFIX}_sapbert.jsonl"
PRED_GATE="pred_${PREFIX}_gate.jsonl"

# —— 环境修复：镜像非法 OMP 值 + 学术代理易握手超时，统一切 hf-mirror ——
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY 2>/dev/null || true
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"   # 自建全量镜像，稳
export HF_HUB_DISABLE_XET=1                                  # 经典下载通道，避开 xet CAS(401/超时)

echo "==================== run_A.sh ===================="
echo "  项目根   : $ROOT"
echo "  输入     : $INPUT"
echo "  hp.obo   : $OBO"
echo "  前缀     : $PREFIX  →  $PRED_SAPBERT / $PRED_GATE"
echo "  SapBERT阈值: $SIM_THRESHOLD   索引: $INDEX_DIR"
echo "  HF_ENDPOINT: $HF_ENDPOINT"
echo "=================================================="

# —— 前置检查 1：输入/obo 存在 ——
for f in "$INPUT" "$OBO"; do
  [ -f "$f" ] || { echo "❌ 找不到文件: $f"; exit 1; }
done

# —— 前置检查 2：GPU 可用（本项目全程 GPU，绝不退 CPU）——
python - <<'PY'
import sys
try:
    import torch
except Exception as e:
    print("❌ 无法 import torch:", e); sys.exit(2)
if not torch.cuda.is_available():
    print("❌ 未检测到 GPU。本项目全程用 GPU，请整卡开机（非无卡模式）后重试。")
    sys.exit(2)
print("✓ GPU:", torch.cuda.get_device_name(0))
PY

# —— 前置检查 3：SapBERT 概念索引已构建 ——
if [ ! -d "$INDEX_DIR" ] || [ -z "$(ls -A "$INDEX_DIR" 2>/dev/null || true)" ]; then
  echo "❌ 概念索引缺失: $INDEX_DIR"
  echo "   先构建（一次性，整卡）："
  echo "     python scripts/s1_identify/build_concept_index.py --obo \"$OBO\" --out-dir \"$INDEX_DIR\" --fp16"
  exit 1
fi

# —— 步骤 1：子任务1 SapBERT 识别 ——
echo ""
echo ">>> [1/3] 子任务1：SapBERT 识别 → $PRED_SAPBERT"
python scripts/s1_identify/baseline_sapbert.py \
    --input "$INPUT" \
    --obo   "$OBO" \
    --index-dir "$INDEX_DIR" \
    --out   "$PRED_SAPBERT" \
    --sim-threshold "$SIM_THRESHOLD" \
    --fp16

# —— 步骤 2：子任务2 门控式归属 ——
echo ""
echo ">>> [2/3] 子任务2：Qwen3-8B 门控式归属 → $PRED_GATE"
python scripts/s2_associate/associate_llm.py \
    --input "$INPUT" \
    --pred  "$PRED_SAPBERT" \
    --out   "$PRED_GATE" \
    --gate-only

# —— 步骤 3：提交前自检 ——
echo ""
echo ">>> [3/3] 提交前自检"
INPUT="$INPUT" PRED_GATE="$PRED_GATE" python - <<'PY'
import json, os, sys
inp, sub_path = os.environ["INPUT"], os.environ["PRED_GATE"]
ref = [json.loads(l) for l in open(inp, encoding="utf-8") if l.strip()]
sub = [json.loads(l) for l in open(sub_path, encoding="utf-8") if l.strip()]
ok = True
if len(sub) != len(ref):
    print("❌ 篇数不一致: 提交 %d vs 输入 %d" % (len(sub), len(ref))); ok = False
if [d["pmc_id"] for d in sub] != [d["pmc_id"] for d in ref]:
    print("❌ pmc_id 顺序/集合与输入不一致（须按输入字典序）"); ok = False
need_e = {"identifier", "type", "offset", "length", "text", "note"}
bad = [d["pmc_id"] for d in sub for e in (d.get("entities") or []) if not need_e <= set(e)]
if bad:
    print("❌ 这些篇的 entities 缺字段:", bad[:5]); ok = False
ne = sum(len(d.get("entities") or []) for d in sub)
na = sum(len(a.get("phenotype") or []) for d in sub for a in (d.get("association") or []))
print("  篇数=%d  entities总数=%d  关联表型总数=%d" % (len(sub), ne, na))
if not ok:
    sys.exit(1)
print("  ✓ 自检通过")
PY

echo ""
echo "=================================================="
echo "✅ 完成。提交文件: $PRED_GATE"
echo "   1) 到天池网页提交 $PRED_GATE"
echo "   2) 拿到线上分后记录（无卡模式即可）："
echo "      python scripts/track/log_online.py --tag llm_gate_v1 --board $PREFIX \\"
echo "          --score <线上Score> --men <..> --doc <..> --mic <..> --mac <..> \\"
echo "          --submit-ts \"<天池提交时间>\" --note \"SapBERT@$SIM_THRESHOLD + Qwen3-8B 门控式归属\""
echo "      python scripts/track/report.py"
echo "=================================================="
