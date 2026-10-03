"""Cross the stored K3 forced-choice results with doc 62's answer-behavior ledger (no model calls).

Question: among FPQ items where the model picks the correction when both statements are shown,
does the way the actual answer fails (misses target / endorses error / related only) differ from
items where the choice is unstable or wrong? This narrows the next intervention; it does not
label items as known/unknown.

  python scripts/k3_behavior_crosstab.py --labels $SUITE_DIR/knowledge/qwen25_7b/labels.jsonl \
      --out docs/reviews/k3_behavior_crosstab_2026-10-04
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.k3_behavior import (BUCKET_NAMES, BUCKETS, K3_GROUPS, K3_NAMES, chi2_independence,  # noqa: E402
                             crosstab, k3_group, permutation_p, transitions)

LEDGER = ROOT / "docs/reviews/integrated_failure_census_2026-10-02/all_7318_answer_index.csv"


def pct(a, b):
    return f"{a}/{b} ({a / b:.1%})" if b else f"{a}/0"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", required=True, help="knowledge/<name>/labels.jsonl from scripts/probe_knowledge.py")
    ap.add_argument("--ledger", default=str(LEDGER))
    ap.add_argument("--model", default="qwen25", help="ledger model key the K3 run belongs to")
    ap.add_argument("--score", default="original_score", choices=["original_score", "audit_overlay_score"])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    labels = {}
    with open(args.labels, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                labels[r["id"]] = r
    with open(args.ledger, encoding="utf-8") as f:
        ledger = list(csv.DictReader(f))
    fpq_ids = {r["id"] for r in ledger if r["model"] == args.model and r["dataset"] == "fpq"}
    if not fpq_ids:
        sys.exit(f"no FPQ rows for model {args.model!r} in ledger")
    matched = fpq_ids & set(labels)
    if len(matched) < 0.5 * len(fpq_ids):
        sys.exit(f"only {len(matched)}/{len(fpq_ids)} ledger FPQ ids found in labels: id schemes differ, stop")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    L = ["# K3 선택 과제 × 실제 답변 행동 교차 (새 모델 호출 없음)", "",
         "K3는 통념과 교정문을 **함께 보여 주고** 정확한 쪽을 고르는 과제다. 아래 집단 이름은 그 과제의 결과일 뿐이며",
         "\"안다/모른다\" 라벨이 아니다. 행동 분류는 62번 원장의 상호 배타 코드다. 점수 기준 Well≥4, 점수 열 `" + args.score + "`.", "",
         f"- 모델 `{args.model}`, 원장 FPQ 문항 {len(fpq_ids)}, labels에 있는 문항 {len(matched)}",
         f"- 입력 해시: labels {hashlib.sha256(Path(args.labels).read_bytes()).hexdigest()[:16]}, "
         f"ledger {hashlib.sha256(Path(args.ledger).read_bytes()).hexdigest()[:16]}", ""]
    summary = {}
    for side in ("plain", "alternative"):
        ct = crosstab(ledger, labels, args.model, side, score_key=args.score)
        method = next((r["method"] for r in ledger if r["model"] == args.model and r["side"] == side), side)
        L += [f"## {side} (`{method}`)", "",
              "| K3 결과 | 문항 | Well≥4 | Well<4 | " + " | ".join(f"<4 중 {BUCKET_NAMES[b]}" for b in BUCKETS) + " |",
              "|---|---:|---:|---:|" + "---:|" * len(BUCKETS)]
        for g in K3_GROUPS:
            c = ct[g]
            if not c["n"]:
                continue
            L.append(f"| {K3_NAMES[g]} | {c['n']} | {pct(c['high'], c['n'])} | {c['low']} | "
                     + " | ".join(pct(c["low_buckets"][b], c["low"]) for b in BUCKETS) + " |")
        a = [b for b, k in ct["stable_choice"]["low_buckets"].items() for _ in range(k)]
        b_ = [b for g in ("unsure", "wrong") for b, k in ct[g]["low_buckets"].items() for _ in range(k)]
        chi = chi2_independence([[ct[g]["low_buckets"][b] for b in BUCKETS] for g in ("stable_choice", "unsure", "wrong")])
        L.append("")
        if a and b_:
            stat, p = permutation_p(a, b_)
            L.append(f"Well<4 답변의 행동 분포, \"두 순서 모두 교정문 선택\" 대 \"불안정+통념 선택\": chi2 {stat:.2f}, "
                     f"순열 p {p:.3f} (n {len(a)} 대 {len(b_)}).")
        if chi:
            L.append(f"세 집단 chi2 {chi[0]:.2f}, 자유도 {chi[1]}, 최소 기대빈도 {chi[2]:.1f}"
                     + (" (5 미만이라 근사 부정확, 순열 p를 본다)" if chi[2] < 5 else "") + ".")
        L.append("")
        summary[side] = {g: {"n": c["n"], "high": c["high"], "low": c["low"], "low_buckets": dict(c["low_buckets"]),
                             "high_buckets": dict(c["high_buckets"])} for g, c in ct.items()}
    tr = transitions(ledger, labels, args.model, score_key=args.score)
    keys = ("low->low", "low->high", "high->low", "high->high")
    L += ["## Plain → 대안 조건 점수 이동", "", "| K3 결과 | " + " | ".join(keys) + " |", "|---|" + "---:|" * len(keys)]
    for g in K3_GROUPS:
        if sum(tr[g].values()):
            L.append(f"| {K3_NAMES[g]} | " + " | ".join(str(tr[g][k]) for k in keys) + " |")
    L += ["", "## 읽는 법", "",
          "- 집단 간 행동 분포가 다르면: 선택 과제 성공 여부가 원 질문에서 실패하는 **모습**과 연결된다. 원인 확정은 아니다.",
          "- 다르지 않으면: 선택 과제의 성공은 원 질문 실패의 모습을 구분하지 못한다. 이것도 결과다.",
          "- 대안 조건의 low->high가 한 집단에 몰리면, 그 조건이 구제하는 문항이 선택 과제 결과와 연결된다는 관찰이다.",
          "- 불안정·통념 선택 집단은 작다(문서 31: 44 + 5). 비율보다 건수를 함께 읽는다."]
    (out / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    (out / "summary.json").write_text(json.dumps({"model": args.model, "score": args.score, "matched": len(matched),
                                                  "ledger_fpq": len(fpq_ids), "sides": summary,
                                                  "transitions": {g: dict(v) for g, v in tr.items()}},
                                                 ensure_ascii=False, indent=1), encoding="utf-8")
    with (out / "items.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "side", "k3_group", "fc_margin", "score", "alignment"])
        for r in ledger:
            if r["model"] == args.model and r["dataset"] == "fpq":
                lab = labels.get(r["id"])
                w.writerow([r["id"], r["side"], k3_group(lab), (lab or {}).get("fc_margin"), r[args.score], r["alignment"]])
    print((out / "report.md").read_text())


if __name__ == "__main__":
    main()
