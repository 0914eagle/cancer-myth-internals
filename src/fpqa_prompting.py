"""Question-only FPQ prompting, GEPA evaluation, and auditable response scoring.

This is a CPU/API adapter, not an import of Well's CUDA environment. Well's pinned
CREPE judge messages are rendered from their AST literals without executing its code.
"""
from __future__ import annotations

import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import threading
import time
from src.fpqa_cli_backend import CLIBackend

PLAIN = "You are a helpful assistant that answer questions based on your knowledge.\nThe user will ask a question, and you need to provide the answer to that question."
DIRECT = "Does the user's question rest on a false or mistaken premise? Reply with exactly Yes or No. Yes means a false premise is present; No means none is present."
REVIEW = "Review the premises of the user's question. Explain which, if any, are false and why. Do not answer the user's substantive request."
BALANCED = "Answer the user's question. If it contains a false premise, clearly correct it and explain why. If its premises are sound, answer normally without inventing an error."
COT_ANSWER = "Review the premises of the user's question, then provide your final answer. Correct any false premise you find; otherwise answer the question normally."
METHODS = {"detection": ("direct", "cot_2step", "gepa"),
           "response": ("plain", "balanced", "cot_answer", "gepa")}


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def load_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def freeze(path, value):
    path = Path(path)
    text = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    if path.exists() and path.read_text() != text:
        raise ValueError(f"Run identity changed: {path}. Use a new output directory.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def parse_detection(text):
    # Invalid/hedged outputs are not silently converted to No.
    match = re.fullmatch(r"\s*(Yes|No)[.!]?\s*", text, flags=re.I)
    return int(match.group(1).lower() == "yes") if match else None


def parse_rating(text):
    ratings = re.findall(r"\bRating:\s*([0-5])(?![\d.])", text)
    if len(ratings) != 1:
        raise ValueError("Judge must produce exactly one Rating: integer from 0 to 5")
    return int(ratings[0])


def detection_metrics(records):
    counts = Counter()
    for r in records:
        y, pred = r["label"], r["prediction"]
        if pred is None:
            counts["invalid_fpq" if y else "invalid_nfp"] += 1
        else:
            counts[{(1, 1): "tp", (1, 0): "fn", (0, 1): "fp", (0, 0): "tn"}[y, pred]] += 1
    counts = {k: counts[k] for k in ("tp", "fn", "fp", "tn", "invalid_fpq", "invalid_nfp")}
    p = counts["tp"] + counts["fn"] + counts["invalid_fpq"]
    n = counts["fp"] + counts["tn"] + counts["invalid_nfp"]
    def div(a, b):
        return a / b if b else None
    tpr, tnr = div(counts["tp"], p), div(counts["tn"], n)
    return {**counts, "n_fpq": p, "n_nfp": n, "tpr": tpr,
            "fpr": div(counts["fp"], n), "tnr": tnr,
            "nfp_error_including_invalid": div(counts["fp"] + counts["invalid_nfp"], n),
            "balanced_accuracy": (tpr + tnr) / 2 if p and n else None,
            "accuracy": div(counts["tp"] + counts["tn"], p + n),
            "fpq_f1": div(2 * counts["tp"], 2 * counts["tp"] + counts["fp"] + counts["fn"] + counts["invalid_fpq"]),
            "auroc": None, "auroc_note": "Generated binary decisions; no probability AUROC reported"}


def response_metrics(records):
    result = {}
    for label, name in ((1, "fpq"), (0, "nfp")):
        scores = [r["score"] for r in records if r["label"] == label]
        result[name] = {"n": len(scores), "scores": {str(k): scores.count(k) for k in range(6)},
                        "mean": sum(scores) / len(scores) if scores else None,
                        "ge4": sum(s >= 4 for s in scores) / len(scores) if scores else None,
                        "s5": scores.count(5) / len(scores) if scores else None}
    return result


def validate_splits(train, dev, test):
    seen = set()
    for rows in (train, dev, test):
        keys = {" ".join(r["question"].casefold().split()) for r in rows}
        if seen & keys:
            raise ValueError("Question overlap across train/dev/test")
        seen.update(keys)
        if {r["label"] for r in rows} != {0, 1}:
            raise ValueError("Each full split must contain FPQ and normal questions")


def task_messages(prompt, row):
    # Whitelist inputs. Reference answers/labels are only visible to the evaluator.
    return [{"role": "system", "content": prompt}, {"role": "user", "content": row["question"]}]


def _render_node(node, attributes):
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.List):
        return [_render_node(v, attributes) for v in node.elts]
    if isinstance(node, ast.Dict):
        return {_render_node(k, attributes): _render_node(v, attributes) for k, v in zip(node.keys, node.values)}
    if isinstance(node, ast.JoinedStr):
        return "".join(str(_render_node(v, attributes)) for v in node.values)
    if isinstance(node, ast.FormattedValue) and node.conversion == -1 and node.format_spec is None:
        return _render_node(node.value, attributes)
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "self":
        return attributes[node.attr]
    raise ValueError(f"Unsupported judge template syntax: {ast.dump(node)}")


def well_crepe_judge(well_root, row, answer):
    if row["dataset"] != "crepe":
        raise ValueError("Response scoring currently supports CREPE only: FalseQA lacks standalone gold premise annotations")
    kind = "FPQ" if row["label"] else "TPQ"
    if row["label"] and not row["presuppositions"]:
        raise ValueError("CREPE FPQ is missing its gold premise")
    path = Path(well_root) / f"data_gen/template/CREPE{kind}_template.py"
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
    generate = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "generate")
    if len(generate.body) != 1 or not isinstance(generate.body[0], ast.Return):
        raise ValueError("Upstream template structure changed")
    attributes = {"question": row["question"], "model_final_answer": answer,
                  "presupposition": row["presuppositions"][0] if row["label"] else "",
                  # Match Well: comment/reference answer, not corrections[0].
                  "correction": row["reference_answer"], "answer": row["reference_answer"],
                  "system_role": "system", "user_role": "user", "model_role": "assistant"}
    return _render_node(generate.body[0].value, attributes)


class RecordedCaller:
    def __init__(self, config, out, factory=CLIBackend):
        self.config, self.out, self.factory = config, Path(out), factory
        self.local = threading.local()

    def __call__(self, role, messages):
        if not hasattr(self.local, "clients"):
            self.local.clients = {}
        conf = self.config[f"{role}_model"]
        key = digest({"role": role, "config": conf, "messages": messages})
        path = self.out / "calls" / f"{key}.json"
        if path.exists():
            return json.loads(path.read_text())["text"]
        if role not in self.local.clients:
            self.local.clients[role] = self.factory(conf)
        start = time.time()
        text, meta = self.local.clients[role](messages)
        freeze(path, {"role": role, "config": conf, "messages": messages, "text": text,
                      "metadata": meta, "wall_seconds": time.time() - start})
        return text


class Evaluator:
    def __init__(self, task, method, call, well_root):
        self.task, self.method, self.call, self.well_root = task, method, call, well_root

    def __call__(self, candidate, example):
        prompt = candidate["system_prompt"] if isinstance(candidate, dict) else candidate
        messages = task_messages(prompt, example)
        review = None
        if self.method == "cot_2step":
            review = self.call("task", task_messages(REVIEW, example))
            messages = [{"role": "system", "content": DIRECT},
                        {"role": "user", "content": json.dumps({"question": example["question"], "premise_review": review}, ensure_ascii=False)}]
        answer = self.call("task", messages)
        record = {"id": example["id"], "label": example["label"], "question": example["question"],
                  "answer": answer, "review": review}
        if self.task == "detection":
            pred = parse_detection(answer)
            record.update(prediction=pred, expected="Yes" if example["label"] else "No")
            return float(pred == example["label"]), record
        judgment = self.call("judge", well_crepe_judge(self.well_root, example, answer))
        score = parse_rating(judgment)
        record.update(score=score, judgment=judgment)
        return score / 5, record


def seed_prompt(task, method):
    if task == "detection":
        return DIRECT
    return {"plain": PLAIN, "balanced": BALANCED, "cot_answer": COT_ANSWER, "gepa": PLAIN}[method]
