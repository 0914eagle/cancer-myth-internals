"""Reuse pinned Well pipeline template bodies and parsers, without GPU/API imports.

Only named template classes and the feedback constant are executed from the AST.
No prompt text, few-shot example, or parser is rewritten here.
"""
import ast
import importlib
from pathlib import Path
from types import SimpleNamespace


def load_templates(root):
    response = importlib.import_module("response")
    result = {}
    selections = {
        "prewome": ("run_prewome.py", {
            "PresuppositionExtractionTemplate", "FeedbackActionTemplate", "FinalAnswerTemplate"}),
        "atomic": ("run_presupposition_pipeline.py", {
            "PresuppositionExtractionTemplate", "LLMCheckTemplate", "FactCheckFinalAnswerTemplate"}),
    }
    for method, (filename, names) in selections.items():
        path = Path(root) / "prompting" / filename
        tree = ast.parse(path.read_text(), filename=str(path))
        nodes = [node for node in tree.body if
                 (isinstance(node, ast.ClassDef) and node.name in names) or
                 (isinstance(node, ast.Assign) and any(
                     isinstance(t, ast.Name) and t.id == "DEFAULT_FEEDBACK_ACTION_SYSTEM_PROMPT"
                     for t in node.targets))]
        if {n.name for n in nodes if isinstance(n, ast.ClassDef)} != names:
            raise ValueError("Upstream template definitions changed")
        namespace = dict(vars(response))
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
        result[method] = SimpleNamespace(**{name: namespace[name] for name in names})
    return result


def extract(question, few_shot, templates, call):
    inputs = dict(question=question, few_shot_data=few_shot, passages=[])
    a = templates["prewome"].PresuppositionExtractionTemplate(**inputs)
    b = templates["atomic"].PresuppositionExtractionTemplate(**inputs)
    if a.generate() != b.generate():
        raise ValueError("Extraction messages differ; cannot share the extraction")
    text = call("extract", "task", a.generate())
    return a.ResponseClass.model_validate_plain_text(text).get()


def answer(question, few_shot, claims, method, templates, call):
    """Only the question and generated claims enter inference; no test annotation."""
    t = templates[method]
    if method == "prewome":
        review = t.FeedbackActionTemplate(question=question, few_shot_data=few_shot,
                                         model_detected_presuppositions=claims, passages=[])
        raw = call("prewome/review", "task", review.generate())
        feedback = review.ResponseClass.model_validate_plain_text(raw).get()
        final = t.FinalAnswerTemplate(question=question, model_feedback_action=feedback,
                                     few_shot_data=few_shot)
        intermediate = {"feedback_action": feedback}
    elif method == "atomic":
        results, raw_checks, invalid = [], [], []
        import re
        for i, claim in enumerate(claims):
            check = t.LLMCheckTemplate(model_detected_presupposition=claim,
                                      passages=[], few_shot_data=few_shot)
            raw = call(f"atomic/check_{i:03d}", "task", check.generate())
            results.append(check.ResponseClass.model_validate_plain_text(raw).get())
            raw_checks.append(raw)
            # Preserve the official parser's fallback, but expose it in results.
            if not re.search(r"\b(yes|true|no|false)\b", raw.lower()):
                invalid.append(i)
        final = t.FactCheckFinalAnswerTemplate(question=question, factcheck_results=results,
                    model_detected_presuppositions=claims, few_shot_data=few_shot)
        intermediate = {"factcheck_results": results, "raw_checks": raw_checks,
                        "unparseable_check_indices": invalid}
    else:
        raise ValueError(method)
    raw = call(f"{method}/answer", "task", final.generate())
    return {"raw_answer": raw,
            "answer": final.ResponseClass.model_validate_plain_text(raw).get(),
            **intermediate}
