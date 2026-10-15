"""Verifier-only GEPA protocol. Gold is used by the response judge, never inference."""
from collections import Counter
from copy import deepcopy

from src.fpqa_scope_factorial import grounded_verify_messages, response_messages
from src.fpqa_uncertainty import parse_verdict


OBJECTIVE = ('Optimize ONLY the verifier system prompt to improve final answers: correct '
             'actual false premises and answer FPQs, while answering NFPs without invented '
             'premise corrections. Maximize equally weighted FPQ and NFP mean Well score / 5.')
BACKGROUND = '''The extractor, grounded candidate lists, response generator instructions and response judge are frozen. Your prompt is used separately for each candidate claim, with the original question and its grounding. It must retain the supplied JSON output schema and may change verification instructions only. No retrieval is used.
Feedback includes the complete question, fixed claims, verifier outputs, final answer, response score and judge explanation. These are observations, not ground-truth causal assignments to the verifier. An extraction omission or a final-answer decision may also explain failure. Keep successful behavior on both question types. Do not memorize example-specific disease facts, names, answers or question IDs in the prompt; write general verification instructions. Do not instruct the verifier to game the judge or place answer-writing instructions in its reasons.
Training and validation each have equal numbers of FPQ and NFP, so the aggregate mean gives the two groups equal weight. The 0-5 Well score is an ordinal response rubric, not a probability or a direct accuracy label for each extracted claim. A FPQ can contain true claims; NFP labels do not establish the truth of every candidate. Gold annotations inform response scoring only. No hard attribution of failure to a stage is supplied. No inference-stage access to labels or annotations is allowed. Preserve exactly the JSON keys verdict, claim_relation, personal_report, reason, source_ids, and their original allowed values. Use [] for source_ids because no sources are supplied.'''


def demonstrations_for(question, examples):
    """Fixed leave-one-question-out demonstration policy, independent of labels."""
    norm = lambda x: ' '.join(x.casefold().split())
    return [e for e in examples if norm(e['question']) != norm(question)]


def validate_splits(train, val, test):
    sets = []
    for rows in (train, val, test):
        ids = {r['id'] for r in rows}
        if len(ids) != len(rows):
            raise ValueError('Duplicate ID within split')
        if not rows or any(r['label'] not in (0, 1) for r in rows):
            raise ValueError('Invalid or empty split')
        sets.append(ids)
    if any(sets[i] & sets[j] for i in range(3) for j in range(i)):
        raise ValueError('Question IDs overlap across splits')
    for rows in (train, val):
        counts = Counter(r['label'] for r in rows)
        if counts[0] != counts[1]:
            raise ValueError('This comparator uses equal groups; unequal groups require weighted aggregation')


def verifier_messages(prompt, question, grounded):
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError('Empty verifier prompt')
    messages = grounded_verify_messages(question, grounded)
    messages[0]['content'] = prompt
    return messages


def execute_pipeline(prompt, question, grounded, template, call):
    """Only observable input goes here. No label/reference accepted by this interface."""
    claims = [c['text'] for c in grounded]
    verdicts = [call(f'verify_{i:03d}', 'task', verifier_messages(prompt, question, c), parse_verdict)
                for i, c in enumerate(grounded)]
    messages = response_messages(question, claims, verdicts, template, enhanced=True)
    raw = call('answer', 'task', messages)
    answer = template.ResponseClass.model_validate_plain_text(raw).get()
    return {'claims': claims, 'grounded_claims': deepcopy(grounded), 'verdicts': verdicts,
            'raw_answer': raw, 'answer': answer}


def feedback(row, generated, score, explanation):
    return {'question': row['question'], 'question_type': 'fpq' if row['label'] else 'tpq',
            'claims': generated['claims'], 'grounded_claims': generated['grounded_claims'],
            'verifier_outputs': generated['verdicts'], 'answer': generated['answer'],
            'raw_score': score, 'normalized_score': score / 5.0,
            'score_explanation': explanation,
            'interpretation': 'End-to-end response score; no causal verifier-responsibility label.'}


def paired_metrics(seed, best):
    a, b = ({r['id']: r for r in rows} for rows in (seed, best))
    if set(a) != set(b):
        raise ValueError('Paired comparison needs the same IDs')
    return {name: {'n': sum(a[i]['label'] == label for i in a),
                   'gain': sum(a[i]['label'] == label and a[i]['score'] < 4 <= b[i]['score'] for i in a),
                   'loss': sum(a[i]['label'] == label and b[i]['score'] < 4 <= a[i]['score'] for i in a)}
            for label, name in [(1, 'fpq'), (0, 'nfp')]}
