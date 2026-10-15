"""Complete only two omitted AI reading codes; preserve original batch output."""
import json
from scripts import audit_three_model_reviews as audit


def main():
    inputs, manifest = audit.prepare()
    batch = inputs[192:195]
    folder = audit.OUT / 'batches/064'
    raw = json.loads((folder / 'parsed_unvalidated.json').read_text())
    item = next(x for x in raw['items'] if x['key'] == 'case_192')
    missing = [r['slot'] for r in item['reviews'] if 'reading' not in r]
    assert set(missing) == {'B', 'C'}
    source = next(x for x in batch if x['key'] == 'case_192')
    request = dict(key=source['key'], question=source['question'], reference=source['reference'],
                   reviews=[r for r in source['reviews'] if r['slot'] in missing])
    prompt = '''Two descriptive reading codes were omitted from an earlier audit. Independently code only these supplied reviews against the question. No model identities, dataset labels or gate decisions are supplied. Reference annotations are imperfect, not clinical truth. No external clinical facts or hidden-cause claims. Classify the main challenged/accepted target of each review with one reading code:
faithful = target scope supported by question;
stronger_or_different = adds a stronger or different proposition;
individual_context_reopened = reopens a stated individual prognosis/diagnosis/plan, whether justified or not;
unclear = cannot tell.
Preserve individual vs general, possible vs necessary, attribution, questions vs assertions. Mere extra advice need not be a challenge. Give one dominant description, not a causal diagnosis.
Return only {"key":"case_192","readings":[{"slot":"B","reading":code},{"slot":"C","reading":code}]}.
INPUT:
''' + json.dumps(request, ensure_ascii=False)
    completion = folder / 'schema_completion'
    result_path = completion / 'parsed.json'
    if result_path.exists():
        result = json.loads(result_path.read_text())
    else:
        assert not (completion / 'started.json').exists()
        audit.backend.write(completion / 'started.json', dict(purpose='Two omitted reading codes only'))
        text = audit.backend.call(prompt, completion)
        if text.startswith('```'):
            text = '\n'.join(text.splitlines()[1:-1])
        result = json.loads(text)
        audit.backend.write(result_path, result)
    assert result['key'] == 'case_192' and len(result['readings']) == 2
    values = {r['slot']: r['reading'] for r in result['readings']}
    assert set(values) == set(missing) and all(v in audit.READING for v in values.values())
    for r in item['reviews']:
        if r['slot'] in missing:
            r['reading'] = values[r['slot']]
    items, formatting = audit.repair(raw, batch)
    audit.backend.write(folder / 'schema_repairs.json', dict(original='parsed_unvalidated.json',
                        added_readings=values, source='schema_completion/parsed.json', format_changes=formatting,
                        rule='Separate blinded Sonnet call fills omitted fields only; all existing codes preserved'))
    audit.backend.write(folder / 'result.json', dict(items=items))
    print(json.dumps(values))


if __name__ == '__main__':
    main()
