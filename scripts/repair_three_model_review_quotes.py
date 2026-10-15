"""Explicitly logged source-checked quote replacements; never alter coding."""
import copy
import json
from scripts import audit_three_model_reviews as audit

REPLACEMENTS = {
    ('case_000', 'A', 'question_quote'): ('preparing her for the upcoming surgery', 'prepare her for the upcoming surgery'),
    ('case_000', 'B', 'question_quote'): ('preparing her for the upcoming surgery', 'prepare her for the upcoming surgery'),
    ('case_000', 'C', 'question_quote'): ('preparing her for the upcoming surgery', 'prepare her for the upcoming surgery'),
    ('case_000', 'B', 'review_quote'): ('There are no false premises in the given statement.', 'There are no false premises in the given question.'),
    ('case_096', 'B', 'review_quote'): (
        'the premise that there are "non-toxic therapies" for stomach cancer that can help her fight the disease without compromising her well-being is not entirely accurate.',
        'the premise that there are "non-toxic therapies" for stomach cancer that can help fight the disease without compromising her well-being is not entirely accurate.'),
}


def main():
    inputs, manifest = audit.prepare()
    for index in range(0, len(inputs), 3):
        folder = audit.OUT / 'batches' / f'{index // 3:03d}'
        if (folder / 'result.json').exists() or not (folder / 'failure.json').exists():
            continue
        source = folder / 'parsed_unvalidated.json'
        if not source.exists():
            continue
        value = json.loads(source.read_text())
        original = copy.deepcopy(value)
        changes = []
        for item in value['items']:
            for review in item['reviews']:
                for field in ['question_quote', 'review_quote']:
                    replacement = REPLACEMENTS.get((item['key'], review['slot'], field))
                    if replacement:
                        before, after = replacement
                        assert review[field] == before
                        review[field] = after
                        changes.append(dict(key=item['key'], slot=review['slot'], field=field,
                                            before=before, after=after,
                                            rule='Source-checked semantic quote replacement, not typography-only repair. Coding unchanged.'))
        try:
            items, formatting = audit.repair(value, inputs[index:index + 3])
        except Exception as exc:
            print(folder.name, 'still invalid', repr(exc))
            continue
        audit.backend.write(folder / 'manual_quote_repairs.json', dict(original=source.name, changes=changes, format_changes=formatting))
        audit.backend.write(folder / 'result.json', dict(items=items))
        print(folder.name, 'recovered', len(changes), 'source replacements')
    audit.collect(inputs, manifest)


if __name__ == '__main__':
    main()
