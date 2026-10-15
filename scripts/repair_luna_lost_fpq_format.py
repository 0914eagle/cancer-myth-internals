"""Recover only unique typographic quote matches; retain the original response."""
import json

from scripts.audit_luna_lost_fpq import OUT, validate
from scripts.audit_luna_review_grounding import normalized


def main():
    inputs = json.loads((OUT / 'inputs.json').read_text())
    for folder in sorted((OUT / 'batches').iterdir()):
        original = folder / 'parsed_unvalidated.json'
        if (folder / 'result.json').exists() or not (folder / 'failure.json').exists() or not original.exists():
            continue
        index = int(folder.name) * 5
        batch = inputs[index:index + 5]
        sources = {x['key']: x for x in batch}
        value = json.loads(original.read_text())
        changes = []
        try:
            for item in value['items']:
                for field, source in [('question_quote', 'question'), ('correction_quote', 'reference_correction'), ('review_quote', 'review')]:
                    quote = item[field]
                    text = sources[item['key']][source]
                    if quote in text:
                        continue
                    needle, _ = normalized(quote)
                    haystack, offsets = normalized(text)
                    start = haystack.find(needle)
                    assert needle and start >= 0 and haystack.find(needle, start + 1) < 0, (item['key'], field, quote)
                    replacement = text[offsets[start]:offsets[start + len(needle) - 1] + 1]
                    item[field] = replacement
                    changes.append({'key': item['key'], 'field': field, 'before': quote, 'after': replacement,
                                    'rule': 'unique case/typography/whitespace/markdown match'})
            validated = validate(value['items'], batch)
            (folder / 'format_repairs.json').write_text(json.dumps({'original': original.name, 'changes': changes}, ensure_ascii=False, indent=2) + '\n')
            (folder / 'result.json').write_text(json.dumps({'items': validated}, ensure_ascii=False, indent=2) + '\n')
            print(folder.name, 'recovered', len(changes))
        except (AssertionError, KeyError) as exc:
            print(folder.name, 'unrecovered', repr(exc))


if __name__ == '__main__':
    main()
