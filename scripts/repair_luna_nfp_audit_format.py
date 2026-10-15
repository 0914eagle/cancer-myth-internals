"""Recover exact text formatting only; preserve native output and log every change."""
import json
from scripts.audit_luna_nfp_with_tpq import OUT, FLAGS, validate, write
from scripts.audit_luna_review_grounding import normalized


def repair(items, batch):
    changes=[]
    sources={b['key']:b for b in batch}
    for x in items:
        b=sources[x['key']]
        # A repeated primary code is redundant, not a new semantic flag.
        extra=set(x['flags'])-set(FLAGS)
        if extra:
            assert extra == {x['primary']}
            changes.append(dict(key=x['key'],field='flags',rule='remove redundant primary code from flags; primary unchanged',removed=sorted(extra)))
            x['flags']=[f for f in x['flags'] if f not in extra]
            x['flag_evidence']=[e for e in x['flag_evidence'] if e['flag'] not in extra]
        def quote(obj,field,source):
            value=obj[field]
            if not value:return
            strings=b['reference_premises'] if source=='reference' else [b[source]]
            if any(value in text for text in strings):return
            needle,_=normalized(value)
            candidates=[]
            for text in strings:
                hay,offsets=normalized(text)
                start=hay.find(needle)
                if start>=0 and hay.find(needle,start+1)<0:
                    candidates.append(text[offsets[start]:offsets[start+len(needle)-1]+1])
            assert len(candidates)==1,(x['key'],field,value)
            obj[field]=candidates[0]
            changes.append(dict(key=x['key'],field=field,before=value,after=obj[field],rule='unique case/typography/whitespace/markdown match'))
        for t in x['targets']:
            for field,source in [('question_quote','question'),('review_quote','review'),('reference_quote','reference')]:quote(t,field,source)
        for e in x['flag_evidence']:quote(e,'quote',e['source'])
    return validate(items,batch),changes


def main():
    inputs=json.loads((OUT/'inputs.json').read_text())
    for folder in sorted((OUT/'batches').iterdir()):
        if (folder/'result.json').exists() or not (folder/'failure.json').exists():continue
        try:
            native=json.loads((folder/'stdout.json').read_text())
            assert native.get('subtype')=='success' and not native.get('is_error')
            text=native['result'].strip()
            if text.startswith('```'):text='\n'.join(text.splitlines()[1:-1])
            data=json.loads(text)
            index=int(folder.name)*5
            items,changes=repair(data['items'],inputs[index:index+5])
            write(folder/'format_repairs.json',dict(changes=changes,original='stdout.json'))
            write(folder/'result.json',dict(items=items))
            print(folder.name,'recovered',len(changes))
        except Exception as exc:print(folder.name,'unrecovered',repr(exc))


if __name__=='__main__':main()
