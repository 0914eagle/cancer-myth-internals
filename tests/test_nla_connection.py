from scripts.nla_connection import select_rows


def test_pilot_selects_only_train_and_does_not_copy_gold():
    rows=[];splits=[]
    for dataset,name in [('cancer_myth','fold0'),('crepe','official')]:
        train=[]
        for label in (0,1):
            for partition in ('train','test','dev'):
                id=f'{dataset}-{partition}-{label}'
                rows.append(dict(id=id,dataset=dataset,label=label,question='Question?',reference_answer='SECRET'))
                if partition=='train':train.append(id)
        splits.append(dict(dataset=dataset,name=name,train=train,
            test=[f'{dataset}-test-0',f'{dataset}-test-1'],dev=[f'{dataset}-dev-0',f'{dataset}-dev-1']))
    selected=select_rows(dict(rows=rows,splits=splits))
    assert len(selected)==4
    assert all('-train-' in r['id'] for r in selected)
    assert {(r['dataset'],r['label']) for r in selected}=={(d,l) for d in ['cancer_myth','crepe'] for l in [0,1]}
    assert all('reference_answer' not in r for r in selected)


def test_leaking_split_is_rejected():
    import pytest
    with pytest.raises(AssertionError):
        select_rows(dict(rows=[],splits=[dict(dataset='cancer_myth',name='fold0',train=['x'],dev=['x'],test=[])]))
