from copy import deepcopy
import pytest
from scripts.audit_luna_nfp_with_tpq import validate
from scripts.repair_luna_nfp_audit_format import repair


def sample():
    source = dict(key='case_0', question='A may occur.', reference_premises=['A may occur.'],
                  review='A does not always occur.')
    item = dict(key='case_0', primary='question_distortion', reference_alignment='faithful',
        review_vs_reference='review_stronger', reference_alignment_reason_ko='범위 일치',
        explanation_ko='가능성을 필연성으로 바꾸었다.', alternative_reading_ko='추가 문맥은 없다.', confidence='medium',
        targets=[dict(grounding='unsupported', question_quote='A may occur.',
                      review_quote='A does not always occur.', reference_quote='A may occur.')],
        flags=['scope_expansion'], flag_evidence=[dict(flag='scope_expansion',source='review',quote='A does not always occur.')])
    return source,item


def test_quotes_and_keys():
    source,item=sample()
    assert validate([item],[source]) == [item]
    bad=deepcopy(item);bad['targets'][0]['review_quote']='A never occurs.'
    with pytest.raises(AssertionError): validate([bad],[source])
    with pytest.raises(AssertionError): validate([item,item],[source])


def test_missing_reference_not_filled_in():
    source,item=sample();source['reference_premises']=[]
    with pytest.raises(AssertionError): validate([item],[source])
    item['reference_alignment']=item['review_vs_reference']='missing'
    item['targets'][0]['reference_quote']=''
    assert validate([item],[source])


def test_every_flag_requires_original_evidence():
    source,item=sample();item['flags'].append('uncertainty_as_falsity')
    with pytest.raises(AssertionError): validate([item],[source])


def test_format_repair_preserves_words_and_rejects_paraphrase():
    source,item=sample();source['review']='**A** does not always occur.'
    fixed,changes=repair([item],[source])
    assert changes and fixed[0]['targets'][0]['review_quote'] in source['review']
    source,item=sample();item['targets'][0]['review_quote']='A never occurs.'
    with pytest.raises(AssertionError):repair([item],[source])
