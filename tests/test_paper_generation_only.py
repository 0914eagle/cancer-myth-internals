import pytest
from scripts.run_paper_fpqa_generation_only import generation_schedule


def test_provider_independent_queue_covers_both_sources_without_qa():
    splits=[{'dataset':'cancer_myth','name':'fold0'},{'dataset':'crepe','name':'official'}]
    jobs=list(generation_schedule(splits,['plain','cot','probe_gate']))
    assert len(jobs)==6
    assert all(qa is False for qa,_,_ in jobs)
    assert {s['dataset'] for _,s,_ in jobs}=={'cancer_myth','crepe'}


def test_reflection_methods_cannot_enter_generation_only_queue():
    with pytest.raises(ValueError,match='reflection'):
        list(generation_schedule([{}],['gepa_both']))
