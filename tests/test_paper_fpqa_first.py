from scripts.run_paper_baselines_fpqa_first import schedule


def test_every_fpqa_condition_precedes_every_qa_condition():
    splits = [{'dataset': 'cancer_myth', 'name': 'fold0'},
              {'dataset': 'crepe', 'name': 'official'}]
    methods = ['plain', 'balanced', 'probe_gate', 'gepa_both']
    jobs = list(schedule(splits, methods))
    assert len(jobs) == 2 * len(splits) * len(methods)
    assert [qa for qa, _, _ in jobs] == [False] * 8 + [True] * 8
    for qa in (False, True):
        assert [(split, method) for q, split, method in jobs if q == qa] == [
            (s, m) for m in methods for s in splits]
