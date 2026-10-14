"""Two AO queries on the exact same saved100 seed activation windows."""
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.reader_gepa_io import PilotIO,save
from scripts.reader_feedback_controls import P,OUT,STANCE_QUERY
from src.paper_baselines import digest


def main():
    directory=OUT/'ao_query_probe'
    io=PilotIO(P/'runtime.json')
    status=directory/'status.json'
    try:
        ids=json.loads((P/'native/cohort_request_ids.json').read_text())
        assert len(ids)==100
        native=[json.loads((P/'native/results'/(i+'.json')).read_text()) for i in ids]
        lookup={r['id']:r for r in json.loads(Path(io.c['bundle']).read_text())['rows']}
        protocol=dict(questions=100,original_query=io.c['ao_prompt'],stance_query=STANCE_QUERY,
            layer_zero_based=31,hf_hidden_state_index=32,source_position='question end, last up to20 question tokens',
            source_activation_reuse='Exact same saved tensor path for both queries; no new source generation.',
            stance_instruction='Read evidence of source acceptance/doubt/rejection; no objective factual judgment or presumed error.',
            no_gepa_selection_use=True)
        path=directory/'protocol.json'
        if path.exists():assert json.loads(path.read_text())==protocol
        else:save(path,protocol)
        jobs={}
        for n in native:
            ident=digest(['ao_stance_probe_v1',n['id'],STANCE_QUERY])
            job=dict(id=ident,native_id=n['id'],activation_path=n['ao_activation_path'],query=STANCE_QUERY)
            request=P/'readers/ao/requests'/(ident+'.json')
            if not request.exists():save(request,job)
            jobs[n['id']]=ident
        if any(not (P/'readers/ao/results'/(i+'.json')).exists() for i in jobs.values()):io.start_reader('ao')
        records=[];joined=[];started=time.monotonic()
        for n in native:
            ident=jobs[n['id']];result=P/'readers/ao/results'/(ident+'.json')
            while not result.exists():
                if io.reader_proc is not None and io.reader_proc.poll() is not None:raise RuntimeError('AO query reader exited')
                if time.monotonic()-started>3600:raise TimeoutError('AO query probe exceeded3600seconds')
                time.sleep(1)
            stance=json.loads(result.read_text());assert stance['query']==STANCE_QUERY
            neutral=io.reader('ao',n)
            row=lookup[n['row_id']]
            records.append(dict(row_id=n['row_id'],dataset=n['dataset'],label=row['label'],question=n['question'],
                actual_answer=n['answer'],source_native_id=n['id'],source_activation_path=n['ao_activation_path'],
                neutral=neutral,stance=stance))
            # Reuse the exact-quote-validated anonymous content auditor. The two
            # answers and activations are identical; only reader queries differ.
            joined.append(dict(id='ao_query/'+n['id'],dataset=n['dataset'],reader='ao',row_id=n['row_id'],
                label=row['label'],question=n['question'],candidate_key='query_control',starting_prompt=False,
                source_native_id=n['id'],seed_reader_explanation=neutral['explanation'],
                current_reader_explanation=stance['explanation'],seed_answer=n['answer'],current_answer=n['answer']))
            save(status,dict(state='running',complete=len(records),expected=100,updated_utc=time.time()))
        save(directory/'query_outputs.json',records)
        save(directory/'reader_trajectories.json',joined)
        save(status,dict(state='complete',complete=100,expected=100,updated_utc=time.time()))
    except Exception as e:
        save(status,dict(state='failed',error=repr(e),updated_utc=time.time()));raise
    finally:io.stop_reader()


if __name__=='__main__':main()
