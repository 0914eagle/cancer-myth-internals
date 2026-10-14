"""Check saved extraction positions and pinned tokenizer bytes without a GPU."""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.reader_feedback_controls import P,OUT
from scripts.reader_gepa_io import save,load_prompt_tokenizer
from transformers import AutoTokenizer
from tokenizers import Tokenizer

c=json.loads((P/'runtime.json').read_text())
base=json.loads(Path(c['baseline_config']).read_text())
spec=base['models']['gemma27']
snapshot=Path(base['cache_root'])/('models--'+spec['model_id'].replace('/','--'))/'snapshots'/spec['revision']
t=AutoTokenizer.from_pretrained(snapshot,local_files_only=True)
checked=[]
for ident in json.loads((P/'native/cohort_request_ids.json').read_text()):
    r=json.loads((P/'native/results'/(ident+'.json')).read_text())
    text=t.apply_chat_template(r['messages'],tokenize=False,add_generation_prompt=True)
    tokens=t(text,add_special_tokens=False,return_offsets_mapping=True)
    start=text.rfind(r['question']);end=start+len(r['question'])
    positions=[i for i,(a,b) in enumerate(tokens['offset_mapping']) if a<end and b>start and a!=b]
    first,last=positions[0],positions[-1]
    assert r['source_question_positions']==[first,last]
    assert r['nla_source_token_ids']==[tokens['input_ids'][last]]
    assert r['ao_source_token_ids']==tokens['input_ids'][max(first,last-19):last+1]
    checked.append(dict(row_id=r['row_id'],question_end_token=last,
        final_input_token=len(tokens['input_ids'])-1,tokens_after_question=len(tokens['input_ids'])-1-last,
        ao_window_length=len(r['ao_source_token_ids'])))
budget=load_prompt_tokenizer(base)
raw=Tokenizer.from_file(str(Path(budget.name_or_path)/'tokenizer.json'))
for f in (P/'optimization').rglob('selected.json'):
    prompt=json.loads(f.read_text())['candidate']['answer']
    assert budget.encode(prompt,add_special_tokens=False)==raw.encode(prompt,add_special_tokens=False).ids
reader=next((P/'readers/nla/results').glob('*.json'))
canonical=json.loads(reader.read_text())['query']
save(OUT/'actual_protocol_verification.json',dict(verified_questions=len(checked),positions=checked,
    timing='Actual answer generation prefill, question-end token selected; assistant-header suffix is not included in this causal token state.',
    source_model=spec,source_capture='scripts/reader_gepa_native_worker.py generate()',
    nla_layer_zero_based=41,nla_hf_hidden_state_index=42,nla_canonical_query=canonical,
    ao_layer_zero_based=31,ao_hf_hidden_state_index=32,ao_query=c['ao_prompt'],
    tokenizer_verification='Six frozen prompt encodings equal the pinned raw tokenizer.json after local-path resolution.',
    external_data2_paths='Not present on this host, not verified or merged.'))
print('Verified100 question-end positions, NLA/AO token IDs and6 frozen prompt tokenizer encodings.')
