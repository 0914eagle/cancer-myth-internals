"""Prepare frozen-feature probes in parallel; publish only held-out detections."""
import argparse,fcntl,json,os,sys,time,threading
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.run_paper_baselines import Worker,check_bundle,ROOT
from src.paper_baselines import HFTask,digest,sha,write_json


def feature_identity(out):
    identity=json.loads((out/'identity.json').read_text())
    adapter={f:sha(ROOT/'scripts'/f) for f in ('run_paper_baselines_v4.py','paper_v4_pipeline.py','paper_batch_inference.py','paper_shared_extraction.py','paper_cli_pool.py')}
    if not (out/'execution_policies'/(digest(adapter)+'.json')).exists():
        raise ValueError('No matching live generation adapter identity')
    return {**identity,'v4_adapter_sha256':adapter}


def save_detection(worker,split,row,gate):
    path=worker.out/'table1'/split['dataset']/split['name']/'probe_gate'/(digest([row['dataset'],row['id']])+'.json')
    if path.exists():return
    probability,threshold=gate(row['question'])
    info={'method':'probe_gate','question':row['question'],'review':'','detector_raw':'',
          'id':row['id'],'dataset':row['dataset'],'label':row['label'],'split':split['name'],'source':split['dataset'],
          'probability':probability,'threshold':threshold,'verdict':int(probability>=threshold),'invalid_detection':False}
    # Unique temporary name: another worker can safely publish the same prediction.
    import tempfile
    path.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w',dir=path.parent,prefix='.probe-',delete=False) as f:
        json.dump(info,f,ensure_ascii=False,indent=2);f.write('\n');temporary=Path(f.name)
    temporary.replace(path)


def prepare(worker,stop_event=None):
    from threadpoolctl import threadpool_limits
    lock=(worker.out/'.probe_preparer.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    status=worker.out/'probe_preparation.json'
    state={'state':'running','mode':'fixed_hidden_states_then_fold_specific_probe','updated_epoch':time.time()}
    def stopped():return (worker.out/'STOP').exists() or (stop_event is not None and stop_event.is_set())
    def progress(**kw):state.update(kw,updated_epoch=time.time());write_json(status,state)
    try:
        for dataset in ('cancer_myth','crepe'):
            splits=[s for s in worker.bundle['splits'] if s['dataset']==dataset]
            ids=list(dict.fromkeys(i for s in splits for part in ('train','dev','test') for i in s[part]))
            progress(dataset=dataset,stage='fixed_features',completed=0,total=len(ids))
            for n,i in enumerate(ids,1):
                if stopped():progress(state='stopped');return
                worker.feature(worker.lookup[i]['question'])
                if n%25==0 or n==len(ids):progress(completed=n)
            for split in splits:
                if stopped():progress(state='stopped');return
                progress(stage='fit_train_select_dev',split=split['name'])
                # Reuse exactly the existing scaler/layer/C/threshold selection.
                with threadpool_limits(limits=2,user_api='blas'):
                    gate=Worker.classifier(worker,split,'probe_gate')
                progress(stage='held_out_predictions',completed=0,total=len(split['test']))
                for n,i in enumerate(split['test'],1):
                    if stopped():progress(state='stopped');return
                    save_detection(worker,split,worker.lookup[i],gate)
                    if n%25==0 or n==len(split['test']):progress(completed=n)
        progress(state='complete')
    except BaseException as error:
        progress(state='failed',error=str(error));raise
    finally:lock.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--bundle',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--model',choices=['qwen25','gemma3'],required=True);p.add_argument('--gpu',required=True)
    a=p.parse_args();os.environ['CUDA_VISIBLE_DEVICES']=a.gpu
    cfg=json.loads(a.config.read_text());bundle=check_bundle(a.bundle);out=a.out/a.model
    worker=SimpleNamespace(args=a,config=cfg,bundle=bundle,out=out,identity=feature_identity(out),
        reg=json.loads((ROOT/cfg['registry']).read_text()),lookup={r['id']:r for r in bundle['rows']})
    worker.backend=HFTask(cfg,a.model)
    def feature(question):
        base=out/'feature_locks';base.mkdir(parents=True,exist_ok=True)
        with (base/(digest(question)+'.lock')).open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            return Worker.feature(worker,question)
    worker.feature=feature
    write_json(out/'probe_device.json',{'visible_gpu':a.gpu,'checkpoint':cfg['models'][a.model],
        'feature':"Plain wrapper; selected hidden layers; last input token; no answer generation",
        'implementation_sha256':sha(Path(__file__))})
    prepare(worker)

if __name__=='__main__':main()
