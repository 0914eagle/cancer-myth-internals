"""Run the older semantic-AO controls with fresh, uniformly Luna feedback."""
import os
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]


def main():
    for dataset in ['cancer_myth','crepe']:
        policy=ROOT/'results/nla_ao_parallel_20261012/qwen8_migration_policy.json'
        if dataset=='crepe' and policy.exists() and json.loads(policy.read_text())['state']=='8b_followup_selected':
            print('27B CREPE deferred by user; followup moved to Qwen3-8B',flush=True)
            return
        env={**os.environ,'HF_HUB_OFFLINE':'1','HF_HOME':'/data3/heejae/hf_cache',
            'FPQA_GEPA_DATASET':dataset,'FPQA_GEPA_QUERY':'semantic',
            'FPQA_GEPA_NATIVE_GPUS':'0,1','FPQA_GEPA_READER_GPUS':'2,3'}
        subprocess.run([sys.executable,'-u','scripts/supervise_gepa_ao_feedback_ablation.py'],
            cwd=ROOT,env=env,check=True)


if __name__=='__main__':main()
