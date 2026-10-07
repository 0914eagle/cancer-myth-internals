"""Python >=3.12 isolated renderer for pinned Well templates and parsers.

No model, network, or API calls. Read one JSON request on stdin, write one JSON.
"""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def handle(req, registry_path, well):
    from src.fpqa_well_pipelines import load_templates
    from src.fpqa_prompting import well_crepe_judge
    reg = json.loads(registry_path.read_text())
    for item in reg['well_templates'].values():
        path = well / 'prompting' / Path(item['source']).name
        if hashlib.sha256(path.read_bytes()).hexdigest() != item['sha256']:
            raise ValueError('Pinned Well template changed: ' + str(path))
    sys.path.insert(0, str(well))
    response = importlib.import_module('response')
    if req['op'] == 'judge':
        return well_crepe_judge(well, req['row'], req['answer'])
    if req['op'] == 'parse':
        return getattr(response, req['parser']).model_validate_plain_text(req['text']).get()
    templates = load_templates(well)
    cls = getattr(templates[req['family']], req['template'])
    stage = cls(**req['kwargs'])
    return {'messages': stage.generate(), 'parser': cls.ResponseClass.__name__}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--registry', type=Path, required=True)
    p.add_argument('--well-root', type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(handle(json.load(sys.stdin), a.registry, a.well_root), ensure_ascii=False))
