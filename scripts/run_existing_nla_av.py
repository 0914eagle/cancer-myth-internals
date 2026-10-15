"""Use medical_nla.run_nla unchanged with a pinned local checkpoint sidecar."""
import importlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from nla_connection import medical_package

config_path = Path(sys.argv[sys.argv.index('--config')+1])
import yaml
config = yaml.safe_load(config_path.read_text())
package = medical_package('/home/eagle0914/medical_nla')
nla = importlib.import_module(package+'.nla')
original_download = nla.hf_hub_download


def pinned_sidecar(repo_id, filename, **kwargs):
    path = Path(repo_id)
    if path.is_dir():
        sidecar = path/filename
        if not sidecar.is_file():
            raise FileNotFoundError(sidecar)
        meta = yaml.safe_load(sidecar.read_text())
        expected = config['nla_model']['expected_d_model']
        assert meta['d_model'] == expected
        assert meta['role'] == 'av'
        assert meta['extraction_layer_index'] == (20 if expected == 3584 else 32)
        return str(sidecar)
    return original_download(repo_id, filename=filename, **kwargs)


nla.hf_hub_download = pinned_sidecar
importlib.import_module(package+'.run_nla').main()
