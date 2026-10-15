"""Convert only downloaded staging weights to the intended BF16 inference dtype."""
import argparse
import collections
import json
from pathlib import Path
import torch
from safetensors import safe_open
from safetensors.torch import save_file

p = argparse.ArgumentParser()
p.add_argument('--input', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
tensors = {}
dtypes = collections.Counter()
with safe_open(str(a.input), framework='pt') as file:
    for key in file.keys():
        tensor = file.get_tensor(key)
        dtypes[str(tensor.dtype)] += 1
        tensors[key] = tensor.to(torch.bfloat16) if tensor.is_floating_point() else tensor
temp = a.output.with_suffix('.tmp')
save_file(tensors, str(temp), metadata={'format': 'pt'})
temp.replace(a.output)
report = dict(original_dtypes=dict(dtypes), inference_dtype='bfloat16',
              tensors=len(tensors), tensor_bytes=sum(t.numel() * t.element_size() for t in tensors.values()))
a.output.with_suffix('.conversion.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report), flush=True)
