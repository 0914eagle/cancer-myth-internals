"""Cross-process cap for all GPT baseline, extraction and optimization requests."""
import fcntl,time
from pathlib import Path
from src.fpqa_cli_backend import CLIBackend


class PooledCLI:
    def __init__(self,config,root,stop,slots=4):
        self.cli=CLIBackend({**config,'exact_well_messages':True})
        self.root=Path(root)/'shared/gpt_slots';self.root.mkdir(parents=True,exist_ok=True)
        self.stop=Path(stop);self.slots=slots
    def __call__(self,messages):
        while True:
            if self.stop.exists():raise RuntimeError('STOP requested')
            for i in range(self.slots):
                lock=(self.root/f'{i}.lock').open('a')
                try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:lock.close();continue
                try:return self.cli(messages)
                finally:lock.close()
            time.sleep(.1)
