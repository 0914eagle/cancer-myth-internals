"""Batch transport for unchanged baseline prompts; preserve per-request cache keys.

Multiple question pipelines submit their next stage to one GPU dispatcher. A
pipeline still waits for its own extraction/review before submitting the next
stage. Batch size is recorded, since greedy output is not bitwise invariant.
"""
from concurrent.futures import Future, ThreadPoolExecutor
import fcntl
import gc
import queue
import threading
import time
from pathlib import Path

from src.paper_baselines import CachedTask, HFTask, digest


def completed_tokens(tokens, eos_ids):
    """Stop at the first EOS, not a trailing pad belonging to another row."""
    for i, token in enumerate(tokens):
        if int(token) in eos_ids:
            return tokens[:i+1]
    return None


class LockedCachedTask(CachedTask):
    def __call__(self, messages):
        path = self.out/'batch_cache_locks'/(digest(messages)+'.lock')
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            return super().__call__(messages)


class BatchedHFTask(HFTask):
    def __init__(self, config, model_key):
        super().__init__(config, model_key)
        self.tokenizer.padding_side = 'left'
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.gpu_lock = threading.Lock()

    def length(self, messages):
        with self.gpu_lock:
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            return len(self.tokenizer(text, add_special_tokens=False)['input_ids'])

    def features(self, messages):
        with self.gpu_lock:
            return super().features(messages)

    def batch(self, requests):
        with self.gpu_lock:
            return self._batch(requests)

    def _batch(self, requests):
        texts = [self.tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in requests]
        lengths = [len(self.tokenizer(t, add_special_tokens=False)['input_ids']) for t in texts]
        if any(n > self.config['max_input_tokens'] for n in lengths):
            raise ValueError('Input exceeds pinned limit; no silent truncation')
        batch = self.tokenizer(texts, add_special_tokens=False, padding=True, return_tensors='pt')
        batch = {k:v.to(self.model.get_input_embeddings().weight.device) for k,v in batch.items()}
        width = batch['input_ids'].shape[1]
        started = time.monotonic()
        with self.torch.inference_mode():
            output = self.model.generate(**batch, do_sample=False,
                max_new_tokens=self.config['max_new_tokens'], pad_token_id=self.tokenizer.pad_token_id)
        elapsed = time.monotonic()-started
        eos = self.model.generation_config.eos_token_id
        eos = set(eos if isinstance(eos, list) else [eos])
        results = []
        for n, row in zip(lengths, output):
            tokens = completed_tokens(row[width:].tolist(), eos)
            if tokens is None:
                results.append(RuntimeError('Generation truncated; retained task has no completed answer'))
            else:
                results.append((self.tokenizer.decode(tokens, skip_special_tokens=True).strip(), {
                    'input_tokens':n, 'output_tokens':len(tokens), 'finish_reason':'eos',
                    'actual_batch_size':len(requests), 'padded_input_tokens':width,
                    'batch_elapsed_seconds':elapsed, 'batch_transport':'left_pad_greedy_v1',
                    'device_map':{k:str(v) for k,v in self.model.hf_device_map.items()}}))
        return results


class MicroBatch:
    """Bounded batch size and worst-case token budget; OOM retries smaller batches."""
    def __init__(self, backend, max_batch=8, token_budget=65536, wait_seconds=.03):
        self.backend, self.max_batch = backend, max_batch
        self.token_budget, self.wait_seconds = token_budget, wait_seconds
        self.queue = queue.Queue()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def __call__(self, messages):
        future = Future()
        self.queue.put((messages, future))
        return future.result()

    def close(self):
        self.queue.put(None)
        self.thread.join()

    def _run(self):
        deferred = None
        while True:
            item = deferred if deferred is not None else self.queue.get()
            deferred = None
            if item is None:
                return
            batch, max_length = [], 0
            deadline = time.monotonic()+self.wait_seconds
            while item is not None:
                messages, future = item
                try:
                    length = self.backend.length(messages)
                    if length > self.backend.config['max_input_tokens']:
                        raise ValueError('Input exceeds pinned limit; no silent truncation')
                except Exception as error:
                    future.set_exception(error)
                else:
                    projected = (len(batch)+1)*(max(max_length,length)+self.backend.config['max_new_tokens'])
                    if batch and projected > self.token_budget:
                        deferred = item
                        break
                    batch.append(item)
                    max_length = max(max_length, length)
                if len(batch) >= self.max_batch:
                    break
                try:
                    item = self.queue.get(timeout=max(0,deadline-time.monotonic()))
                except queue.Empty:
                    break
                if item is None:
                    self.queue.put(None)
                    break
            if batch:
                self._deliver(batch)

    def _deliver(self, batch):
        retry_smaller = False
        try:
            results = self.backend.batch([messages for messages,_ in batch])
            if len(results) != len(batch):
                raise RuntimeError('Batch result count mismatch')
        except Exception as error:
            if isinstance(error, self.backend.torch.cuda.OutOfMemoryError) and len(batch)>1:
                retry_smaller = True
            else:
                for _, future in batch:
                    future.set_exception(error)
                return
        if retry_smaller:
            # Drop the failed generate frame before clearing allocator caches.
            gc.collect()
            self.backend.torch.cuda.empty_cache()
            self.max_batch = max(1, min(self.max_batch, len(batch)//2))
            print(f'[batch] OOM: reducing batch cap to {self.max_batch}; same prompts and token limits', flush=True)
            for offset in range(0,len(batch),self.max_batch):
                self._deliver(batch[offset:offset+self.max_batch])
            return
        for (_,future),result in zip(batch,results):
            if isinstance(result,Exception):
                future.set_exception(result)
            else:
                future.set_result(result)


def run_rows(worker, rows, split, method, candidate, gate, qa, workers, on_complete):
    """Keep only workers jobs in flight, not an entire 18k-question QA backlog."""
    from concurrent.futures import wait, FIRST_COMPLETED
    iterator = iter(rows)
    def submit(pool,row):
        if (worker.out/'STOP').exists():
            raise RuntimeError('STOP requested')
        return pool.submit(worker.process,row,split,method,candidate,gate,qa)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending=set()
        for _ in range(workers):
            row=next(iterator,None)
            if row is not None: pending.add(submit(pool,row))
        done_count=0
        while pending:
            done,pending=wait(pending,return_when=FIRST_COMPLETED)
            for future in done:
                future.result()
                done_count+=1
                on_complete(done_count)
                row=next(iterator,None)
                if row is not None: pending.add(submit(pool,row))
