"""Small reproducible mechanism study; no claim of production serving performance."""
import argparse
import contextlib
import io
import json
import math
import platform
import random
import time
from pathlib import Path

import torch


class Pool:
    """Experimental extension: reference counting and real tensor copy-on-write."""
    def __init__(self, blocks, size, dim, device='cpu', dtype=torch.float32):
        self.size = size
        self.k = torch.zeros(blocks, size, dim, device=device, dtype=dtype)
        self.v = torch.zeros_like(self.k)
        self.refs = [0] * blocks
        self.free = list(range(blocks))
        random.Random(42).shuffle(self.free)

    def alloc(self):
        if not self.free:
            raise MemoryError('pool exhausted')
        b = self.free.pop()
        assert self.refs[b] == 0
        self.refs[b] = 1
        return b

    def drop(self, b):
        assert self.refs[b] > 0
        self.refs[b] -= 1
        if self.refs[b] == 0:
            self.free.append(b)


class Seq:
    def __init__(self, pool):
        self.pool, self.table, self.n = pool, [], 0

    def append(self, k, v):
        p = self.pool
        offset = self.n % p.size
        if offset == 0:
            self.table.append(p.alloc())
        elif p.refs[self.table[-1]] > 1:
            old = self.table[-1]
            new = p.alloc()  # Failure leaves existing state intact.
            p.k[new].copy_(p.k[old])
            p.v[new].copy_(p.v[old])
            p.drop(old)
            self.table[-1] = new
        p.k[self.table[-1], offset] = k
        p.v[self.table[-1], offset] = v
        self.n += 1

    def fork(self):
        other = Seq(self.pool)
        other.table, other.n = self.table.copy(), self.n
        for b in other.table:
            self.pool.refs[b] += 1
        return other

    def read(self):
        p = self.pool
        return (p.k[self.table].reshape(-1, p.k.shape[-1])[:self.n],
                p.v[self.table].reshape(-1, p.v.shape[-1])[:self.n])

    def close(self):
        for b in self.table:
            self.pool.drop(b)
        self.table, self.n = [], 0


def original_tests(root, device):
    ns = {}
    notebook = json.loads((root / 'PagedAttention_Tutorial.ipynb').read_text(encoding='utf-8'))
    with contextlib.redirect_stdout(io.StringIO()):
        for cell in notebook['cells']:
            if cell['cell_type'] == 'code':
                exec(''.join(cell['source']), ns)
    rows = []
    for size in (4, 16, 32):
        for n in sorted(set((1, size-1, size, size+1, 127, 257))):
            for seed in (0, 1, 2):
                torch.manual_seed(seed)
                count = math.ceil(n/size) + 7
                pool = ns['KVCacheAllocator'](count, size, 64)
                pool.k_cache = pool.k_cache.to(device)
                pool.v_cache = pool.v_cache.to(device)
                random.Random(seed).shuffle(pool.free_blocks)
                seq = ns['SequenceState'](0, pool)
                k, v = torch.randn(n, 64, device=device), torch.randn(n, 64, device=device)
                q = torch.randn(1, 64, device=device)
                with contextlib.redirect_stdout(io.StringIO()):
                    for i in range(n):
                        seq.append_token(k[i], v[i])
                out, weights = ns['paged_attention'](q, seq, pool)
                # Independent float64 reference, not the implementation's block reconstruction.
                ref = torch.softmax(q.double() @ k.double().T / 8, -1) @ v.double()
                error = (out.double()-ref).abs().max().item()
                assert torch.allclose(out.double(), ref, atol=2e-6, rtol=2e-5)
                assert len(set(seq.block_table)) == math.ceil(n/size)
                assert abs(weights.sum().item()-1) < 2e-6
                rows.append(dict(block_size=size, length=n, seed=seed, max_abs_error=error))
    return rows


def lifecycle_tests(device='cpu', dtype=torch.float32):
    cases = []
    for size in (4, 16, 32):
        for n in (size-1, size, size+1):
            p = Pool(32, size, 8, device=device, dtype=dtype)
            a = Seq(p)
            k = torch.randn(n, 8, device=device, dtype=dtype)
            v = torch.randn_like(k)
            for i in range(n):
                a.append(k[i], v[i])
            b, c = a.fork(), a.fork()
            shared = len(a.table)
            assert sum(r > 0 for r in p.refs) == shared
            for index, seq in enumerate((a, b, c)):
                token = torch.ones(8, device=device, dtype=dtype) * (index + 11)
                seq.append(token, -token)
                kr, vr = seq.read()
                assert torch.equal(kr[:-1], k) and torch.equal(vr[:-1], v)
                assert torch.equal(kr[-1], token) and torch.equal(vr[-1], -token)
            for index, seq in enumerate((a, b, c)):
                assert torch.equal(seq.read()[0][-1], torch.ones(8, device=device, dtype=dtype)*(index+11))
            used = sum(r > 0 for r in p.refs)
            expected = shared + (3 if n % size == 0 else 2)
            assert used == expected
            b.close(); a.close(); c.close()
            assert len(p.free) == 32 and len(set(p.free)) == 32 and sum(p.refs) == 0
            cases.append(dict(block_size=size, prefix_length=n, shared_blocks=shared,
                              after_branch_blocks=used, reclaimed_blocks=32))
    p = Pool(1, 4, 8, device=device, dtype=dtype)
    a = Seq(p)
    a.append(torch.ones(8, device=device, dtype=dtype), torch.ones(8, device=device, dtype=dtype))
    b = a.fork()
    try:
        b.append(torch.zeros(8, device=device, dtype=dtype), torch.zeros(8, device=device, dtype=dtype))
        raise AssertionError('expected OOM')
    except MemoryError:
        assert a.n == b.n == 1 and p.refs == [2] and a.table == b.table
    b.close(); a.close()
    assert len(p.free) == 1
    return dict(device=str(device), dtype=str(dtype), branch_cases=cases, oom_atomicity=True)


def allocation_trace():
    """Identical logical workload; slot accounting, NOT measured CUDA allocations."""
    rng = random.Random(20261022)
    requests = [dict(id=i, arrival=i//4, prompt=rng.randint(17, 256),
                     output=rng.randint(8, 128)) for i in range(64)]
    rows = []
    for size in (4, 8, 16, 32, 64):
        useful = paged = static = peak_paged = peak_static = peak_active = 0
        for tick in range(max(r['arrival']+r['output'] for r in requests)):
            active = [r for r in requests if r['arrival'] <= tick < r['arrival']+r['output']]
            lengths = [r['prompt']+tick-r['arrival'] for r in active]
            u = sum(lengths)
            p = sum(math.ceil(n/size)*size for n in lengths)
            s = len(active)*512
            assert p-u <= len(active)*(size-1)
            useful += u; paged += p; static += s
            peak_paged = max(peak_paged,p); peak_static = max(peak_static,s)
            peak_active = max(peak_active,len(active))
        rows.append(dict(block_size=size, useful_slot_ticks=useful, paged_slot_ticks=paged,
                         static_512_slot_ticks=static, paged_utilization=useful/paged,
                         static_utilization=useful/static, slot_tick_savings=1-paged/static,
                         peak_paged_slots=peak_paged, peak_static_slots=peak_static,
                         peak_active=peak_active))
    # External fragmentation: the same freed intervals, sufficient total space but no contiguous extent.
    free_intervals = [(0, 16), (32, 16)]
    fragmentation = dict(free_slots=sum(n for _, n in free_intervals), request_slots=24,
                         largest_extent=16, contiguous_first_fit_fits=False,
                         paged_block_size=8, paged_fits=True)
    assert fragmentation['free_slots'] >= 24 > max(n for _, n in free_intervals)
    return dict(requests=requests, results=rows, constructed_fragmentation_case=fragmentation,
                boundary='Synthetic slot accounting; static max-length reservation baseline only. '
                         'No GPU-memory/throughput claims; ideal exact-size contiguous storage uses useful slots.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--output', default='results/cpu.json')
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(42)
    if args.device == 'cuda':
        assert torch.cuda.is_available()
    root = Path(__file__).resolve().parents[1]
    start = time.time()
    result = dict(environment=dict(python=platform.python_version(),
                                   torch=torch.__version__, cuda_build=torch.version.cuda,
                                   device=args.device), original_notebook=original_tests(root,args.device),
                  extension_lifecycle=lifecycle_tests(args.device), allocation_trace=allocation_trace())
    result['elapsed_seconds'] = time.time()-start
    result['status'] = 'passed'
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(dict(status=result['status'], original_cases=len(result['original_notebook']),
                         max_abs_error=max(r['max_abs_error'] for r in result['original_notebook']),
                         lifecycle_cases=len(result['extension_lifecycle']['branch_cases']),
                         elapsed_seconds=result['elapsed_seconds'], output=str(target))))


if __name__ == '__main__':
    main()
