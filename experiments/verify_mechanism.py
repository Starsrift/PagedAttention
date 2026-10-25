"""Record and verify the teaching cache lifecycle on real PyTorch tensors.

This checks Pool/Seq, not a production paged kernel or model serving backend.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import time

import torch

from run_small import Pool, Seq


ROOT = Path(__file__).resolve().parents[1]
DTYPES = {"float32": torch.float32, "float16": torch.float16,
          "bfloat16": torch.bfloat16}
TOLERANCES = {"float32": (2e-6, 2e-5), "float16": (1e-3, 1e-2),
              "bfloat16": (1e-2, 5e-2)}


def attention(q, k, v):
    return torch.softmax(q @ k.T / math.sqrt(k.shape[-1]), -1) @ v


class Study:
    def __init__(self, device, name, seed, blocks=16):
        self.device, self.name = device, name
        self.dtype = DTYPES[name]
        self.pool = Pool(blocks, 4, 8, device=device, dtype=self.dtype)
        self.active = {}
        self.generator = torch.Generator().manual_seed(seed)
        self.checks = 0
        self.max_error = 0.0

    def create(self, name):
        self.active[name] = (Seq(self.pool), [], [])

    def append(self, name):
        seq, keys, values = self.active[name]
        k = torch.randn(8, generator=self.generator).to(self.dtype)
        v = torch.randn(8, generator=self.generator).to(self.dtype)
        seq.append(k.to(self.device), v.to(self.device))
        keys.append(k)
        values.append(v)

    def fork(self, source, target):
        seq, keys, values = self.active[source]
        self.active[target] = (seq.fork(), keys.copy(), values.copy())

    def close(self, name):
        self.active.pop(name)[0].close()

    def state(self):
        return {"sequences": {name: {"length": seq.n, "table": seq.table.copy()}
                              for name, (seq, _, _) in self.active.items()},
                "refs": self.pool.refs.copy(), "free": self.pool.free.copy()}

    def check(self):
        expected_refs = Counter(b for seq, _, _ in self.active.values() for b in seq.table)
        assert self.pool.refs == [expected_refs[i] for i in range(len(self.pool.refs))]
        assert len(self.pool.free) == len(set(self.pool.free))
        assert set(self.pool.free) == {i for i, refs in enumerate(self.pool.refs) if refs == 0}
        for seq, keys, values in self.active.values():
            assert seq.n == len(keys) == len(values)
            assert len(seq.table) == math.ceil(seq.n / self.pool.size)
            assert len(set(seq.table)) == len(seq.table)
            if not keys:
                continue  # Empty attention is outside this experiment's contract.
            k, v = seq.read()
            expected_k, expected_v = torch.stack(keys), torch.stack(values)
            assert torch.equal(k.cpu(), expected_k)
            assert torch.equal(v.cpu(), expected_v)
            q = torch.arange(1, 9, dtype=torch.float32).reshape(1, 8).div(8).to(self.dtype)
            ref = attention(q.double(), expected_k.double(), expected_v.double())
            out = attention(q.to(self.device), k, v).double().cpu()
            atol, rtol = TOLERANCES[self.name]
            torch.testing.assert_close(out, ref, atol=atol, rtol=rtol)
            self.max_error = max(self.max_error, (out - ref).abs().max().item())
            self.checks += 1

    def finish(self):
        for name in list(self.active):
            self.close(name)
        self.check()
        assert len(self.pool.free) == len(self.pool.refs)


def walkthrough(device, dtype):
    study = Study(device, dtype, seed=42, blocks=8)
    snapshots = []

    def record(event):
        study.check()
        snapshots.append({"event": event, **study.state()})

    study.create("A")
    for _ in range(5):
        study.append("A")
        record("append_prompt_A")
    study.append("A")
    record("decode_A_to_6_tokens")
    study.fork("A", "B")
    record("fork_A_to_B_shared_prefix")
    study.append("A")
    record("append_A_copy_shared_tail")
    study.append("B")
    record("append_B_preserve_A")
    for _ in range(2):
        study.append("A")
        record("append_A_cross_block_boundary")
    study.close("B")
    record("close_B_keep_shared_prefix_for_A")
    study.close("A")
    record("close_A_reclaim_all_blocks")
    study.finish()
    return {"snapshots": snapshots, "sequence_checks": study.checks,
            "max_abs_error": study.max_error}


def negative_control(device, dtype):
    study = Study(device, dtype, seed=17)
    study.create("A")
    for _ in range(6):
        study.append("A")
    study.check()
    seq, keys, values = study.active["A"]
    original = seq.table.copy()
    q = torch.arange(1, 9, dtype=torch.float32).reshape(1, 8).div(8).to(study.dtype)
    ref = attention(q.double(), torch.stack(keys).double(), torch.stack(values).double())
    try:
        seq.table[0], seq.table[1] = seq.table[1], seq.table[0]
        k, v = seq.read()
        data_mismatch = not torch.equal(k.cpu(), torch.stack(keys))
        out = attention(q.to(device), k, v).double().cpu()
        atol, rtol = TOLERANCES[dtype]
        output_mismatch = not torch.allclose(out, ref, atol=atol, rtol=rtol)
        assert data_mismatch and output_mismatch, "Test failed to detect corrupted block order"
        error = (out - ref).abs().max().item()
    finally:
        seq.table[:] = original
    study.check()
    study.finish()
    return {"injected_fault": "swap first and second logical block mappings",
            "data_mismatch_detected": data_mismatch,
            "attention_mismatch_detected": output_mismatch,
            "corrupted_output_max_abs_error": error, "restored_state_passed": True}


def mixed_operations(device, dtype, seed, steps):
    study = Study(device, dtype, seed)
    rng = random.Random(seed)
    study.create("s0")
    next_id = 1
    counts = Counter()
    for _ in range(steps):
        if not study.active:
            study.create(f"s{next_id}")
            next_id += 1
        name = rng.choice(list(study.active))
        roll = rng.random()
        if roll < 0.2 and len(study.active) < 5:
            study.fork(name, f"s{next_id}")
            next_id += 1
            counts["fork"] += 1
        elif roll > 0.85 or study.active[name][0].n >= 24:
            study.close(name)
            counts["close"] += 1
        else:
            before = study.state()
            try:
                study.append(name)
                counts["append"] += 1
            except MemoryError:
                assert study.state() == before
                counts["oom_state_preserved"] += 1
        study.check()
    study.finish()
    return {"seed": seed, "steps": steps, "operations": dict(counts),
            "sequence_checks": study.checks, "max_abs_error": study.max_error,
            "all_blocks_reclaimed": True}


def main():
    if not __debug__:
        raise RuntimeError("Assertions must be enabled for this verification experiment")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("steps must be positive")
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    if args.device == "cuda":
        assert torch.cuda.is_available()
    start = time.perf_counter()
    results = {}
    for dtype in DTYPES:
        if args.device == "cuda" and dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
            continue
        results[dtype] = {"walkthrough": walkthrough(args.device, dtype),
                          "negative_control": negative_control(args.device, dtype),
                          "mixed_operations": [mixed_operations(args.device, dtype, seed, args.steps)
                                               for seed in (0, 1, 2)]}
        print(json.dumps({"dtype": dtype, "status": "passed"}), flush=True)
    result = {"status": "passed", "device": args.device, "torch": torch.__version__,
              "python": platform.python_version(), "cuda_build": torch.version.cuda,
              "gpu": torch.cuda.get_device_name(0) if args.device == "cuda" else None,
              "source_sha256": {f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest()
                                for f in ("experiments/run_small.py", "experiments/verify_mechanism.py")},
              "boundary": "Pool/Seq extension with single-head synthetic Q/K/V. Empty attention, concurrency, real model generation and native paged kernels are not covered. Mixed-operation OOM checks cover pool allocation failure only.",
              "results": results, "elapsed_seconds": time.perf_counter() - start}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"status": "passed", "output": str(output),
                      "elapsed_seconds": result["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
