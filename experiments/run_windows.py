"""Native Windows CUDA study of the teaching implementation, not a serving benchmark."""
import argparse
import contextlib
import ctypes
import gc
import hashlib
import io
import json
import math
import platform
import random
import shutil
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy
import torch

from run_small import Pool, lifecycle_tests


ROOT = Path(__file__).resolve().parents[1]
DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
TOLERANCES = {"float32": (2e-6, 2e-5), "float16": (1e-3, 1e-2), "bfloat16": (1e-2, 5e-2)}


def command(*args):
    result = subprocess.run(args, cwd=ROOT, capture_output=True, timeout=20)
    def decode(value):
        if b"\x00" in value:
            return value.decode("utf-16-le", errors="replace").strip()
        try:
            return value.decode("utf-8").strip()
        except UnicodeDecodeError:
            return value.decode("mbcs" if platform.system() == "Windows" else "utf-8", errors="replace").strip()
    return {"exit_code": result.returncode, "stdout": decode(result.stdout), "stderr": decode(result.stderr)}


def environment():
    properties = torch.cuda.get_device_properties(0)
    free, total = torch.cuda.mem_get_info()
    files = ("PagedAttention_Tutorial.ipynb", "experiments/run_small.py", "experiments/run_windows.py",
             "pyproject.toml", "uv.lock", ".python-version", "uv.ps1")
    class MemoryStatus(ctypes.Structure):
        _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [
            (name, ctypes.c_ulonglong) for name in
            ("total_phys", "avail_phys", "total_page", "avail_page", "total_virtual", "avail_virtual", "extended")]
    memory = MemoryStatus()
    memory.length = ctypes.sizeof(memory)
    if platform.system() == "Windows":
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
            cpu_name = winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory)):
            raise ctypes.WinError()
    else:
        cpu_name = platform.processor()
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "os": platform.platform(), "python": platform.python_version(),
        "python_runtime": str(Path(sys.base_prefix).relative_to(ROOT)),
        "venv": str(Path(sys.prefix).relative_to(ROOT)),
        "torch_module": str(Path(torch.__file__).relative_to(ROOT)),
        "numpy_module": str(Path(numpy.__file__).relative_to(ROOT)),
        "uv": command("uv", "--version")["stdout"],
        "torch": torch.__version__, "numpy": numpy.__version__, "cuda_build": torch.version.cuda,
        "gpu": properties.name, "compute_capability": list(torch.cuda.get_device_capability()),
        "gpu_total_bytes": total, "gpu_free_bytes_at_start": free,
        "cpu": cpu_name, "torch_cpu_threads": torch.get_num_threads(),
        "ram_total_bytes": memory.total_phys, "ram_available_bytes": memory.avail_phys,
        "disk_free_bytes": shutil.disk_usage(ROOT).free,
        "nvidia_smi": command("nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.used,utilization.gpu,temperature.gpu", "--format=csv"),
        "wsl_status": command("wsl", "--status"),
        "source_commit": command("git", "rev-parse", "HEAD")["stdout"],
        "source_branch": command("git", "branch", "--show-current")["stdout"],
        "source_status": command("git", "status", "--porcelain")["stdout"],
        "source_sha256": {file: hashlib.sha256((ROOT / file).read_bytes()).hexdigest() for file in files},
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
    }


def load_notebook():
    namespace = {}
    notebook = json.loads((ROOT / "PagedAttention_Tutorial.ipynb").read_text(encoding="utf-8"))
    with contextlib.redirect_stdout(io.StringIO()):
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                exec("".join(cell["source"]), namespace)
    return namespace


def case(namespace, size, length, dtype, seed):
    # Generate on CPU, then compare the exact quantized inputs with an FP64 CPU reference.
    generator = torch.Generator().manual_seed(seed)
    k_cpu = torch.randn(length, 64, generator=generator).to(dtype)
    v_cpu = torch.randn(length, 64, generator=generator).to(dtype)
    q_cpu = torch.randn(1, 64, generator=generator).to(dtype)
    reference = torch.softmax(q_cpu.double() @ k_cpu.double().T / 8, -1) @ v_cpu.double()
    pool = namespace["KVCacheAllocator"](math.ceil(length / size) + 7, size, 64)
    pool.k_cache = pool.k_cache.to(device="cuda", dtype=dtype)
    pool.v_cache = pool.v_cache.to(device="cuda", dtype=dtype)
    random.Random(seed).shuffle(pool.free_blocks)
    sequence = namespace["SequenceState"](0, pool)
    k, v, q = k_cpu.cuda(), v_cpu.cuda(), q_cpu.cuda()
    with contextlib.redirect_stdout(io.StringIO()):
        for index in range(length):
            sequence.append_token(k[index], v[index])
    return pool, sequence, k, v, q, reference


def correctness(namespace):
    rows = []
    skipped = []
    for name, dtype in DTYPES.items():
        if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
            skipped.append({"dtype": name, "reason": "CUDA device reports no BF16 support"})
            continue
        atol, rtol = TOLERANCES[name]
        for size in (4, 16, 32):
            for length in (1, size - 1, size, size + 1, 127, 257, 1024, 2048):
                for seed in (0, 1, 2):
                    pool, sequence, k, v, q, reference = case(namespace, size, length, dtype, seed)
                    output, weights = namespace["paged_attention"](q, sequence, pool)
                    actual = output.double().cpu()
                    torch.testing.assert_close(actual, reference, atol=atol, rtol=rtol)
                    torch.testing.assert_close(weights.float().sum(), torch.ones((), device="cuda"), atol=atol, rtol=rtol)
                    assert len(set(sequence.block_table)) == math.ceil(length / size)
                    rows.append({"dtype": name, "block_size": size, "length": length, "seed": seed,
                                 "max_abs_error": (actual - reference).abs().max().item(),
                                 "atol": atol, "rtol": rtol})
    return {"cases": rows, "skipped": skipped}


def measure(function, warmup, repeats, iterations):
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    samples = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(iterations):
            function()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1000 / iterations)
    return {"wall_ms_per_call_samples": samples, "median_wall_ms_per_call": statistics.median(samples),
            "peak_extra_allocated_bytes": torch.cuda.max_memory_allocated() - baseline}


def timing(namespace, warmup, repeats, iterations):
    rows = []
    for name in ("float32", "float16"):
        dtype = DTYPES[name]
        for size in (4, 16, 32):
            for length in (128, 512, 1024, 2048):
                pool, sequence, k, v, q, reference = case(namespace, size, length, dtype, 42)
                def paged():
                    return namespace["paged_attention"](q, sequence, pool)
                def contiguous():
                    weights = torch.softmax(q @ k.T / 8, -1)
                    return weights @ v, weights
                for function in (paged, contiguous):
                    torch.testing.assert_close(function()[0].double().cpu(), reference,
                                               atol=TOLERANCES[name][0], rtol=TOLERANCES[name][1])
                # Alternate order to reduce systematic first/second measurement bias.
                functions = [("notebook_cat", paged), ("already_contiguous", contiguous)]
                if len(rows) % 2:
                    functions.reverse()
                measurements = {label: measure(function, warmup, repeats, iterations)
                                for label, function in functions}
                rows.append({"dtype": name, "block_size": size, "length": length,
                             "head_dim": 64, "batch": 1, **measurements})
                print(json.dumps({"timing_case": len(rows), "dtype": name, "block_size": size,
                                  "length": length}), flush=True)
    return rows


def memory_probe():
    gc.collect()
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    pool = Pool(4096, 16, 128, device="cuda", dtype=torch.float16)
    block = pool.alloc()
    live = torch.cuda.memory_allocated()
    expected = 2 * 4096 * 16 * 128 * 2
    assert live - baseline == expected
    pool.drop(block)
    after_drop = torch.cuda.memory_allocated()
    assert len(pool.free) == 4096 and after_drop == live
    del pool
    gc.collect()
    torch.cuda.synchronize()
    after_delete = torch.cuda.memory_allocated()
    assert after_delete == baseline
    return {"blocks": 4096, "block_size": 16, "head_dim": 128, "dtype": "float16",
            "expected_pool_bytes": expected, "baseline_allocated_bytes": baseline,
            "live_allocated_bytes": live, "after_logical_free_allocated_bytes": after_drop,
            "after_pool_delete_allocated_bytes": after_delete,
            "boundary": "PyTorch live tensor allocation, not total board VRAM. Logical free returns a block to the preallocated pool; CUDA caching may retain reserved memory."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=10)
    args = parser.parse_args()
    if min(args.warmup, args.repeats, args.iterations) < 1:
        parser.error("warmup, repeats and iterations must all be positive")
    assert torch.cuda.is_available(), "CUDA PyTorch and a visible NVIDIA GPU are required"
    torch.set_num_threads(1)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    start = time.perf_counter()
    result = {"environment": environment(),
              "parameters": {"warmup": args.warmup, "repeats": args.repeats, "iterations": args.iterations,
                             "seed": 42, "correctness_seeds": [0, 1, 2]},
              "boundary": "Single-head teaching code uses Python block lookup and torch.cat. Timings include CPU dispatch and synchronization of groups of calls; they are not pure kernel time, serving throughput, TTFT or TPOT. Contiguous baseline starts with packed K/V and excludes packing/growth. Desktop GPU activity is uncontrolled. OOM tests exhaust a logical block pool, not the GPU."}
    namespace = load_notebook()
    with torch.inference_mode():
        result["correctness"] = correctness(namespace)
        print(json.dumps({"correctness_cases": len(result["correctness"]["cases"])}), flush=True)
        result["cuda_lifecycle"] = {name: lifecycle_tests("cuda", dtype) for name, dtype in DTYPES.items()
                                    if dtype != torch.bfloat16 or torch.cuda.is_bf16_supported()}
        result["memory_probe"] = memory_probe()
        result["timing"] = timing(namespace, args.warmup, args.repeats, args.iterations)
    result["nvidia_smi_at_end"] = command("nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.used,utilization.gpu,temperature.gpu", "--format=csv")
    result["elapsed_seconds"] = time.perf_counter() - start
    result["status"] = "passed"
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"status": "passed", "output": str(output), "elapsed_seconds": result["elapsed_seconds"]}))


if __name__ == "__main__":
    main()
