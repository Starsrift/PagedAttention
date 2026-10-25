"""Execute E1-E5 against a local immutable model and record its experiment plan."""
import argparse
from collections import Counter
from datetime import datetime, timezone, timedelta
import gc
import hashlib
import json
import math
from pathlib import Path
import platform
import statistics
import subprocess
import time
import traceback

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from model_paged_cache import PagedCache


ROOT = Path(__file__).resolve().parents[1]
ATOL, RTOL = 0.02, 0.002
PROMPTS = ["用一句话解释什么是键值缓存。", "What is the capital of France? Answer briefly.",
           "请列出学习编程的两个建议。"]


def digest(file):
    result = hashlib.sha256()
    with file.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def sync():
    torch.cuda.synchronize()


def clean():
    gc.collect()
    sync()
    torch.cuda.empty_cache()


def gpu_status():
    data = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.used,utilization.gpu",
                           "--format=csv"], capture_output=True, timeout=20)
    return data.stdout.decode("utf-8", errors="replace").strip()


def forward(model, ids, cache):
    total = cache.get_seq_length() + ids.shape[-1]
    mask = torch.ones((1, total), dtype=torch.long, device=ids.device)
    out = model(input_ids=ids, attention_mask=mask, past_key_values=cache,
                use_cache=True, logits_to_keep=1)
    logits = out.logits[:, -1, :]
    assert torch.isfinite(logits).all(), "Nonfinite logits"
    return logits


def new_dynamic(model):
    return DynamicCache(config=model.config)


def new_paged(model, block, length, extra=4):
    return PagedCache(model.config, block, math.ceil(length / block) + extra)


def exact_cache(paged, reference):
    for index, layer in enumerate(paged.layers):
        k, v = layer.materialize()
        rk, rv = reference[index]
        assert k.shape == rk.shape and v.shape == rv.shape
        assert k.dtype == rk.dtype and v.dtype == rv.dtype
        assert torch.equal(k, rk) and torch.equal(v, rv), f"K/V mismatch at layer {index}"


def freeze(paged):
    return [(k.clone(), v.clone()) for k, v in paged]


def refs_check(caches):
    for index in range(len(caches[0].layers)):
        pool = caches[0].layers[index].pool
        expected = Counter(block for cache in caches for block in cache.layers[index].table)
        assert pool.refs == [expected[b] for b in range(len(pool.refs))]
        assert len(pool.free) == len(set(pool.free))
        assert set(pool.free) == {b for b, refs in enumerate(pool.refs) if refs == 0}


def compare(a, b):
    maximum = (a.float() - b.float()).abs().max().item()
    torch.testing.assert_close(a, b, atol=ATOL, rtol=RTOL)
    return maximum


def generation(model, tokenizer, ids, steps):
    cache = new_dynamic(model)
    generated = []
    for step in range(steps):
        logits = forward(model, ids if step == 0 else token, cache)
        token = logits.argmax(-1, keepdim=True)
        generated.append(token.item())
        if token.item() == model.config.eos_token_id:
            break
    row = {"prompt_tokens": ids.shape[-1], "generated_token_ids": generated,
           "generated_text": tokenizer.decode(generated, skip_special_tokens=True),
           "cached_tokens": cache.get_seq_length(), "status": "passed"}
    assert generated
    return row


def e2_roundtrip(model, token_source, rows):
    for block in (4, 16, 32):
        for length in sorted({1, block - 1, block, block + 1, 127}):
            reference = new_dynamic(model)
            logits = forward(model, token_source[:, :length], reference)
            del logits
            paged = new_paged(model, block, length)
            for index, source in enumerate(reference.layers):
                paged.update(source.keys, source.values, index,
                             {"cache_position": torch.arange(length, device="cuda")})
            exact_cache(paged, reference)
            for layer in paged.layers:
                assert len(layer.table) == math.ceil(length / block)
                assert len(layer.table) * block - length <= block - 1
            negative = None
            if length > block:
                layer = paged.layers[0]
                original = layer.table.copy()
                try:
                    layer.table[0], layer.table[1] = layer.table[1], layer.table[0]
                    k, v = layer.materialize()
                    negative = not (torch.equal(k, reference.layers[0].keys) and
                                    torch.equal(v, reference.layers[0].values))
                    assert negative, "Corrupted block order was not detected"
                    del k, v
                finally:
                    layer.table[:] = original
                exact_cache(paged, reference)
            stats = paged.stats()
            assert not stats["persistent_contiguous_shadow"]
            rows.append({"block_size": block, "length": length, "layers_checked": len(paged.layers),
                         "bitwise_equal": True, "negative_control_detected": negative,
                         "stats": stats, "status": "passed"})
            paged.close()
            assert all(not any(layer.pool.refs) for layer in paged.layers)
            del paged, reference
            print(json.dumps({"experiment": "E2", "B": block, "T": length, "status": "passed"}), flush=True)


def e3_generation(model, tokenizer, prompts, steps, rows):
    for block in (4, 16, 32):
        for prompt_index, ids in enumerate(prompts):
            dynamic = new_dynamic(model)
            paged = new_paged(model, block, ids.shape[-1] + steps)
            row = {"block_size": block, "prompt_index": prompt_index,
                   "prompt_tokens": ids.shape[-1], "step_errors": [], "generated_token_ids": [],
                   "status": "running"}
            rows.append(row)
            d_input, p_input = ids, ids
            for step in range(steps):
                d_logits = forward(model, d_input, dynamic)
                p_logits = forward(model, p_input, paged)
                error = (d_logits.float() - p_logits.float()).abs().max().item()
                d_token = d_logits.argmax(-1, keepdim=True)
                p_token = p_logits.argmax(-1, keepdim=True)
                top2 = d_logits.float().topk(2).values[0]
                row["step_errors"].append({"step": step, "max_abs_logit_error": error,
                                           "top2_margin": (top2[0] - top2[1]).item(),
                                           "reference_token": d_token.item(), "paged_token": p_token.item()})
                compare(d_logits, p_logits)
                assert torch.equal(d_token, p_token), f"Greedy token divergence at step {step}"
                assert paged.get_seq_length() == dynamic.get_seq_length()
                row["generated_token_ids"].append(p_token.item())
                d_input, p_input = d_token, p_token
                if d_token.item() == model.config.eos_token_id:
                    break
            row.update({"status": "passed", "all_tokens_equal": True,
                        "max_abs_logit_error": max(v["max_abs_logit_error"] for v in row["step_errors"]),
                        "generated_text": tokenizer.decode(row["generated_token_ids"], skip_special_tokens=True),
                        "stats": paged.stats()})
            assert not row["stats"]["persistent_contiguous_shadow"]
            paged.close()
            del dynamic, paged, d_logits, p_logits
            print(json.dumps({"experiment": "E3", "B": block, "prompt": prompt_index,
                              "tokens": len(row["generated_token_ids"]), "status": "passed"}), flush=True)


def e4_branches(model, source, branch_tokens, rows):
    for block in (4, 16, 32):
        for length in (block - 1, block, block + 1):
            ids = source[:, :length]
            paged = new_paged(model, block, length + 2, extra=3)
            logits = forward(model, ids, paged)
            del logits
            original = freeze(paged)
            a, b = paged.fork(), paged.fork()
            refs_check([paged, a, b])
            before_blocks = paged.stats()["physical_blocks_in_use"]
            errors = []
            a_values = None
            for branch_index, branch in enumerate((a, b)):
                dynamic = new_dynamic(model)
                logits = forward(model, ids, dynamic)
                del logits
                token = branch_tokens[branch_index]
                expected = forward(model, token, dynamic)
                actual = forward(model, token, branch)
                errors.append(compare(expected, actual))
                exact_cache(branch, [(layer.keys, layer.values) for layer in dynamic.layers])
                exact_cache(paged, original)
                if branch is a:
                    exact_cache(b, original)
                    a_values = freeze(a)
                else:
                    exact_cache(a, a_values)
                refs_check([paged, a, b])
                del dynamic, expected, actual
            copies = paged.stats()["cow_copies"]
            assert copies == (2 * len(paged.layers) if length % block else 0)
            assert paged.stats()["physical_blocks_in_use"] == before_blocks + 2 * len(paged.layers)
            a.close()
            refs_check([paged, a, b])
            b.close()
            refs_check([paged, a, b])
            paged.close()
            refs_check([paged, a, b])
            assert all(len(layer.pool.free) == layer.blocks for layer in paged.layers)
            rows.append({"block_size": block, "prefix_tokens": length,
                         "branch_max_abs_logit_errors": errors, "cow_copies_all_layers": copies,
                         "all_blocks_reclaimed": True, "status": "passed"})
            del paged, a, b, original, a_values
            print(json.dumps({"experiment": "E4", "B": block, "T": length, "status": "passed"}), flush=True)


def e4_oom(model, source):
    # Test layer allocation failure only. Do not claim a full forward is transactional.
    dynamic = new_dynamic(model)
    logits = forward(model, source[:, :3], dynamic)
    del logits
    paged = PagedCache(model.config, block_size=4, blocks_per_layer=1)
    for index, layer in enumerate(dynamic.layers):
        paged.update(layer.keys, layer.values, index)
    branch = paged.fork()
    original = freeze(paged)
    before = [(layer.n, layer.table.copy(), layer.pool.refs.copy(), layer.pool.free.copy())
              for layer in branch.layers]
    failures = 0
    for index, layer in enumerate(branch.layers):
        try:
            layer.update(dynamic.layers[index].keys[..., :1, :], dynamic.layers[index].values[..., :1, :])
            raise AssertionError("Expected logical OOM")
        except MemoryError:
            failures += 1
        assert (layer.n, layer.table, layer.pool.refs, layer.pool.free) == before[index]
    exact_cache(paged, original)
    exact_cache(branch, original)
    branch.close()
    paged.close()
    assert all(not any(layer.pool.refs) for layer in paged.layers)
    return {"layers_checked": failures, "allocation_failure_preserved_state": True,
            "scope": "per-layer shared-tail allocation failure, not whole-forward atomicity", "status": "passed"}


def measure_once(model, ids, steps, kind):
    clean()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    cache = new_dynamic(model) if kind == "dynamic" else new_paged(model, 16, ids.shape[-1] + steps)
    sync()
    start = time.perf_counter()
    logits = forward(model, ids, cache)
    sync()
    prefill_ms = (time.perf_counter() - start) * 1000
    token = logits.argmax(-1, keepdim=True)
    del logits
    times = []
    for _ in range(steps):
        sync()
        start = time.perf_counter()
        logits = forward(model, token, cache)
        sync()
        times.append((time.perf_counter() - start) * 1000)
        token = logits.argmax(-1, keepdim=True)
        del logits
    live = torch.cuda.memory_allocated() - baseline
    peak = torch.cuda.max_memory_allocated() - baseline
    length = cache.get_seq_length()
    head_dim = getattr(model.config, "head_dim", model.config.hidden_size // model.config.num_attention_heads)
    per_token = 2 * model.config.num_hidden_layers * model.config.num_key_value_heads * head_dim * 2
    stats = cache.stats() if kind == "paged" else {"logical_kv_bytes": length * per_token}
    if kind == "paged":
        assert stats["pool_tensor_bytes"] == (math.ceil((ids.shape[-1] + steps) / 16) + 4) * 16 * per_token
        assert stats["mapped_blocks"] == math.ceil(length / 16) * model.config.num_hidden_layers
        assert not stats["persistent_contiguous_shadow"]
        cache.close()
    del cache, token
    clean()
    return {"prefill_ms": prefill_ms, "decode_ms_samples": times,
            "median_decode_ms": statistics.median(times), "final_cached_tokens": length,
            "allocated_delta_bytes": live, "peak_extra_allocated_bytes": peak,
            "kv_bytes_per_token_all_layers": per_token, "stats": stats}


def e5_measurements(model, ids, rows):
    for kind in ("dynamic", "paged"):
        measure_once(model, ids, 8, kind)
    for repeat in range(3):
        order = ("dynamic", "paged") if repeat % 2 == 0 else ("paged", "dynamic")
        for kind in order:
            row = measure_once(model, ids, 8, kind)
            row.update({"repeat": repeat, "kind": kind, "prompt_tokens": ids.shape[-1], "decode_steps": 8})
            rows.append(row)
            print(json.dumps({"experiment": "E5", "repeat": repeat, "kind": kind}), flush=True)


def main():
    if not __debug__:
        raise RuntimeError("Assertions must be enabled")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=ROOT / "models/Qwen3-0.6B")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--new-tokens", type=int, default=32)
    parser.add_argument("--plan", type=Path, default=ROOT / "experiments/SMALL_MODEL_PLAN.md")
    args = parser.parse_args()
    if args.new_tokens < 1:
        parser.error("new-tokens must be positive")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError("Use a new output file for each run")
    result = {"status": "running", "started_at": datetime.now(timezone(timedelta(hours=8))).isoformat(),
              "experiments": {}, "parameters": {"new_tokens": args.new_tokens, "block_sizes": [4, 16, 32],
                                                  "logit_atol": ATOL, "logit_rtol": RTOL, "seed": 42},
              "boundary": "Real multi-layer GQA model with paged storage plus contiguous reconstruction. No native paged attention kernel or serving benchmark."}
    def save():
        args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    try:
        torch.set_num_threads(1)
        torch.manual_seed(42)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        assert torch.cuda.is_available()
        manifest = json.loads((args.model_dir / "experiment-model-manifest.json").read_text(encoding="utf-8"))
        assert manifest["status"] == "verified"
        for file, record in manifest["files"].items():
            assert digest(args.model_dir / file) == record["sha256"], f"Model file changed: {file}"
        result["model"] = manifest
        result["environment"] = {"python": platform.python_version(), "torch": torch.__version__,
                                 "transformers": transformers.__version__, "cuda_build": torch.version.cuda,
                                 "gpu": torch.cuda.get_device_name(0), "nvidia_smi_start": gpu_status()}
        plan_path = args.plan.resolve()
        assert plan_path.is_relative_to(ROOT), "Experiment plan must be inside the repository"
        result["parameters"]["plan"] = plan_path.relative_to(ROOT).as_posix()
        files = [plan_path.relative_to(ROOT).as_posix(), "experiments/run_small_model.py", "experiments/model_paged_cache.py",
                 "experiments/prepare_small_model.py", "pyproject.toml", "uv.lock", "uv.ps1"]
        result["source_sha256"] = {file: digest(ROOT / file) for file in files}
        print(json.dumps({"phase": "loading_model", "revision": manifest["revision"]}), flush=True)
        tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True, trust_remote_code=False)
        model = AutoModelForCausalLM.from_pretrained(args.model_dir, dtype=torch.float16,
                                                    attn_implementation="eager", local_files_only=True,
                                                    trust_remote_code=False).to("cuda").eval()
        result["environment"].update({"model_parameter_count": sum(p.numel() for p in model.parameters()),
                                      "model_parameter_bytes": sum(p.numel() * p.element_size() for p in model.parameters()),
                                      "config": model.config.to_dict()})
        prompts = []
        chat_options = {"enable_thinking": False} if model.config.model_type == "qwen3" else {}
        result["parameters"]["chat_template_options"] = chat_options
        result["parameters"]["decoding"] = "greedy regression, not a model-quality evaluation"
        for text in PROMPTS:
            chat = tokenizer.apply_chat_template([{"role": "user", "content": text}], tokenize=False,
                                                 add_generation_prompt=True, **chat_options)
            prompts.append(tokenizer(chat, return_tensors="pt").input_ids.to("cuda"))
        source_text = "Paging manages key and value caches in fixed size blocks. " * 80
        source = tokenizer(source_text, return_tensors="pt").input_ids[:, :127].to("cuda")
        assert source.shape[-1] == 127
        branch_tokens = [tokenizer(word, add_special_tokens=False, return_tensors="pt").input_ids[:, :1].to("cuda")
                         for word in ("apple", "river")]
        assert not torch.equal(branch_tokens[0], branch_tokens[1])
        with torch.inference_mode():
            result["active_experiment"] = "E1"
            result["experiments"]["E1"] = {"prompts": PROMPTS, "cases": [], "status": "running"}
            save()
            for ids in prompts:
                result["experiments"]["E1"]["cases"].append(generation(model, tokenizer, ids, args.new_tokens))
            result["experiments"]["E1"]["status"] = "passed"
            save()
            print(json.dumps({"experiment": "E1", "status": "passed"}), flush=True)
            for label, function, parameters in [
                ("E2", e2_roundtrip, (model, source)),
                ("E3", e3_generation, (model, tokenizer, prompts, args.new_tokens)),
                ("E4", e4_branches, (model, source, branch_tokens)),
                ("E5", e5_measurements, (model, source[:, :64])),
            ]:
                result["active_experiment"] = label
                result["experiments"][label] = {"status": "running", "cases": []}
                save()
                function(*parameters, result["experiments"][label]["cases"])
                if label == "E4":
                    result["experiments"][label]["oom"] = e4_oom(model, source)
                result["experiments"][label]["status"] = "passed"
                save()
                clean()
        result["status"] = "passed"
        result.pop("active_experiment", None)
        result["environment"]["nvidia_smi_end"] = gpu_status()
        save()
        print(json.dumps({"status": "passed", "output": str(args.output)}), flush=True)
    except Exception as error:
        result["status"] = "failed"
        result["error"] = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
        label = result.get("active_experiment")
        if label:
            result["experiments"][label]["status"] = "failed"
        save()
        raise


if __name__ == "__main__":
    main()
