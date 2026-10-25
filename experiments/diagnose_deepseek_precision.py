"""Check baseline activations in FP16/BF16 before any paged cache is introduced."""
import argparse
import gc
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    root = Path(__file__).resolve().parents[1]
    model_dir = root / "models/DeepSeek-R1-Distill-Qwen-1.5B"
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    chat = tokenizer.apply_chat_template([{"role": "user", "content": "用一句话解释什么是键值缓存。"}],
                                        tokenize=False, add_generation_prompt=True)
    ids = tokenizer(chat, add_special_tokens=False, return_tensors="pt").input_ids.to("cuda")
    result = {"scope": "ordinary DynamicCache prefill only, no paged adapter", "prompt_token_ids": ids.tolist(),
              "bf16_supported": torch.cuda.is_bf16_supported(), "runs": {}}
    for name, dtype in (("float16", torch.float16), ("bfloat16", torch.bfloat16)):
        model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=dtype, local_files_only=True,
                                                    attn_implementation="eager").to("cuda").eval()
        records = []
        def hook(label):
            def check(module, inputs, output):
                value = output[0] if isinstance(output, tuple) else output
                finite = torch.isfinite(value).all().item()
                records.append({"module": label, "finite": finite,
                                "max_abs": value.abs().max().item() if finite else None})
            return check
        handles = []
        for index, layer in enumerate(model.model.layers):
            handles.append(layer.self_attn.register_forward_hook(hook(f"layer_{index}.attention")))
            handles.append(layer.mlp.register_forward_hook(hook(f"layer_{index}.mlp")))
        with torch.inference_mode():
            bad_weights = [key for key, value in model.named_parameters() if not torch.isfinite(value).all().item()]
            cache = DynamicCache(config=model.config)
            out = model(input_ids=ids, attention_mask=torch.ones_like(ids), past_key_values=cache,
                        use_cache=True, logits_to_keep=1)
            finite = torch.isfinite(out.logits).all().item()
        result["runs"][name] = {"nonfinite_weight_names": bad_weights, "logits_finite": finite,
                                 "first_nonfinite_module": next((r for r in records if not r["finite"]), None),
                                 "activation_checks": records}
        for handle in handles:
            handle.remove()
        del handles, out, cache, model
        gc.collect()
        torch.cuda.empty_cache()
        print(json.dumps({"dtype": name, "logits_finite": finite,
                          "first_nonfinite_module": result["runs"][name]["first_nonfinite_module"]}), flush=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
