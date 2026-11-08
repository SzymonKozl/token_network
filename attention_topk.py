"""
For every token of the input text file, find the most important previous tokens it attends to in Qwen3-30B-A3B
(official FP8 checkpoint by default: block-wise FP8 weights in the linear layers, attention itself in fp32 here).

    score[i, j] = mean over selected layers ( mean over all heads ( softmax-prob[layer, head, i, j] ) ),  j < i
    norm_score[i, j] = score[i, j] / sum over j' < i of score[i, j']   (row renormalized over previous tokens)
Kept per token i: union of the top-K previous tokens and the top-P nucleus (smallest set of highest-scoring
previous tokens whose norm_scores sum to >= P).

Inference runs on Modal; results are pushed to a public HF dataset with configs:
    tokens:    one row for the document (text, input_ids, tokens, offsets)
    attention: one row per token        (token_idx, token_id, attended_idx, scores, norm_scores, candidate_mass)
plus run_config.json with the parameters used.

Attention probabilities are computed exactly (fp32) by a custom causal attention backend, chunked over query rows so
that memory stays bounded for long inputs; the rest of the model runs as usual (FP8 linear layers, bf16 elsewhere).

Usage:
    python attention_topk.py --repo user/my-dataset --file a.txt --top-k 5 --top-p 0.5 --layers 24-47
"""
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any, Dict, List, Optional
import json
import time

import modal


MODEL_NAME = "Qwen/Qwen3-30B-A3B-FP8"
NATIVE_CONTEXT = 32768
ATTN_IMPL = "attn_probe"
HF_CACHE_VOL_NAME = "hf-cache"
HF_CACHE_VOL_PATH = "/hf_cache"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install([
        "torch",
        "transformers==5.19.0",
        "accelerate==1.12.0",
        # FP8 matmul kernels (loaded from the HF kernel hub)
        "kernels==0.17",
        "datasets",
        "huggingface_hub",
    ])
    .env({"HF_HOME": HF_CACHE_VOL_PATH})
)
app = modal.App("attention-topk", image=image)


# ---------------------------------------------------------------------------------------------------------------------
# attention probing (runs remotely)
# ---------------------------------------------------------------------------------------------------------------------

class AttentionRecorder:
    """Accumulates the sum over recorded layers of the head-averaged attention matrix [T, T] (fp32)."""

    def __init__(self, layers: List[int], mem_budget_gb: float):
        self.layers = set(layers)
        self.mem_budget_bytes = mem_budget_gb * 2**30
        self.acc = None
        self.seen_layers: List[int] = []


RECORDER: Optional[AttentionRecorder] = None


def probe_attention_forward(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
    """
    Drop-in replacement of HF eager causal attention, computed in fp32 and chunked over queries. Ignores
    `attention_mask` (single unpadded sequence, no KV cache) and builds the causal mask itself.

    query: [1, H, T, D]; key, value: [1, Hkv, T, D]. Returns (attn_output [1, T, H, D], None).
    """
    import torch

    rec = RECORDER
    bsz, n_heads, seq_len, head_dim = query.shape
    assert bsz == 1, "probe attention supports a single sequence"
    assert key.shape[2] == seq_len, "probe attention expects a full forward pass without KV cache"
    assert getattr(module, "sliding_window", None) is None, "sliding window attention is not supported"
    n_kv = key.shape[1]
    groups = n_heads // n_kv
    dev = query.device
    record = rec is not None and module.layer_idx in rec.layers
    if record:
        if rec.acc is None:
            rec.acc = torch.zeros(seq_len, seq_len, dtype=torch.float32, device=dev)
        rec.seen_layers.append(module.layer_idx)

    # head h uses kv head h // groups (same as HF repeat_kv)
    q = query[0].float().view(n_kv, groups, seq_len, head_dim)
    k = key[0].float().unsqueeze(1)    # [Hkv, 1, T, D]
    v = value[0].float().unsqueeze(1)  # [Hkv, 1, T, D]
    out = torch.empty(n_kv, groups, seq_len, head_dim, dtype=torch.float32, device=dev)

    # ~3 live fp32 tensors of shape [H, chunk, T] per chunk
    budget = rec.mem_budget_bytes if rec is not None else 4 * 2**30
    chunk = max(1, min(seq_len, int(budget // (3 * 4 * n_heads * seq_len))))

    for start in range(0, seq_len, chunk):
        end = min(seq_len, start + chunk)
        q_pos = torch.arange(start, end, device=dev)[:, None]
        k_pos = torch.arange(end, device=dev)[None, :]
        logits = torch.matmul(q[:, :, start:end], k[:, :, :end].transpose(-1, -2)) * scaling
        # causal mask: query i sees keys j <= i
        logits.masked_fill_(k_pos > q_pos, float("-inf"))
        probs = torch.softmax(logits, dim=-1)
        del logits
        out[:, :, start:end] = torch.matmul(probs, v[:, :, :end])
        if record:
            rec.acc[start:end, :end] += probs.mean(dim=(0, 1))
        del probs

    attn_output = out.view(1, n_heads, seq_len, head_dim).transpose(1, 2).contiguous().to(query.dtype)
    return attn_output, None


def _register_attention():
    from transformers import AttentionInterface
    from transformers.masking_utils import AttentionMaskInterface

    AttentionInterface.register(ATTN_IMPL, probe_attention_forward)
    # causal mask is built inside probe_attention_forward
    AttentionMaskInterface.register(ATTN_IMPL, lambda *args, **kwargs: None)


def select_tokens(avg, top_k: int, top_p: float, row_chunk: int = 512) -> List[Dict[str, Any]]:
    """
    avg: [T, T] layer- and head-averaged attention probabilities (lower triangular).
    Per row i: top-K previous tokens  ∪  top-P nucleus under norm_scores. Results sorted by score, descending.
    """
    import torch

    seq_len = avg.shape[0]
    res = []
    for start in range(0, seq_len, row_chunk):
        end = min(seq_len, start + row_chunk)
        q_pos = torch.arange(start, end, device=avg.device)[:, None]
        k_pos = torch.arange(seq_len, device=avg.device)[None, :]
        valid = k_pos < q_pos # shape=(row_chunk, seq_len)
        scores = torch.where(valid, avg[start:end], torch.zeros_like(avg[start:end])) # shape=(row_chunk, seq_len)
        cand_mass = scores.sum(dim=-1, keepdim=True) # shape=(row_chunk, 1)
        norm = scores / cand_mass.clamp(min=1e-30) # shape=(row_chunk, seq_len)
        vals, idx = torch.where(valid, norm, torch.full_like(norm, -1.0)).sort(dim=-1, descending=True) # shape=(row_chunk, seq_len)
        is_valid = vals >= 0 # shape=(row_chunk, seq_len)
        rank = torch.arange(seq_len, device=avg.device)[None, :] # shape=(1, seq_len)
        mass_before = torch.cumsum(vals.clamp(min=0), dim=-1) - vals.clamp(min=0) # shape=(row_chunk, seq_len)
        selected = is_valid & ((rank < top_k) | (mass_before < top_p)) # shape=(row_chunk, seq_len)
        counts = selected.sum(dim=-1).tolist() # shape=(row_chunk,)
        sel_idx = idx[selected].tolist() # shape=(sum(counts),)
        sel_raw = scores.gather(1, idx)[selected].tolist() # shape=(sum(counts),)
        sel_norm = vals[selected].tolist() # shape=(sum(counts),)
        mass = cand_mass.squeeze(-1).tolist() # shape=(row_chunk)
        offset = 0
        for r, c in enumerate(counts):
            res.append({
                "attended_idx": sel_idx[offset:offset + c],
                "scores": sel_raw[offset:offset + c],
                "norm_scores": sel_norm[offset:offset + c],
                "candidate_mass": mass[r],
            })
            offset += c
    return res


def _load_model(model_name: str, attn_impl: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype="auto", device_map="cuda",
                                                 attn_implementation=attn_impl)
    model.eval()
    torch.set_grad_enabled(False)
    return model, tokenizer


def _tokenize(tokenizer, text: str) -> Dict[str, Any]:
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids = enc["input_ids"]
    return {
        "input_ids": ids,
        "tokens": [tokenizer.decode([t]) for t in ids],
        "offsets": [list(o) for o in enc["offset_mapping"]],
    }


def _attention_matrix(model, input_ids: List[int], layers: List[int], mem_budget_gb: float):
    import torch

    global RECORDER
    RECORDER = AttentionRecorder(layers, mem_budget_gb)
    ids = torch.tensor([input_ids], device="cuda")
    # base model only: skips the [T, vocab] logits
    model.model(input_ids=ids, use_cache=False)
    rec, RECORDER = RECORDER, None
    assert sorted(rec.seen_layers) == sorted(layers), (rec.seen_layers, layers)
    return rec.acc.div_(len(layers))


# FP8 weights are ~31 GB (bf16: ~61 GB); the rest of the 80 GB H100 goes to the [T, T] attention accumulator
@app.function(
    gpu="H100",
    timeout=3600,
    volumes={HF_CACHE_VOL_PATH: modal.Volume.from_name(HF_CACHE_VOL_NAME, create_if_missing=True)},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    memory=65536,
)
def run_remote(
    text: str,
    repo: str,
    model_name: str,
    top_k: int,
    top_p: float,
    layers: Optional[List[int]],
    max_tokens: Optional[int],
    mem_budget_gb: float,
) -> Dict[str, Any]:
    import datasets
    import torch
    from huggingface_hub import HfApi

    _register_attention()
    model, tokenizer = _load_model(model_name, ATTN_IMPL)
    n_layers = model.config.num_hidden_layers
    layers = list(range(n_layers)) if layers is None else layers
    assert all(0 <= l < n_layers for l in layers), f"layers must be in [0, {n_layers})"

    tok = _tokenize(tokenizer, text)
    if max_tokens is not None and len(tok["input_ids"]) > max_tokens:
        print(f"truncating {len(tok['input_ids'])} -> {max_tokens} tokens")
        tok = {k: v[:max_tokens] for k, v in tok.items()}
    n = len(tok["input_ids"])
    assert n > 0, "empty input"
    if n > NATIVE_CONTEXT:
        print(f"warning: {n} tokens exceeds the native context of {NATIVE_CONTEXT}")

    t0 = time.time()
    avg = _attention_matrix(model, tok["input_ids"], layers, mem_budget_gb)
    selected = select_tokens(avg, top_k, top_p)
    print(f"{n} tokens, {time.time() - t0:.1f}s, peak GPU memory {torch.cuda.max_memory_allocated() / 2**30:.1f} GB, "
          f"{sum(len(s['attended_idx']) for s in selected) / n:.2f} attended tokens / token")
    del avg

    token_rows = [{"text": text, **tok}]
    attn_rows = [
        {"token_idx": i, "token_id": tok["input_ids"][i], **sel}
        for i, sel in enumerate(selected)
    ]
    run_cfg = {
        "model": model_name,
        "top_k": top_k,
        "top_p": top_p,
        "layers": layers,
        "max_tokens": max_tokens,
        "aggregation": "mean over heads, then mean over layers",
        "norm_scores": "score / sum of scores of previous tokens",
        "selection": "top-K previous tokens ∪ top-P nucleus under norm_scores",
    }
    datasets.Dataset.from_list(token_rows).push_to_hub(repo, config_name="tokens", private=False)
    datasets.Dataset.from_list(attn_rows).push_to_hub(repo, config_name="attention", private=False)
    HfApi().upload_file(
        path_or_fileobj=json.dumps(run_cfg, indent=2).encode(),
        path_in_repo="run_config.json",
        repo_id=repo,
        repo_type="dataset",
    )
    return {"n_tokens": n, "layers": layers}


# ---------------------------------------------------------------------------------------------------------------------
# local entrypoint
# ---------------------------------------------------------------------------------------------------------------------

def parse_layers(spec: str) -> Optional[List[int]]:
    """'all' -> None; '0,3,10-12' -> [0, 3, 10, 11, 12]."""
    if spec.strip().lower() == "all":
        return None
    layers = set()
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            layers.update(range(int(a), int(b) + 1))
        else:
            layers.add(int(part))
    return sorted(layers)


def main(args: Namespace):
    assert args.top_k >= 0 and 0.0 <= args.top_p <= 1.0
    path = Path(args.file)
    with app.run():
        res = run_remote.remote(
            path.read_text(), args.repo, args.model, args.top_k, args.top_p, parse_layers(args.layers),
            args.max_tokens, args.mem_budget_gb,
        )
    print(f"pushed {res['n_tokens']} token rows (layers {res['layers']}) to https://huggingface.co/datasets/{args.repo}")


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--file", required=True, help="path to the input text file")
    parser.add_argument("--repo", required=True, help="HF dataset repo to push to, e.g. user/name (public)")
    parser.add_argument("--model", default=MODEL_NAME, help="HF model id (full causal attention, e.g. Qwen3 family; "
                                                            "bf16 original: Qwen/Qwen3-30B-A3B)")
    parser.add_argument("--top-k", type=int, default=5, help="top-K previous tokens per token (0 disables)")
    parser.add_argument("--top-p", type=float, default=0.5, help="top-P nucleus of previous tokens per token (0 disables)")
    parser.add_argument("--layers", default="all", help="layers to average over: 'all' or e.g. '0,3,24-47'")
    parser.add_argument("--max-tokens", type=int, default=None, help="truncate the document to this many tokens")
    parser.add_argument("--mem-budget-gb", type=float, default=4.0, help="GPU memory per attention chunk")
    with modal.enable_output():
        main(parser.parse_args())
