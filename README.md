# Network science on graph formed by attention matrix
## Data collection details
Model: `Qwen/Qwen3-30B-A3B-FP8`. We use quantized version to reduce computation footprint. Inference is done on Modal cloud platform (big thanks for free 30$ credits) on H100 GPU. Current setup should work for around 30k token sequences.
## Aggregation
We average the attention scores across attention heads, then across layers, to obtain single matrix. Other aggregation methods are also worth considering in the future (for example taking max value, or just focusing on a single layer.)

### Env setup
```bash
uv venv && uv pip install -r requirements.txt
```
Modal login
```bash
uv run modal setup
```
Requires Modal secret `huggingface-secret` (with `HF_TOKEN`, write access) and uses the `hf-cache` volume.
### Usage
```bash
uv run attention_topk.py --repo user/my-dataset --file a.txt --top-k 5 --top-p 0.5 --layers 24-47
```
### Output dataset format
- config `tokens`, one row: `text, input_ids, tokens, offsets` (char offsets into
  `text`)
- config `attention`, one row per token:
  token_idx, token_id, attended_idx, scores, norm_scores, candidate_mass`
  (`attended_idx` sorted by score, descending; `candidate_mass` = sum of all attention scores for the token (including rejected in filtering))
- `run_config.json`: parameters of the run
### Datasets produced up to now:
####  `SzymonKozl/graph_test1` - https://huggingface.co/datasets/SzymonKozl/graph_test1
Test one, english plain text of Shakespeare "Midnight summer's dream" (credits to https://github.com/cobanov/shakespeare-dataset.git). around 27k tokens. Min. 1k connections for each token. Took around 0.2$ to generate.

Q: How to use this in R?

A: Ask chat gpt, provide him above link