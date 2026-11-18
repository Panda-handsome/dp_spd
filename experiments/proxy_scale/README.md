# Qwen2.5 proxy-generator scale evaluation

This experiment measures the scale-quality-cost trade-off of the DP-SPD
client-side continuation generator.  It is not a comparison of different
server LLMs: the DP mechanism, WikiText-2 examples, token projection model,
quality scorer, MIA scorer, seed, `epsilon`, `k`, and dummy ratio are held
fixed.  Only the Qwen2.5 proxy model changes.

The launcher defaults to the existing DP-SPD setting `target_epsilon=3`,
`k=9`, `r=0.2`, `epsilon_non_sensitive=3`, and
`epsilon_private=7.82`.  The last value is the current `k=9, epsilon=3`
calibration; if you change the target privacy budget or `k`, change
`EPSILON_PRIVATE` to the corresponding calibrated value rather than treating
`target_epsilon` as a noise-scale argument by itself.

## What is measured

For every real WikiText-2 sequence, DP-SPD creates one real candidate and `k`
continuation dummies.  The proxy sees the same privatized, nearest-token
projected prefix as the actual protocol.

- `proxy_oracle_suffix_ppl`: the tested proxy's PPL on the clean held-out
  suffix conditioned on that DP-projected prefix.  This measures continuation
  capacity without sampling variance; lower is better.
- `raw_dummy_rouge_l_corpus`: ROUGE-L of raw sampled dummy suffixes against
  the clean suffix. `surface_dummy_rouge_l_corpus` repeats it after the fixed
  DP suffix perturbation and token projection, i.e. for server-visible text.
  Both use `rouge_score` ROUGE-L F1 with stemming, which is the same metric
  implementation used by the historical `evaluate.load("rouge")` evaluator,
  but works offline.
- `quality_dummy_ppl`: PPL of the final server-visible dummy candidates from a
  fixed evaluator (default: Qwen2.5-0.5B).  It is separate from the tested
  proxy, so larger proxies are not given a stronger PPL judge.
- `client_proxy_generation_seconds`, `client_proxy_seconds_per_group`, and
  `client_proxy_dummy_tokens_per_second`: CUDA-synchronized time inside only
  `proxy_model.generate`. `client_dataset_build_seconds` is reported
  separately, so CPU-side DP noise and nearest-token projection do not get
  mistaken for decoding time.
- `mia_ppl_asr_mean` and `mia_min_k_asr_mean`: a fixed MIA scorer selects the
  one real candidate among `k+1` final projected candidates using PPL and
  Min-K. Lower is better; `mia_random_baseline=1/(k+1)` is the reference.

The grouped JSONL hides candidate text by default. Pass
`--save_candidate_text true` only for local debugging.

## Full four-scale run

On the server holding the four supplied model directories:

```bash
cd /path/to/dp_spd
MODEL_ROOT=models \
DATASET_PATH=/absolute/path/to/data/wikitext/wikitext-2-v1 \
GPU32_MAX_MEMORY='0:44GiB,1:44GiB' \
NUM_SAMPLES=300 \
bash scripts/run_proxy_generator_scale_qwen25.sh
```

On a four-GPU server, the launcher has two phases. It first runs the 0.5B,
3B, and 7B proxies concurrently on GPUs `0`, `1`, and `2`. After all three
finish and their memory is released, it shards the 32B proxy across GPUs
`0,1` and runs its fixed 0.5B transport, quality scorer, and MIA scorer on GPU
`2`. GPU `3` is not needed. The 32B model needs about 64 GiB for bfloat16
weights, so two 48-GiB cards are appropriate. For 80-GiB cards, set
`GPU32_MAX_MEMORY='0:72GiB,1:72GiB'` instead.

For a cheap smoke test before the full run, use:

```bash
NUM_SAMPLES=2 MAX_LENGTH=64 DUMMY_K=2 \
MODEL_ROOT=models \
DATASET_PATH=/absolute/path/to/data/wikitext/wikitext-2-v1 \
bash scripts/run_proxy_generator_scale_qwen25.sh
```

The smoke command still expects the four checkpoints.  To test only a locally
available 0.5B or 3B checkpoint, invoke
`experiments/proxy_scale/run_proxy_generator_scale.py` directly with the
corresponding `--proxy_model` and a fixed 0.5B transport/scorer.

## Outputs

The full run writes to `outputs/proxy_generator_scale/`:

- `*_config.json`: complete settings and the fixed/varied model roles.
- `*_groups.jsonl`: one non-sensitive metric record per candidate group.
- `*_summary.csv`: one row per proxy scale; use this for tables.
- `logs/qwen25_{05b,3b,7b,32b}.log`: independent worker logs.
- `qwen25_proxy_generator_scale.png`: the requested four-panel scale figure.

To regenerate the figure after changing labels or selecting a result subset:

```bash
python experiments/proxy_scale/plot_proxy_generator_scale.py \
  --summary_glob 'outputs/proxy_generator_scale/*_summary.csv' \
  --output outputs/proxy_generator_scale/qwen25_proxy_generator_scale.png
```
