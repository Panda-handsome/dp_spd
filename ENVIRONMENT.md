# Tested environment

Validation date: 2026-11-18 (Asia/Shanghai). The tests used the existing Conda environment `myenv` on Linux.

| Component | Observed version or state |
| --- | --- |
| Python | 3.10.19 |
| PyTorch | 2.9.0+cu128 |
| Transformers | 5.7.0 |
| Datasets | 2.16.1 |
| Accelerate | 1.13.0 |
| NumPy | 1.26.4 |
| scikit-learn | 1.7.2 |
| CUDA in this session | Unavailable (`torch.cuda.is_available() == False`; no GPU visible to the process) |

[requirements-tested.txt](requirements-tested.txt) lists the observed versions of the direct Python dependencies. It is a compact version record, not a full transitive lockfile. Install a PyTorch build appropriate for the target machine and CUDA driver.

`pip check` reports one pre-existing environment conflict: `sentence-transformers==2.2.2` requires `transformers<5`, while `myenv` has `transformers==5.7.0`. This repository does not import `sentence-transformers`; the tested DP-SPD path completed successfully. A fresh environment containing only this project's dependencies avoids that unrelated constraint.

## Validation performed

- `scripts/run_classification.sh sst2 --dry_run true` completed in `myenv`.
- `scripts/run_generation.sh wikitext --dry_run true` completed in `myenv`.
- A CPU run loaded a local Qwen2.5-0.5B-Instruct checkpoint and executed the client pipeline on one manually supplied sentence with one dummy. It completed prefix perturbation, dummy suffix generation, candidate shuffling, and nearest-token projection. The saved output contained two candidates and selected the real candidate by the recorded shuffle index.
- The same small run also completed with API-only projection disabled, exercising the embedding-candidate path.

The example command below uses a checkpoint path supplied by the reader. It writes output under the ignored `outputs/` directory:

```bash
conda activate myenv
python experiments/examples/generate_dpspd_examples.py \
  --manual_text 'The movie begins with a promising idea and a strong cast, but the story slows down and the ending feels predictable despite several memorable scenes.' \
  --max_examples 1 --max_length 28 --dummy_ratio 0.25 --n_dummy 1 \
  --continuation_model /path/to/Qwen2.5-0.5B-Instruct \
  --continuation_dtype float32 --device cpu \
  --include_projected_api_text true \
  --output_dir outputs/smoke --output_prefix cpu_example
```

The full IMDb, WikiText-2, CNN/DailyMail, WildChat, GPU speed, and baseline sweeps were not run in this validation. They require their specified model checkpoints, datasets, and, for practical runtimes, suitable GPUs. A dry run prints configuration and does not establish metric reproducibility.
