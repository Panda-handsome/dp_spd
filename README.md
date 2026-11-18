# DP-SPD

Code for **Ask in the Crowd: Differentially Private LLM Inference via Dummy-Augmented Shuffling** (NeurIPS 2026).

DP-SPD perturbs a shared query prefix, generates prefix-conditioned dummy suffixes on the client, perturbs and shuffles the candidate queries, and recovers the response for the real query. The repository includes embedding-access and token-projection experiments, privacy attacks, baseline adapters, and a shared-prefix KV-cache benchmark.

## Repository layout

| Path | Contents |
| --- | --- |
| `dpspd/` | Client pipeline, privacy noise, shuffling, response selection, and server model wrapper |
| `experiments/classification/` | SST-2, QNLI, and IMDb utility |
| `experiments/generation/` | WikiText-2, Alpaca, Dolly-15k, and CNN/DailyMail utility |
| `experiments/api_only/`, `experiments/wildchat/` | Token projection, black-box utility, and WildChat evaluation |
| `experiments/proxy_scale/`, `experiments/speed/` | Continuation-model scale and KV-cache efficiency |
| `attack/` | Query identification, embedding inversion, and GPT-based attacks |
| `baseline/` | Adapted comparison methods and aggregation scripts |
| `scripts/` | Experiment launchers |

## Installation

Use Python 3.10 or newer. Install a PyTorch build matching your CUDA runtime, then install the remaining dependencies. The tested `myenv` versions and CPU smoke results are recorded in [ENVIRONMENT.md](ENVIRONMENT.md).

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you already use Conda, activate your environment and install the same requirements with `pip install -r requirements.txt`.

The default experiment paths are relative to the repository root. Obtain the model checkpoints and datasets under the corresponding locations, or override paths through launcher environment variables or Python CLI options:

```text
models/Llama-2-7b-chat-hf
models/bert-base-uncased-sst2
models/bert-base-uncased-qnli
models/Qwen2.5-0.5B-Instruct
models/Qwen3-14B
models/Qwen3-32B
data/imdb
data/glue/sst2
data/glue/qnli
data/wikitext/wikitext-2-v1
data/alpaca
data/databricks_dolly
data/cnn_dailymail
data/wildchat/
```

Most evaluation scripts load from local files by default. Prepare Hugging Face datasets with their expected splits and columns, for example IMDb `train`/`test`, GLUE SST-2 `train`/`validation`, WikiText-2 `train`/`test`, and CNN/DailyMail `article`/`highlights`. Model weights, datasets, and generated outputs are excluded from version control. Check each upstream model and dataset license before obtaining or redistributing them.

## Quick start

Run from the repository root. Inspect parameters with `python <entrypoint> --help`.

```bash
# Classification: Table 1 and the IMDb sweep
bash scripts/run_classification.sh sst2 --dry_run true
bash scripts/run_classification.sh sst2 --transport_mode token --max_eval_samples 1000
bash scripts/run_classification.sh imdb --max_eval_samples 1000

# Generation: WikiText-2 and instruction datasets
bash scripts/run_generation.sh wikitext --dry_run true
bash scripts/run_generation.sh wikitext --max_samples 300
bash scripts/run_generation.sh alpaca --max_samples 300

# CNN/DailyMail with Qwen3-14B
bash scripts/run_generation.sh cnndm --dry_run true
```

`--dry_run true` shows the resolved configuration; the CNN/DailyMail entry also checks model and dataset paths. Specify `MODEL_PATH`, `DATASET_PATH`, and `DEVICE` when using different locations or hardware. Outputs are written under `outputs/` by default.

## Paper experiment map

| Paper result | Main entrypoint |
| --- | --- |
| Table 1, Figure 2: classification | `scripts/run_classification.sh`, `scripts/run_table1_dpspd_token.sh` |
| Figures 3–4: generation and comparison methods | `scripts/run_generation.sh`, `scripts/run_figure4_baseline_generation.sh`, `scripts/run_qwen3_cnndm_baselines.sh` |
| Tables 2–5: stronger models, WildChat, and privacy attacks | `scripts/run_wildchat_dpspd_blackbox_sweep.sh`, `scripts/run_attack.sh`, `scripts/run_wildchat_gpt4o_mini_attack_from_groups.sh` |
| Figure 5: continuation-model scale | `scripts/run_proxy_generator_scale_qwen25.sh` |
| Figure 6: shared-prefix KV-cache timing | `scripts/run_speed_benchmark.sh` |
| Appendix: API-only projection and interface gap | `scripts/run_api_only_projection_ablation.sh`, `scripts/run_whitebox_blackbox_gap.sh` |

The shell launchers include experiment-specific budgets and model settings. Read the matching script before a full sweep, especially its GPU assignment and sample count. Some methods in `baseline/` are **style adapters** for the paper's common LLaMA/Qwen evaluation harness rather than complete reproductions of the original methods. Their individual READMEs describe the adapted protocol and its limits. The SnD result in Table 1 has no implementation in the supplied source tree and cannot be reproduced from this repository.

## Privacy and evaluation notes

- The prefix and suffix use separate perturbation budgets. Candidate shuffling applies to suffix branches; the shared prefix is privatized separately. The argument `--target_epsilon` is a reporting label in some entrypoints and does not itself change local noise parameters.
- Embedding mode assumes access to the server's input embeddings. API-only mode projects noisy embeddings to tokens before sending text to a text-only backend. Keep these transport modes separate when comparing results.
- Attack scripts create deterministic evaluation manifests in `fixed_eval_sets/` from locally supplied datasets. These manifests contain source text and are ignored by git.
- Hosted API experiments read credentials from environment variables such as `OPENAI_API_KEY`. Never put keys in scripts or committed files. GPT-based attack prompts in the attack code are part of the evaluation protocol.
- The default experiments use independent single-query evaluations. The paper discusses the limits for repeated queries and long-context interactions.

## Citation

Use [CITATION.cff](CITATION.cff) or cite the paper:

```bibtex
@inproceedings{liu2026ask,
  title={Ask in the Crowd: Differentially Private LLM Inference via Dummy-Augmented Shuffling},
  author={Liu, Zhihao and Guo, Zixiong and Shao, Shuo and He, Yu and Wang, Wenli and Chen, Meihui and Wang, Di and Hu, Yuke},
  booktitle={Advances in Neural Information Processing Systems},
  year={2026}
}
```
