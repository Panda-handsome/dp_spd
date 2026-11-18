# IMDb Figure-2 baseline adapters

`run_imdb_classification.py` evaluates CAPE, CusText, DP-Forward, InferDPT,
SanText, and TextObfuscator under the Figure-2 IMDb evaluation
protocol:

- local IMDb `train` / `test`, both shuffled with seed 42 before selection;
- frozen local LLaMA-2-7B-Chat, 128 BPE tokens, final active hidden state;
- all 25,000 privatized IMDb train representations and all 25,000
  independently privatized IMDb test representations (the legacy drivers pass
  `--downstream_task_train_size -1 --downstream_task_test_size -1`);
- the legacy `EnhancedClsModel`: `4096 -> 2048 -> 2`, ReLU, SGD (`lr=1e-3`,
  momentum `0.9`), 50 epochs, batch 256, reporting its best test accuracy
  across epochs exactly as the previous Figure-2 driver did.

This is a controlled LLaMA feature-probe comparison, not a retroactive claim
that the original papers reported the same classifier task.  The emitted CSV
records both the reproduced mechanism and any omitted original component:

| Method | Retained mechanism | Required IMDb/LLaMA adaptation |
| --- | --- | --- |
| CAPE | min--max representation + Laplace noise | no IMDb private-attribute labels exist for CAPE adversarial training |
| CusText | top-k exponential replacement | BPE input-token mapping; local Counter-Fitting word vectors are absent |
| SanText | exponential token sanitization | BPE vocabulary from raw IMDb train text instead of word/BERT setup |
| InferDPT | RANTEXT | server-only feature probing; no remote LLM/local extractor pair |
| DP-Forward | official analytic Gaussian + clipping | frozen feature probe rather than noisy fine-tuning |
| TextObfuscator | split boundary Laplace noise | frozen LLaMA split rather than jointly trained RoBERTa/prototypes |

For a short CUDA smoke test, reduce the data and expensive mapping domains:

```bash
cd dp_spd
CUDA_VISIBLE_DEVICES=0 python baseline/run_imdb_classification.py \
  --method santext --epsilon 2.0 --device cuda:0 \
  --max_train_samples 8 --max_eval_samples 8 --max_length 32 --batch_size 2 \
  --classifier_epochs 2 --classifier_batch_size 4 --santext_mapping_max_samples 16
```

Do not reduce `--inferdpt_candidate_vocab_size` or mapping sizes for the
reported sweep.  Those options only make the perturbation smoke test fast.

Before launching several full SanText or InferDPT epsilon jobs, build their
shared IMDb-train vocabulary exactly once with `--prepare_mapping_only true`.
