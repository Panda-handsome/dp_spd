# SanText / SanText+ baseline

This directory is a clean-room reproduction and generation-comparison adapter
for Yue et al., *Differential Privacy for Text Analytics via Natural Text
Sanitization* (Findings of ACL-IJCNLP 2021). It was checked against the
[ACL Anthology paper](https://aclanthology.org/2021.findings-acl.337/) and the
authors' [public implementation](https://github.com/xiangyue9607/SanText).
The original repository and its GloVe model are not vendored here.

## Mechanism reproduced

For an input token `x` and output candidate `y`, SanText samples

```text
P[y | x] = softmax_y(-epsilon * ||phi(x) - phi(y)||_2 / 2).
```

It is a token-wise exponential mechanism using Euclidean distances between
word embeddings. In the original code, a full probability matrix is
precomputed. The LLaMA adapter calculates precisely the required probability
rows on demand in GPU chunks, so it does not need to store a vocabulary-square
matrix. The sampling distribution is unchanged.

`SanText` sanitizes every in-vocabulary token. Its LLaMA comparison adapter
also sanitizes test-only BPE IDs by uniformly sampling from the public
reference vocabulary rather than silently preserving them. Consequently every
active, non-special token is processed and CSV rows enforce a mapped fraction
of 1.0.

`SanText+` is also available through `--variant santext_plus`. As in the
paper/release, it defines the least-frequent `w` fraction of tokens as
sensitive, always sanitizes these tokens, and sanitizes a non-sensitive token
with probability `p`; otherwise that common token is deliberately left intact.
The paper's Table-1 setting is the default here: `w=0.9`, `p=0.3`. Its formal
guarantee is UMLDP with `epsilon_0=ln(1/p)`, not ordinary LDP over every word;
keep it in a separate labelled row from SanText.

The release's literal sensitive-set boundary (`words[-int(w*len(words))-1:]`)
is retained, including its one-token boundary effect. `V` is built from the
public WikiText-2 `train,validation` splits—the train/dev analogue in the
authors' implementation—using the LLaMA tokenizer. This is an adaptation of
the original task vocabulary construction, not a use of test text.

## Figure-3 generation adapter

The original paper evaluates word-level sanitization with BERT
classification/MLM. A sanitized word can have a different LLaMA BPE length,
so a raw word-level output cannot be position-aligned with DP-SPD's clean-label
full-sequence PPL. `run_generation_figure3.py` is therefore explicitly named
and recorded as a **full-sequence LLaMA-token SanText-style adapter**:

- substitutes discrete LLaMA BPE IDs using the paper's Euclidean exponential
  mechanism on LLaMA's input embedding table;
- sanitizes every active non-special token in the 256-token sequence for
  `SanText`;
- uses the existing DP-SPD `calculate_ppl` with clean unmodified labels and
  the existing `calculate_rouge` using sanitized context / clean continuation;
- writes the selected filtering protocol, coverage, actual replacement rate,
  mapping vocabulary sizes, and SanText+ `w,p` settings into every CSV.

This makes utility measurements comparable with Figure 3, but it is **not** a
word-level result from the original SanText paper. Label the chart/table row
as `SanText-style (LLaMA-token adapter)`. The epsilon is a per-token metric
LDP (SanText) or UMLDP (SanText+) budget—not DP-SPD's shuffled end-to-end
epsilon. Like CusText, it has no dummy ratio `r` or dummy count `k`: plot one
horizontal baseline per epsilon across the `k` values, rather than creating
unreal `r,k` entries.

Run from the repository root after activating the environment that can access the local
CUDA driver:

```bash
source .venv/bin/activate
```

Four-GPU smoke test, SanText (every token sanitized):

```bash
MAX_SAMPLES=8 MAPPING_MAX_SAMPLES=256 \
bash scripts/run_santext_generation_sweep.sh
```

Full WikiText-2 result (300 selected test lines; full public train/validation
reference vocabulary):

```bash
MAX_SAMPLES=300 \
bash scripts/run_santext_generation_sweep.sh
```

The script assigns epsilon `{1.5, 2.0, 2.5, 3.0}` to CUDA `{4, 5, 6, 7}` in
that order and waits for all runs. Outputs are placed under
`outputs/baselines/santext/wikitext_figure3_santext/`.

To run the separately labelled paper-style SanText+ variant:

```bash
VARIANT=santext_plus SENSITIVE_FRACTION=0.9 \
NON_SENSITIVE_SANITIZE_PROBABILITY=0.3 \
MAX_SAMPLES=300 \
bash scripts/run_santext_generation_sweep.sh
```

To match the older `dp_generate/results.py` selection/evaluation conventions
instead of the current Figure-3 convention, use its character filter, 300
examples, batch size 32, and non-stemmed ROUGE consistently:

```bash
SELECTION_PROTOCOL=legacy_dp_generate ROUGE_USE_STEMMER=false \
MAX_SAMPLES=300 MAX_LENGTH=256 BATCH_SIZE=32 \
CONTEXT_LEN=32 MAX_NEW_TOKENS=32 \
OUTPUT_DIR=outputs/baselines/santext/wikitext_legacy_dp_generate_santext \
bash scripts/run_santext_generation_sweep.sh
```
