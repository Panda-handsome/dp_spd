# CusText / CusText+ baseline

This directory is a clean-room reproduction of Chen et al., *A Customized
Text Sanitization Mechanism with Differential Privacy* (Findings of ACL 2023).
It was checked against the [ACL Anthology paper](https://aclanthology.org/2023.findings-acl.355/)
and the authors' [public implementation](https://github.com/sai4july/CusText).
The original code and embeddings are not vendored here.

## What is reproduced

For each input word, CusText constructs a small semantic output set of size
`K` and uses the exponential mechanism to replace the word.  This
implementation follows the paper/release details:

- cosine similarity for Counter-Fitting vectors, or Euclidean distance for
  GloVe vectors;
- per-input-row utility normalization, followed by
  `softmax(epsilon * utility / 2)`;
- `release_compatible`, which mirrors the public repository's default
  mutable-vector masking procedure; use this when matching its utility runs;
- strict `conservative`, which consumes output candidates so group supports do
  not overlap and is the appropriate setting for a formal shared-support DP
  statement;
- `aggressive` as a release-compatible ablation only; its overlapping supports
  should not be described as the conservative DP construction;
- CusText+ (`--variant custext_plus`) preserves the NLTK English stopword list;
- the released code's whitespace splitting and numeric fallback are available
  through `--numeric_policy release_compatible`.

The `word_vectors` mode is the paper-faithful mode.  Like the authors' release,
it builds the input mapping from words observed in the selected mapping corpus;
unmapped held-out words are released unchanged.  Every result reports this
coverage, so that fact cannot be hidden in an aggregate utility number.

`model_tokens` is a separate, clearly labelled adaptation for a direct
fixed-model DP-SPD comparison: it partitions every non-special BERT token ID
using the target model's input embedding table and substitutes IDs before
forward propagation.  It retains the CusText group/exponential-mechanism
construction, but it is not a word-level reproduction and must not be labelled
as the original paper result.  CusText+ is deliberately unavailable in this
mode because stopword preservation has no unambiguous WordPiece equivalent.

At the token level, `epsilon` is the budget of one replacement mechanism over
its mapped domain.  A document-level statement requires the usual composition
accounting for the private token changes in that document; do not directly
equate this number with DP-SPD's end-to-end shuffled-inference budget.

## Protocols

`--protocol paper` (default) sanitizes train and evaluation examples, then
fine-tunes a new BERT classifier for three epochs.  This matches the original
utility protocol.

`--protocol inference` sanitizes only evaluation inputs and runs an existing
fine-tuned classifier checkpoint.  Use this mode for a like-for-like comparison
with the current DP-SPD SST-2/QNLI fixed-server experiments.  These two
protocols make different retraining assumptions and must be reported in
separate tables or clearly separated columns.

By default `--mapping_scope train` avoids looking at validation text while
building the mapping.  The public SST-2 release combined train and dev for
that step; `--mapping_scope train_eval` is available only for a release-style
ablation and should be declared as such.

## Running SST-2 and QNLI

Run from the repository root.  The paper uses Counter-Fitting vectors with cosine
similarity; pass a compatible plaintext vector file (`word value1 ...`, with an
optional `<count> <dimension>` header).

```bash
python baseline/custext/run_classification.py \
  --task sst2 \
  --protocol paper \
  --representation word_vectors \
  --embedding_path /path/to/counter-fitted-vectors.txt \
  --metric cosine \
  --epsilon 1.0 --top_k 20 \
  --mapping_strategy release_compatible \
  --save_model true
```

CusText+ changes only the variant flag:

```bash
python baseline/custext/run_classification.py \
  --task qnli \
  --protocol paper \
  --representation word_vectors \
  --embedding_path /path/to/counter-fitted-vectors.txt \
  --variant custext_plus \
  --epsilon 1.0 --top_k 20
```

For the fixed-server inference comparison that needs no external word-vector
file, run the adaptation explicitly:

```bash
python baseline/custext/run_classification.py \
  --task sst2 \
  --protocol inference \
  --representation model_tokens \
  --model_path models/bert-base-uncased-sst2 \
  --epsilon 1.0 --top_k 20 \
  --max_eval_samples 1000
```

The first run writes a reusable mapping below
`outputs/baselines/custext/mappings/`; it contains candidate IDs/words and
sampling probabilities but not the embedding vectors.  A subsequent run can
reuse that cache.  Use `--force_rebuild_mapping true` after changing the
mapping corpus or representation.  `--dry_run true` validates the selected
configuration without loading data or models.

The result CSV includes `protocol`, `representation`, mapping-group size,
mapped-token fraction, preserved stopword count, and accuracy.  Keep these
columns when making the comparison table: in particular, a low coverage value
means the word-vector baseline is leaving part of the held-out text unchanged.

## WikiText-2 generation baselines

`run_generation.py` retains the original **word-level** CusText mechanism on
LLaMA-2-7B-Chat. It is useful for a separate textual-CusText study, but it
must not be used in Figure 3: replacing a word can change its LLaMA BPE
length, so a full sanitized word sequence cannot align position-by-position
with the clean labels used by DP-SPD's PPL calculation.

For the direct Figure-3 comparison, use `run_generation_figure3.py` through
`scripts/run_custext_generation_sweep.sh`. It applies the same CusText
candidate-group/exponential-mechanism construction to the LLaMA input-token
embedding vocabulary. Every active **non-special BPE token** in the complete
256-token input is sampled and replaced; the runner then calls the unchanged
DP-SPD `calculate_ppl` and `calculate_rouge` paths. Its CSV checks and records
that mapped fraction is 1.0 over eligible tokens, as well as the realized
replacement fraction (a sampled output may legitimately equal its input).

This is necessarily a `CusText-style` / `token-level CusText adaptation`, not
the original word-level paper implementation. Label it that way in Figure 3.
It has no dummy ratio `r` or dummy count `k`; plot one horizontal baseline per
epsilon across the `k` axis rather than fabricating `r,k` values. Its epsilon
is a **per-token** exponential-mechanism budget, rather than DP-SPD's shuffled
end-to-end budget.

The public Counter-Fitting vector file is expected at
`data/embeddings/counter-fitted-vectors.txt` when running from the repository root.
On this machine, activate the project's `dpllm` Conda environment first: the
base environment's PyTorch build requires a newer NVIDIA driver and cannot
initialize CUDA.

```bash
source .venv/bin/activate
```

Run a four-GPU Figure-3 smoke test as follows:

```bash
MAX_SAMPLES=8 \
bash scripts/run_custext_generation_sweep.sh
```

To reproduce the older `dp_generate/results.py` WikiText-2 protocol exactly,
use its `len(text)>50` selection rule, its 300 examples, batch size 32, and
non-stemmed ROUGE:

```bash
SELECTION_PROTOCOL=legacy_dp_generate ROUGE_USE_STEMMER=false \
MAX_SAMPLES=300 MAX_LENGTH=256 BATCH_SIZE=32 \
CONTEXT_LEN=32 MAX_NEW_TOKENS=32 \
OUTPUT_DIR=outputs/baselines/custext/wikitext_legacy_dp_generate_full \
bash scripts/run_custext_generation_sweep.sh
```

The `selection_protocol` and `rouge_use_stemmer` fields in the resulting CSV
are part of the experimental record; do not combine rows that differ in either
field.

For the final setting, retain `top_k=20`, cosine similarity, and the selected
mapping strategy (the default `release_compatible` mirrors the authors'
public-code grouping), then run with `MAX_SAMPLES=300`. One CSV and log file
are written for each of epsilon 1.5, 2.0, 2.5, and 3.0 under
`outputs/baselines/custext/wikitext_figure3/`.
