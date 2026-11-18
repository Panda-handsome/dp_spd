# CAPE baseline

This directory contains a clean-room reproduction of the perturbation part of
Plant, Gkatzia and Giuffrida, *CAPE: Context-Aware Private Embeddings for
Private Language Learning* (EMNLP 2021).  It was checked against the
[paper](https://aclanthology.org/2021.emnlp-main.628/) and the authors'
[official repository](https://github.com/NapierNLP/CAPE).

## Original CAPE protocol

CAPE is **not a text-generation method**.  Its released TensorFlow code:

1. freezes BERT and mean-pools the contextual sequence hidden states into one
   review representation;
2. min--max normalises that representation to `[0, 1]` and adds coordinatewise
   `Laplace(1 / epsilon)` noise (the paper configures `Delta_f = 1`);
3. learns rating and private-attribute classifiers.  A gradient reversal head
   prevents the representation from exposing a supplied private attribute.

The paper trains on UK Trustpilot reviews, whose labels include a rating and
gender, location, and birth-year-derived private attributes.  It reports
classification Accuracy/F1 and attacker Accuracy/F1.  It has no WikiText-2,
causal-LM, PPL, ROUGE-L, or frozen-LLaMA protocol.

## Local WikiText-2 adapter

`run_generation_figure3.py` is a deliberately narrow **CAPE-DP-only-style
LLaMA adapter**.  It uses the paper's min--max and `Laplace(1 / epsilon)`
operation on every active LLaMA input-embedding coordinate in a sequence, and
then invokes the unmodified DP-SPD PPL/ROUGE-L functions with clean labels.

This allows an auditable stress test under the same model, rows, context and
metrics as Figure 3, but it is not full CAPE:

- WikiText-2 does not provide CAPE's protected gender/location/birth-year
  labels, so no adversarial head can be trained without inventing a different
  private task.
- The paper perturbs one mean-pooled *contextual BERT* vector, whereas this
adapter must retain a causal sequence and perturbs LLaMA input embeddings.
- Repeating the paper's `1 / epsilon` coordinate scale over an entire sequence
  does not establish the paper's record-level epsilon-LDP guarantee.  The CSV
  explicitly records this as a noise-scale parameter rather than an end-to-end
  privacy claim.

The inverse-CDF sampler excludes only the float32 endpoint that would make
`log(0)` and produce an artificial infinite noise value.  It does not clip any
finite Laplace draw.  Result CSVs record this numerical guard.

Consequently, do **not** call the output “CAPE” in a paper table and do not
place it on the main DP-SPD Figure-3 PPL curve.  If retained, label it
`CAPE-DP-only-style (LLaMA sequence adapter)` and treat it as an appendix
diagnostic.  The appropriate faithful CAPE comparison is a private-attribute
classification experiment on a dataset that supplies those attributes.

## Run

From `dp_spd/` with the project dependencies installed:

```bash
# Four-GPU smoke test
MAX_SAMPLES=4 bash scripts/run_cape_generation_sweep.sh

# Same 300-sample scale as the local Figure-3 harness
MAX_SAMPLES=300 bash scripts/run_cape_generation_sweep.sh
```

The sweep assigns epsilon `{1.5, 2.0, 2.5, 3.0}` to CUDA `{4, 5, 6, 7}` and
writes one CSV per epsilon to `outputs/baselines/cape/wikitext_figure3_laplace_guard_v2/`.
