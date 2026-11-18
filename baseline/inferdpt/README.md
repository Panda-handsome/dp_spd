# InferDPT / RANTEXT baseline

This is a clean-room reproduction of the RANTEXT perturbation mechanism and
the two-stage InferDPT pipeline from Tong et al., *InferDPT: Privacy-Preserving
Inference for Closed-Box Large Language Models* (TDSC 2025). It was checked
against the [paper](https://tianweiz07.github.io/Papers/25-tdsc-3.pdf) and the
authors' [official implementation](https://github.com/mengtong0110/InferDPT).

## What the original work does

InferDPT has two stages:

1. RANTEXT perturbs each advertised-vocabulary BPE token of the private
   document. For every input occurrence it adds coordinate-wise Laplace noise
   to the embedding, uses the resulting radius to construct a **random**
   adjacency list, and samples one output from it with the exponential
   mechanism.
2. A black-box remote LLM receives only the perturbed prompt. A local,
   smaller extraction LLM receives the raw prefix plus the perturbed
   generation, then writes a continuation aligned with the raw prefix.

The original setup is GPT-4 remotely, the first 11,000 English `cl100k_base`
tokens represented by `text-embedding-ada-002`, and a local 4-bit 7B model.
It evaluates Wikitext-103/CNN-DailyMail with diversity, MAUVE and coherence;
it does **not** report causal PPL or ROUGE-L on frozen LLaMA-2.

The public code contains the RANTEXT perturbation program and a manual
extraction prompt, but not the provided embedding JSON files or an automated
extractor. Its exact `epsilon == 2` branch divides by zero; this reproduction
uses the finite `epsilon <= 2` branch required by Eq. (13), and records that
choice in source comments.

## Local LLaMA generation adapter

`run_generation.py` uses your local LLaMA-2-7B-Chat as a reproducible test
harness. It keeps the InferDPT data flow: the **remote role** receives only
the RANTEXT-perturbed BPE IDs, then the **local extractor role** sees the raw
prefix plus remote perturbed output. To fit one 48-GB GPU per epsilon, the
same frozen LLaMA instance performs both roles sequentially.

The original cl100k/ada-002 pair is unavailable locally and does not match a
LLaMA remote tokenizer. The adapter therefore constructs a public vocabulary
of the 11,000 most frequent non-special LLaMA BPE IDs in WikiText-2
train/validation, and uses LLaMA's input embedding table as `phi`. This is a
necessary **representation adaptation**, not an original GPT-4 result. The
CSV contains the vocabulary source, coverage and random-adjacency statistics.
Tokens outside this public `V` are discarded—matching RANTEXT Algorithm 2,
not silently released unchanged.

`rouge_l` is the final extraction output versus the clean continuation and
uses the same `calculate_rouge_from_texts` / Evaluate ROUGE implementation as
DP-SPD. `extractor_conditional_ppl` is separately provided only as a
diagnostic: it teacher-forces the clean continuation after the extraction
prompt. It is **not directly comparable** to DP-SPD Figure-3 PPL, because
InferDPT's local extractor is deliberately allowed to see the raw prefix.

Use the final `rouge_l` as the local-LLaMA generation quality result. Do not
place `extractor_conditional_ppl` in the Figure-3 server-PPL curve or label
this row as the published GPT-4 InferDPT result.

## Run

Run from the repository root in an environment where `python` can access CUDA:

```bash
source .venv/bin/activate
```

Four-GPU smoke test:

```bash
MAX_SAMPLES=4 REMOTE_MAX_NEW_TOKENS=8 MAX_NEW_TOKENS=8 \
bash scripts/run_inferdpt_generation_sweep.sh
```

Full local-LLaMA run:

```bash
MAX_SAMPLES=300 bash scripts/run_inferdpt_generation_sweep.sh
```

The script assigns epsilon `{1.5, 2.0, 2.5, 3.0}` to CUDA `{4, 5, 6, 7}` and
writes results to `outputs/baselines/inferdpt/wikitext_local_llama/`.
