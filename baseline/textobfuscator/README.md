# TextObfuscator-style frozen-LLaMA generation adapter

This directory is a deliberately scoped adaptation for the DP-SPD WikiText-2
generation protocol; it is not a claim that the original TextObfuscator paper
reports PPL/ROUGE-L or a drop-in reproduction of its classification results.

## Paper versus this adapter

Zhou et al., *TextObfuscator* (Findings ACL 2023), split a **12-layer
RoBERTa-base classifier** after client layer 3. The client sends every token
hidden state at that boundary after adding independent Laplace noise with
scale `1 / eps` in the released code. Client and server are jointly trained
on NER/SST-2/AGNews with task loss plus prototype centre-close and
prototype-separation losses. For sentence tasks, contextual states are
KMeans-clustered (`k=100`) and re-divided with task TF-IDF. It evaluates
classification utility and privacy attacks, not language generation.

Paper: <https://aclanthology.org/2023.findings-acl.337/>. Official code:
<https://github.com/zhouxiaolu1997/TextObfuscator>.

This frozen-LLaMA adapter:

- splits Llama-2-7B-Chat after three decoder layers and applies
  coordinate-wise `Laplace(1 / epsilon)` to **every active LLaMA BPE token**
  at that boundary;
- uses clean next-token labels for PPL and a 32-token private context followed
  by a clean 32-token reference for ROUGE-L, following the DP-SPD Figure-3
  evaluator definition;
- builds one public WikiText-2 frozen-client KMeans cache (`k=100`) as an
  audit only. It does **not** alter logits, because the required task labels,
  task-specific TF-IDF redivision, and joint task/prototype training do not
  exist for this generation setting.

Rows are named `TextObfuscator-style-frozen-LLaMA-split` and record every
omitted component. They should be presented as a clearly marked adapted
baseline, not as original TextObfuscator or a formal end-to-end DP guarantee.

## Run

From `dp_spd/`, build the shared public prototype audit once:

```bash
python baseline/textobfuscator/run_generation_figure3.py \
  --model_path models/Llama-2-7b-chat-hf \
  --dataset_path data/wikitext/wikitext-2-v1 \
  --epsilon 1.5 --device cuda:4 --build_prototypes_only true
```

Then smoke-test all four epsilon values:

```bash
MAX_SAMPLES=4 bash scripts/run_textobfuscator_generation_sweep.sh
```

Run the Figure-3-sized evaluation:

```bash
MAX_SAMPLES=300 bash scripts/run_textobfuscator_generation_sweep.sh
```

The sweep maps epsilons `1.5, 2.0, 2.5, 3.0` to CUDA `4, 5, 6, 7` and reuses
the same prototype cache.
