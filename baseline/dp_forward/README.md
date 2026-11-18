# DP-Forward-style frozen-LLaMA generation adapter

This is a scoped adaptation for the DP-SPD WikiText-2 Figure-3 evaluation,
not a claim that the original DP-Forward paper reports language-model
perplexity or ROUGE-L.

## Original protocol

DP-Forward protects a sequence in the forward pass: it clips the
representation matrix to a fixed Frobenius norm and applies an analytically
calibrated Gaussian mechanism for `(epsilon, delta)` local DP. It evaluates
**jointly fine-tuned BERT classifiers** (SST-2, QQP, MNLI), and can perturb
input embeddings, encoder sublayers, or the task output. Its release command
uses `norm_c=1`, `delta=1e-5`, `local_dp=True`, `noise_position=encoder`,
`noise_layer=10`, `encoder_sub_noise_position=add_and_norm_2`, and enables
noise at inference. The public release's `add_noise` implementation clips a
sequence then adds an i.i.d. Gaussian matrix with the release's analytic-GM
standard deviation.

Paper: <https://arxiv.org/abs/2309.06746>. Official release:
<https://github.com/xiangyue9607/DP-Forward>.

## This Figure-3 adaptation

- Frozen local `Llama-2-7b-chat-hf`; no fine-tuning is invented for WikiText-2.
- The default `noise_layer=28` means noise after LLaMA decoder layer 28
  (29/32 layers processed), matching the release's `11/12` deep encoder
  position as closely as possible. The remaining three layers are the server.
- Every active contextual token representation is clipped jointly per
  sequence to `norm_c=1` and receives independent Gaussian noise.
- `epsilon` and `delta=1e-5` use the exact scalar calibration in the official
  `dp_noise.py` release. For the default budgets, standard deviations are
  `2.582564, 1.993812, 1.634002, 1.390593`.
- PPL uses clean next-token labels after a noised full representation; ROUGE-L
  uses a noised 32-token prefix and clean 32-token reference with the same
  Evaluate ROUGE/stemmer definition as DP-SPD.

The CSV method name is `DP-Forward-style-frozen-LLaMA-deep-boundary` and
records omitted noisy fine-tuning, classifier head, and label randomized
response. Do not label it as the original DP-Forward classification result or
as an end-to-end guarantee for a frozen LLaMA not trained under the mechanism.

## Run

From `dp_spd/`:

```bash
# Four-GPU smoke test
MAX_SAMPLES=4 bash scripts/run_dp_forward_generation_sweep.sh

# Figure-3-sized run
MAX_SAMPLES=300 bash scripts/run_dp_forward_generation_sweep.sh
```

The script maps epsilon `1.5, 2.0, 2.5, 3.0` to CUDA `4, 5, 6, 7`; its
defaults match the shared Figure-3 model/data/length/batch protocol.
