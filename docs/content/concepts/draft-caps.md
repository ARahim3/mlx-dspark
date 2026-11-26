---
title: Draft caps and calibration
nav_title: Draft caps and calibration
description: How many tokens to draft per round has no universal answer. It depends on your chip, the model's precision and the MLX version, so mlx-dspark measures it on your Mac.
scripts: widgets
---

The **cap** is how many drafted tokens get verified per round. More drafts can mean more tokens per pass, but every extra verified row costs something, and the next draft is less likely to survive than the last. The best cap is where those two curves cross.

## Play with it

This is the cost model the engine reasons with, in miniature. `p` is how often each drafted token survives; the curve is how verification cost grows with the number of rows verified. The shapes are illustrative, the logic is real:

<div class="costmodel" data-costmodel markdown="0">
  <div class="cm-controls">
    <label>How often each draft is right <output data-cm-p>80%</output>
      <input type="range" name="cm-p" min="50" max="95" step="1" value="80">
    </label>
    <label>What the drafter costs <output data-cm-d>12% of a step</output>
      <input type="range" name="cm-d" min="4" max="40" step="1" value="12">
    </label>
    <fieldset class="seg">
      <legend>How verify cost grows with rows</legend>
      <label><input type="radio" name="cm-curve" value="flat" checked><span>Flat to 8 rows (8-bit, current kernels)</span></label>
      <label><input type="radio" name="cm-curve" value="rising"><span>Rising, then flat (4-bit)</span></label>
      <label><input type="radio" name="cm-curve" value="steep"><span>Steep (mixture of experts)</span></label>
    </fieldset>
  </div>
  <div class="cm-chart" aria-live="polite"></div>
  <p class="cm-readout"></p>
</div>

In formula form, for a cap `C`:

```text
speedup(C) ≈ accepted(C) / (drafter + verify(C + 1))

accepted(C) = 1 + p + p² + … + p^C      (tokens committed per round)
verify(w)   = cost of one pass over w rows, in single-token steps
```

On a datacenter GPU, verifying 8 rows costs about the same as 1, so the answer is "draft a lot". On Apple Silicon the verify curve has real shape, and that shape is a property of the chip × the model's quantization × the MLX version × the kernels in use. It has moved four times in this project's life: MLX 0.32's matmul kernels, then mlx-dspark's own verify kernel in v0.12.0, then its replacement in September 2026, which moved every re-measured 8-bit model from cap 4 to cap 7, then MLX 0.32.5's own few-row kernels, which made widths 4–8 about 10% cheaper on an M4 Pro and removed a cliff at widths 7–8 on M5.

## How mlx-dspark picks it

**With no `--max-draft`** (the default), the first run of each model measures this Mac's curves: verify cost at each width, the drafter's cost, and how verify cost grows with context depth. That takes about 5–20 seconds and is cached per model, quantization and MLX version. The engine derives the best fixed cap from those curves. That's the "derived cap" every published number uses.

**`--max-draft auto`** picks the cap *every round* instead, from the measured curves, live acceptance and observed round times. It can also **park** speculation entirely (plain decoding with periodic probe rounds) on content where speculation would lose, which is why it is the recommended setting for [Bonsai](/models/ternary-bonsai-27b/), a pair that hovers around break-even. Elsewhere it is not the faster choice: in our measurements it lands at the derived cap on some pairs and a few percent under it on others (up to 12% under on one), because it spends rounds exploring caps the derived one already ruled out. Use the default unless you have a reason not to.

**`--max-draft N`** pins it. A cap you set is never overridden, not even by the depth adjustment below. Pin only a value you've measured on this Mac.

**At long context** the derived cap is depth-aware: it shrinks the verify width when the measured depth cost says a narrower verify pays. Below about 4k tokens it's a no-op. See [Long context](/concepts/long-context/).

### Don't copy caps from someone else's Mac

The caps in the results tables are an M4 Pro's. A faster, higher-bandwidth chip can land *lower*, not higher: an M5 Max may pick cap 2 for the 8-bit 27B where the M4 Pro picks 7, because cheaper verification shifts the optimum, and forcing 7 there is a small net loss. Prefer no flag, or `--max-draft auto`.

## The confidence threshold

`--confidence-threshold 0.3` lets the drafter's confidence head stop drafting early in a round. It pays only when **both** hold:

1. the verify curve still rises inside the cap's window, so a shorter verify actually saves time; and
2. the drafter leaves acceptance headroom, meaning it's often wrong partway through the block, so there's something to save.

On Qwen3.6-35B-A3B (a steep MoE curve, acceptance swinging from 2.8 on chat to 7.0 on math) it lifts 1.27× to 1.32×. On most 8-bit models the curve is flat, and it measures slightly worse. It's drafter-specific as well as quantization-specific, so don't paste it across pairs.

## The kernels behind the curve

The command line and server install two custom Metal kernels, each enabled per shape only after a one-time probe on your Mac proves it both faster and numerically sound. Everything else stays on MLX's kernels.

Small-M verify kernel
: MLX's quantized matmul re-reads the whole weight matrix *per row* at the few-row widths verification uses. This kernel runs the matmul on the GPU's matrix units with the weights as the shared operand, so each 4- or 8-bit weight group is read once for up to 16 rows (widths 5–16 at 4-bit, 6–16 at 8-bit). On Qwen3.8-27B 4-bit, verifying 5–8 rows now costs about 1.4× a single decode step. Since MLX 0.32.5 MLX's own kernels do this for most layers and are faster, so the probe keeps this one only for the narrow layers MLX's route skips. `--no-small-m` turns it off for comparisons.

Multi-row attention kernel
: MLX's decode attention gives every query row its own pass over the KV cache, so a width-8 verify at 32k context reads the cache about 8 times. This kernel reads each KV tile once for all rows: 1.5–4× on those attention calls, about 1.2× on a whole 32k-deep 27B verify. The probe decides per draft width, and separately for tree verifies: MLX 0.32.5 matches the kernel on ordinary verify calls at some widths, while its masked path, which every tree verify takes, is still 3–4× slower there. `--no-multirow-attn` turns it off.

**On M5 the small-M kernel is off.** It wins the probe's micro-benchmark there but can stall a sustained generation for about 105 seconds, which a micro-benchmark can't detect, and since MLX 0.32.5 MLX's own kernels cover those rows on M5 anyway. The attention kernel and the tree-verify kernel are on there since v0.21.4, verified end to end on an M5 Pro by a contributor. GPUs newer than M5 start with all three off; `MLX_DSPARK_FORCE_SMALL_M=1`, `MLX_DSPARK_FORCE_MULTIROW=1` and `MLX_DSPARK_FORCE_TREE_KERNEL=1` override the gates for a comparison run. If you have one, paired numbers are very welcome in an issue.

**On M1 and M2 the GPU accepts fewer threads per launch** for these kernels than M3/M4 do. The probes notice when a launch is refused and run the kernel at a size the GPU accepts, or leave that shape on MLX's own path. They never stop the model from loading.

Output stays greedy-correct with either kernel: the model still verifies every token. Token ids can differ from MLX's kernels only at floating-point ties.

## The calibration cache

Measurements live in `~/.cache/mlx_dspark/calibration.json`, keyed by model, quantization, MLX version and kernel context. After an MLX upgrade each model re-measures once by itself. `GET /calibration` on a running server returns the curves in use.

The measurement is one-shot, so a curve taken on a hot GPU (right after a long benchmark, say) reads high and can pick a too-low cap. If a model's derived cap looks off, delete its entries from `calibration.json` and run it again from cold.
