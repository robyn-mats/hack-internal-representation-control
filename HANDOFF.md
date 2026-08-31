# Handoff — 2026-08-31

State of the experiment at the point the RunPod GPU pod was released. Written so
a fresh session can continue without re-deriving anything. Design rationale is in
`PLAN.md`, commitments in `PREREGISTRATION.md`, findings in `NOTES.md`; this file
is only *where things stand and what to do next*.

## ⚠ Before anything else

**Terminate the POD, keep the VOLUME.** `robyn_retail_salmon_elephant`
(`gmqfdy4bo2`, 300 GB, US-KS-2) holds ~11 GB of activation files that cost 15 GPU
hours to produce and cannot be regenerated cheaply, plus 123 GB of model weights.
Pods and network volumes are separate resources; releasing the pod does not touch
the volume. There is also a second, unattached 300 GB volume
(`key_indigo_harrier_volume`, EUR-IS-4) that appears unused — check before
deleting, but it is billing.

**Activation files do not travel.** `artifacts/` is gitignored and large. The
measure stage is what converts activations into a few-MB parquet that *can* be
committed. Anything needing raw activations needs the volume remounted.

## Done and frozen

| | |
|---|---|
| Carriers | 7, screened from the paper's 50 (43 were contaminated) |
| Conditions | 67 phrasings / 25 cells, `irc/conditions.csv` |
| Concepts | 50; split 10 pilot / 40 held out, stratified, seeded |
| Scaffold | tagged form, amended 2026-08-31 — deviation 6.81% → 0.04% |
| Stimuli | `stimuli.csv`, 23,450 rows / 23,107 prompts |
| Readouts built | concept vectors (`word_tokens`) + 50 latent selections at 16k |

**Runs** (all in `artifacts/runs/`, all with `event: completed` records):

- `pilot2/generated` — 4,627 trials, 2 non-exact
- `pilot2/teacher_forced` — 4,627 trials
- `heldout1/generated` — 18,487 trials, 33 non-exact (0.18%; 0.055% excluding
  the concept `information`, see `NOTES.md`)
- `heldout1/teacher_forced` — the 33 deviant trials only
- `pilot1/*` — **pre-amendment**, collected under the old scaffold. Not
  confirmatory data. Keep: its deviation rates are the scaffold-dependence result.

**Stage 2 so far:** pooling rule is **`token_mean`** (decided; rule 3 is
degenerate for a copying task, `topk_mean` dropped for selecting on the measured
quantity). **The analysis layer is still open** — it needs the pilot measured.

## Next steps, in order

**1. On the pod, before releasing it — CPU, ~20 min.** The only step that must
happen while the volume is mounted:

    python3 rk_scripts/09_measure.py --run-id pilot2 --pass generated --device cpu

Then **commit `artifacts/runs/pilot2/generated/results/*.parquet`** — they are
small and are what makes everything below portable. (`artifacts/` is gitignored,
so add them with `git add -f`.)

**2. Off-pod — write the Stage 2 chooser.** Does not exist yet. Reads the
parquet, computes A-vs-T7 separation per layer at `token_mean`, prints the four
layers, names the winner. ~50 lines. That is the last Stage 2 value.

**3. Commit the Stage 2 amendment** naming the analysis layer. Until this is
committed, held-out data must not be measured — that is the standing commitment
and the reason held-out generation was allowed to run early.

**4. Needs the volume again.** Held-out measurement:
   - 262k latent selection at the chosen layer (**GPU**, ~1 h, 11.3 GB download).
     Change `SAE_ID_TEMPLATE` in `irc/constants.py` to `width_262k` at that point,
     not before — the pilot's selection must stay reproducible.
   - held-out measure (**CPU**), then commit its parquet.
   - The held-out **concept-vector** readout needs no SAE, so the secondary
     readout is available without the 262k step.

**5. Off-pod — the confirmatory analysis.** Does not exist. 15 contrasts, average
over carriers then phrasings within cell, paired across the 40 concepts, BCa
bootstrap at 10,000 resamples, Holm across the family, **Q0 as a hard gate**. Plus
per-cell deviation/leak rates and the dilution diagnostics in `PLAN.md` §4.

## Working on a Mac rather than the pod

`POD_NOTES.md` rules do **not** apply — that file is pod-specific. On a laptop
`uv` is correct, as `CLAUDE.md` says. No GPU means: measure stages run with
`--device cpu`, and anything needing the 27B model or SAE *selection* needs the
pod back. SAE *encoding* is a small matmul and runs fine on CPU.

## Two process rules learned the hard way

- **Never edit a shell script while it is running.** Bash reads scripts by byte
  offset; rewriting one mid-run made it re-execute its launch line and start a
  redundant 18,487-trial run. See the `queue_heldout.sh` incident in the log.
- **Per-trial cost estimates do not transfer between stages.** Rule 3 was
  estimated at 2.2 h from the teacher-forced pass's 2.08 s/trial and ran in 7
  minutes at 0.09 s/trial. Measure before budgeting.

## Open questions for the writeup

- `information` is meta-linguistic and behaves differently — away-instructions
  land on the carrier text and delete words. 23 of 33 held-out deviations.
  Reported, **not excluded** (`NOTES.md`).
- The analysis layer is chosen at 16k and applied at 262k. Flat-vs-peaked layer
  curve determines how load-bearing that is.
- `pilot1` vs `pilot2` is a real scaffold-dependence result: identical
  instructions, 6.81% vs 0.04% deviation.
