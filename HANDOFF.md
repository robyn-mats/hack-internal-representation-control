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
| **Stage 2: pooling rule** | **`token_mean`** (rule 3 degenerate) — commit `61ab4f2` |
| **Stage 2: analysis layer** | **40** — commit `b54ee96` |

**Runs** (all in `artifacts/runs/`):

- `pilot2/generated` — 4,627 trials, 2 non-exact. **Measured.**
- `pilot2/teacher_forced` — 4,627 trials. **Measured.**
- `heldout1/generated` — 18,487 trials, 33 non-exact (0.18%). **Measured.**
- `heldout1/teacher_forced` — the 33 deviant trials only. **Measured.**
- `pilot1/*` — **pre-amendment**, old scaffold. Not confirmatory data.

## What the pilot established

1. **The SAE readout has a severe floor effect.** 86–98% of readouts are exactly
   `0.0` depending on layer; 92% at layer 40, the best. A positive control
   (`rk_scripts/12_latent_positive_control.py`) shows this is **not** an
   instrument failure — zero SELECTION_DEAD at every layer, latents firing at
   3,000–16,000 in the context they were selected from. So the zeros are real:
   instructing the model to focus on a concept while it copies an unrelated
   sentence largely does not activate that concept's latents.
2. **The concept vector is far more sensitive** — pilot A-vs-T7 dz 2.29 at layer
   40 against the SAE's 0.71, with no zero problem. It is registered as
   *secondary*; decide how to report that before running the contrasts.
3. **Q0 holds at layers 16, 31 and 40; it fails at 53.** Layer 16 passes only
   trivially (A 1.75, T1 1.72, T7 1.57).
4. **Latent selection is ragged.** k=5 is a ceiling: 92 of 200 (concept, layer)
   cells have k<5 and 9 have k=0. Held-out usable n by layer: 40 / 37 / 37 / 38.
   At layer 40 the confirmatory n is **37, not 40** (detectable dz ~0.62).
5. **Teacher forcing is not neutral** on the deviant subset — `p_stop_soon`
   spans 0.016 to 0.9995 across cells. Note `surprisal` is stored as a
   **per-token list**, so it must be pooled before any per-cell summary.

## Next steps, in order

1. **Write the confirmatory analysis.** Does not exist. 15 contrasts, BCa
   bootstrap, Holm across the family, Q0 as a hard gate first. CPU only.
   Layer 40, `token_mean`, `latent_sum` primary; all trials, compliant-only as
   the robustness check.
2. **Deviation and leak rates per cell** — first-class outcomes in this fork,
   not exclusion criteria.
3. **Teacher-forcing robustness check**, pooling the per-token surprisal.
4. **262k SAEs** — pre-registered for the confirmatory run, needs a GPU pod:
   ~22 GB of weights, re-run `select_latents`, re-run `measure`. Note the
   analysis layer was chosen at 16k and would be applied at 262k. Given finding
   1, decide whether 262k is expected to fix the floor effect before paying for
   it.

## Mac / CPU-only work

Everything in steps 1–3. The parquets under
`artifacts/runs/*/*/results/*.parquet` are a few MB and are committed with
`git add -f` (artifacts/ is gitignored). Raw activations do **not** travel and
need the volume remounted.

## Two process rules learned the hard way

- **Never edit a running bash script.** Bash reads by byte offset, so an edit
  mid-run resumes at a shifted position and can re-execute a line. This cost a
  redundant 18,487-trial run. Python is safe (compiled up front).
- **`pkill -f <pattern>` matches your own shell.** It has self-terminated a
  command mid-sequence twice. Enumerate PIDs, skip `$$`, then kill.
