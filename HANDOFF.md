# Handoff — 2026-09-05

State at the point the RunPod pod was released. Written so a fresh session on a
laptop can finish the project without re-deriving anything.

Design rationale is in `PLAN.md`, commitments in `PREREGISTRATION.md`, findings
in `NOTES.md`. This file is only *where things stand and what to do next*.

## The short version

The experiment is **finished and measured**. What remains is one analysis run
and a writeup, all of which works on a laptop with pandas and scipy. No GPU, no
pod, no volume.

**The held-out set has never been unblinded.** Running the confirmatory analysis
is the one-way door.

## Setup on a laptop

```bash
git clone <repo> && cd hack-internal-representation-control
pip install pandas numpy scipy pyarrow python-dotenv     # NO torch needed
```

Verified: `14_confirmatory.py`, `15_compliance.py` and `13_readout_health.py`
import no torch, transformers, huggingface_hub or sae_lens. Every input they read
is committed. No `.env` is required (`load_dotenv` on a missing file is a no-op).
Needs pandas >= 2.2.

Ignore the `uv` instructions in `CLAUDE.md` — those are for the pod. Plain
`python3` is fine.

## Where the data is

| what | where | size |
|---|---|---|
| Readouts, generations, provenance, stimuli | **git** (`artifacts/runs/*/`) | ~22 MB |
| Packed activations, concept vectors, latent selections | **HF private dataset** `RobynMATS/whitebear-acts` | 11.5 GB |
| Model + SAE weights | nowhere — re-download from the Hub | 125 GB |

The HF archive is verified: 68 files, 0 missing, 0 size mismatches, sha256 for
every file in `archive_manifest.json` (committed). The activations are
residual-stream reads and therefore **SAE-width independent**, so any future
readout can be computed without regenerating anything.

## What is done

| | |
|---|---|
| Design | 7 screened carriers x 25 cells (67 phrasings) x 50 concepts, 10 pilot / 40 held out |
| Generation | 27,674 trials over 4 passes; held-out deviation 0.18% |
| Measurement | 730,626 readout rows; 3 readouts x 4 layers x 2-3 poolings |
| Stage 2 | pooling `token_mean` (`61ab4f2`), analysis layer **40** (`b54ee96`) |
| Analysis code | confirmatory, compliance, positive control, consolidation, archive |
| Archive | uploaded and verified (`abefa6a`) |

Both Stage 2 values were committed **before** held-out was measured. That
ordering is the check, and it is visible in git history.

## What the pilot established

1. **The SAE readout has a severe floor effect** — 92.1% of readouts are exactly
   0.0 at layer 40. A positive control proved the instrument works (the selected
   latents fire at 3,000-16,000 in the context they were selected from), so the
   zeros are real measurements: instructing the model to focus on a concept
   while it copies an unrelated sentence largely fails to activate that concept.
2. **The zeros are not spread evenly.** Informative concepts per contrast on the
   pilot: 9 of 9 for Q0 and Q3c, 5 of 9 for Q3 (the primary), 3 of 9 for Q1 and
   Q4, **2 of 9 for Q5e** (`irrelevant` vs `not relevant` — the comparison that
   motivates the fork). Cells directing attention *toward* the concept fire;
   away and declarative cells largely do not.
3. **The concept vector is far more sensitive** — pilot dz 2.29 vs the SAE's
   0.71 at layer 40, with no zero problem. Registered *secondary*; the
   2026-09-05 amendment requires it be reported for all 15 contrasts with equal
   prominence.
4. **Q0 holds at layers 16, 31 and 40; it fails at 53.** Layer 16 passes only
   trivially (A 1.75, T1 1.72, T7 1.57).
5. **Latent selection is ragged.** k=5 is a ceiling: 92 of 200 (concept, layer)
   cells have k<5, 9 have k=0. At layer 40 the confirmatory n is **37, not 40**.
6. **Teacher forcing is not neutral** on the deviant subset — `p_stop_soon`
   spans 0.016 to 0.9995 across cells. `surprisal` is stored as a **per-token
   list** and must be pooled before any per-cell summary.

## Next steps

### 1. Run the confirmatory analysis — THE UNBLINDING

```bash
python3 rk_scripts/14_confirmatory.py --run-id heldout1 --out conf_sae.json
python3 rk_scripts/14_confirmatory.py --run-id heldout1 --readout concept_vector --out conf_cv.json
python3 rk_scripts/14_confirmatory.py --run-id heldout1 --compliant-only --out conf_sae_compliant.json
```

Layer and pooling come from the committed `stage2_values.json`, not from flags.
Q0 is enforced as a hard gate and the script exits if it fails. Every contrast
prints `informative=N` and a verdict of `significant` / `null` / `UNDERPOWERED`,
per the 2026-09-05 amendment.

**Read `PREREGISTRATION.md`'s amendment table first.** Two things in the analysis
are implementation decisions rather than registered commitments, both flagged in
the script's docstring:

- how a family member listing several comparisons yields one p-value (average of
  per-concept differences for parallel contrasts; Friedman omnibus for gradients
  and heterogeneous sets)
- the `informative_n >= 15` threshold for calling a null a null

Neither has been signed off. They are defensible and documented, but they are the
places a reviewer would push.

### 2. Compliance and leak rates

```bash
python3 rk_scripts/15_compliance.py --run-id heldout1
```

Expect a floor. The tagged scaffold cut deviation to 0.18%, so the registered
deviation prediction is close to untestable — a real trade-off worth reporting,
not a bug.

### 3. Teacher-forcing robustness

Pool the per-token `surprisal` list first. `pilot2/teacher_forced` is the full
pass that can actually estimate the confound; `heldout1/teacher_forced` is
deviating trials only, so it is conditioned on deviation.

### 4. Write it up

The floor effect is a result, not an obstacle. The defensible headline:

> Telling the model to *focus on* a concept while copying an unrelated sentence
> reliably activates that concept's SAE latents. Telling it to ignore, suppress
> or deem the concept irrelevant produces almost no measurable activation —
> largely indistinguishable from never mentioning the concept.

## Deliberately not doing

- **The 262k SAE arm.** Prediction on record before any spend: it will not fix
  the floor. The positive control showed 16k *can* represent these concepts;
  the zeros mean they are not active during copying, and a finer decomposition
  catches weaker activation rather than manufacturing activation. The archive
  keeps the option open at no ongoing cost.
- Hurdle/two-part models, co-primary readouts, rank tests. Rank tests are
  explicitly rejected in the amendment: Wilcoxon discards zero differences, so
  on Q5e it would test 2 concepts while looking more robust.

## Rules that still bind

- `PREREGISTRATION.md` may gain amendments but nothing already written may be
  edited. **After unblinding, no further amendments** — the split is spent.
- Deviation and leak rates are results, not exclusion criteria.
- The confirmatory analysis runs on **all trials**; compliant-only is the
  robustness check.
- American English throughout, including code comments.

## Traps this project hit, worth not repeating

- **Never edit a running bash script.** Bash reads by byte offset, so an edit
  mid-run resumes at a shifted position and can re-execute a line. Cost a
  redundant 18,487-trial run. Python is safe (compiled up front).
- **`pkill -f` and `pgrep -f` match your own shell.** Self-terminated a command
  three times, once discarding an unapplied edit. Use a bracket pattern like
  `'14_conf[i]rmatory'`, or enumerate PIDs and skip `$$`.
- **`git add -f` aborts the whole add on one bad pathspec.** Produced a commit
  whose message described 19 files while containing 2 (`747ef0d`, corrected by
  `3b99907`). Add per-path in a loop.
- **`free` inside a container reports the host, not the cgroup.** The pod's real
  limit was 233 GiB, not the ~2 TB `free` showed, and page cache counts against
  it.
- **Silent-wrong beats loud-wrong for damage.** Three of the measure stage's
  five bugs produced plausible-looking output: a 0-row parquet reported as
  success, a baseline collapsed from 50 concepts to 1, and `k=0` cells summing
  to a fabricated `0.0`. Run a stage on 20 trials and *read the row counts*
  before running it on 27,674.
