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

Nothing below needs a GPU except step 5.

1. **Run the confirmatory analysis.** Built and committed but NOT yet run on
   held-out — that is the unblinding, and it is a one-way door.

       python3 rk_scripts/14_confirmatory.py --run-id heldout1 --out conf.json
       python3 rk_scripts/14_confirmatory.py --run-id heldout1 --compliant-only
       python3 rk_scripts/14_confirmatory.py --run-id heldout1 --readout concept_vector

   Layer and pooling are read from the committed `stage2_values.json`, not from
   flags. Q0 is enforced as a hard gate and the script exits if it fails.
   **One thing wants sign-off first:** how a family member with several
   comparisons yields one p-value. The pre-registration fixes the family at 15
   without saying, so the script adopts a documented rule (average of
   per-concept differences for parallel contrasts; Friedman omnibus for
   gradients and heterogeneous sets) and says so in its docstring and in the
   amendment. That is an implementation decision, not a registered commitment.

2. **Compliance tables.** `python3 rk_scripts/15_compliance.py --run-id heldout1`
   Expect a floor: the tagged scaffold cut deviation to 0.18% on held-out, so
   the registered deviation prediction is close to untestable.

3. **Teacher-forcing robustness check.** `surprisal` is stored as a
   **per-token list** and must be pooled before any per-cell summary. On the
   33-trial deviant subset `p_stop_soon` spans 0.016 to 0.9995 across cells, so
   forcing is not neutral there — but that subset is conditioned on deviation,
   and `pilot2/teacher_forced` is the full pass that can actually estimate it.

4. **Finish the archive** (needs the volume, not a GPU):

       python3 rk_scripts/16_consolidate_acts.py --run-id pilot2 --all-passes --verify
       python3 rk_scripts/16_consolidate_acts.py --run-id heldout1 --all-passes --verify
       python3 rk_scripts/17_archive_hf.py --repo <user>/whitebear-acts --dry-run
       python3 rk_scripts/17_archive_hf.py --repo <user>/whitebear-acts

   State at 2026-08-31 18:15: `pilot2/generated` is packed and **bitwise
   verified, 0 mismatches**; the other three passes are not packed. Delete any
   partial `acts_packed.pt` before re-running — a truncated pack plus its index
   could look valid. Originals are never deleted by the script.

   **Blocked on a write token.** `HF_TOKEN` in `.env` is read-only
   (`role: read`); create one with write access and the archive script will run.
   It refuses to start on a read token rather than failing partway through 8 GB.

5. **262k SAE arm** (GPU). Download ~22 GB of 262k weights, re-run
   `select_latents`, re-run `measure`. The stored activations are
   residual-stream reads and therefore SAE-width independent, so **no
   regeneration is needed** — which is the whole reason to archive them. Note
   the analysis layer was chosen at 16k and would be applied at 262k, and that
   today's finding (92% of readouts exactly zero at the best layer, with the
   instrument validated) is the thing 262k is meant to improve. Decide whether
   it plausibly fixes a floor effect before paying for it.

## Mac / CPU-only work

Steps 1-3 need nothing but the repo: `git clone`, then pandas + scipy. The
committed inputs are ~22 MB. Raw activations do **not** travel and are not read
by any remaining analysis.

## Two process rules learned the hard way

- **Never edit a running bash script.** Bash reads by byte offset, so an edit
  mid-run resumes at a shifted position and can re-execute a line. This cost a
  redundant 18,487-trial run. Python is safe (compiled up front).
- **`pkill -f <pattern>` matches your own shell**, and so does `pgrep -f`. This
  self-terminated a command three times in one session, once discarding an
  unapplied edit. Enumerate PIDs and skip `$$`, or use a bracket pattern like
  `'16_consolidat[e]'`.

## Also worth knowing

- `git status` and `git commit` take minutes in this repo on the pod, because
  git walks a working tree holding 27,674 activation files on a network
  filesystem. Commit with explicit paths, and expect to background it.
- **`git add -f` aborts the entire add on one bad pathspec** rather than adding
  the rest. This produced a commit whose message described 19 files while
  containing 2 (`747ef0d`, corrected by `3b99907`). Add per-path in a loop.
- `free` inside the container reports the **host** (~2 TB), not the cgroup. The
  real limit is `/sys/fs/cgroup/memory/memory.limit_in_bytes` — 233 GiB — and
  page cache counts against it.
