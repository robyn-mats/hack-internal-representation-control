# Pre-registration

**Committed before any experimental data exists.** Nothing in this file may be
revised after results are collected. Revisions before Stage 2 are permitted and
must be recorded as dated amendments at the bottom, with reasons.

- Repository: fork of Kamp (2026) *Intentional Control of Internal States in Gemma 3 27B*
- Design rationale: `PLAN.md`. Operating conventions: `CLAUDE.md`.
- Incidental findings and methodological notes: `NOTES.md`
- Committed: **[fill in date]**, git commit **[fill in hash]**

---

## Two-stage structure

Some choices genuinely depend on pilot data. Rather than pretend otherwise, this
is a two-stage pre-registration.

**Stage 1 (this file, now).** Everything not dependent on data, plus the *rules*
by which the pilot-dependent values will be derived.

**Stage 2 (after the pilot, before the held-out run).** The derived values are
recorded in the amendment section below and committed. Only then does the
held-out grid run. Held-out data is not inspected before this commit.

---

## Concept split

Drawn once, stratified by grammatical number, with `random.Random(20260827)`.
Fixed; not redrawn under any circumstances. Source of truth for the concepts and
their agreement features is `irc/concepts.csv`, derived from `irc/words_paper.py`.

**Pilot (10)** — used to derive Stage 2 values. Never enters confirmatory tests.
3 mass / 7 plural.

> amphitheaters, frosts, kaleidoscopes, rubber, satellites, secrecy, silver,
> trumpets, vegetables, xylophones

**Held out (40)** — confirmatory analyses only. 14 mass / 26 plural.

> algorithms, aquariums, avalanches, bags, blood, boulders, bread, cameras,
> caverns, constellations, contraptions, denim, deserts, dirigibles, dust,
> dynasties, fountains, harmonies, illusions, information, lightning,
> masquerades, memories, milk, mirrors, monoliths, oceans, origami, peace,
> phones, plastic, poetry, quarries, sadness, snow, sugar, treasures, trees,
> volcanoes, youths

All 50 concepts are used; the two sets are disjoint and exhaust
`irc/concepts.csv`. Stratification verified against the observed 17 mass / 33
plural split.

**Power.** Confirmatory n = 40 concepts. Paired tests at α = .05 two-sided detect
a standardised paired difference of **dz ≈ 0.46 uncorrected**. The confirmatory
family is Holm-corrected across Q0–Q10, so the primary contrast requires
**dz ≈ 0.58** in the worst case. Effects smaller than that are not detectable
here, and a null will be reported as a null, not as absence of an effect.

---

## Fixed in advance (Stage 1)

**Model.** `google/gemma-3-27b-it`, bf16, greedy decoding. Never quantized or
fp32 — dtype changes the activations being measured. Where a smaller model is
used for development, it is `google/gemma-3-4b-it` at matched relative depth
(layer 22 of 34 ≈ 65%; 40 of 62 ≈ 65%). No results from a dev profile enter the
writeup. Carrier screening and all work reported here ran on the 27b model.

**Readouts.** Primary: Gemma Scope 2 SAE latent activation, `resid_post`.
Secondary: cosine with difference-in-means concept vectors, `word_tokens`
extraction variant. Both reported for every contrast; the SAE is the headline.

**Token positions.** Response tokens only. **Position 0 (BOS) excluded** from all
activation measurements — its residual norm is ~20× other tokens and the SAEs
were not trained on it.

**Primary method is generation.** Teacher-forcing is a robustness check on the
same prompts. Per-condition surprisal of the forced tokens is reported: flat
surprisal means teacher-forcing is near-neutral, surprisal tracking condition
means it is not and the forced comparison is confounded.

**Compliance.** Exactness is recorded, not used as a filter. Two rates are
reported per cell as first-class outcomes:

- **Deviation rate** — the completion is not an exact match to the target
  carrier sentence.
- **Leak rate** — the concept word or an inflected form of it appears in the
  completion. Inflected forms are taken from `irc/concepts.csv`. Leak is scored
  on the completion only, never on the prompt.

The confirmatory analysis runs on **all trials**; compliant-only is reported as
a robustness check. *(Rationale: conditioning on compliance conditions on an
outcome and opens a collider path — and per upstream's own note, it may discard
exactly the trials where the model engaged with the concept hardest.)*

**Unit of analysis.** The concept. Average over carriers, then over phrasings
within a cell, then paired tests across concepts. **Cells, not phrasings, are
compared**; `phrasing_id` is a replicate within `cell_id`.

**Statistics.** Paired comparisons across concepts. Effect size: standardised
paired mean difference (dz). Confidence intervals: BCa bootstrap over concepts,
10,000 resamples. Correction: Holm across the confirmatory family (Q0–Q10),
α = .05, two-sided throughout.

**Condition set.** Frozen at 67 phrasings across 25 cells in `irc/conditions.csv`
as of this commit.

**Floor control pairing.** For `T6`, each concept is paired with a fixed partner
concept drawn once by seeded shuffle of the held-out list; the pairing is
recorded in `stimuli.csv` and not changed.

---

## Carrier selection (Stage 1 — complete and frozen)

Carrier screening is **stimulus norming, not a result**: it involves no
condition, no instruction frame and no dependent variable, so nothing about any
contrast can be read off it. It was therefore completed and frozen *before* this
file was committed, and the values below are fixed rather than derived at
Stage 2.

**Motivation.** The declarative families (B, E, F, K, L, M, S, T6) assert
something about the concept. `X is irrelevant to this task` is a **false
statement** when the carrier is about X, which is a different manipulation from
an irrelevance instruction. This narrows the claim to *the effect of `irrelevant`
when it is true*, and leaves the crossed version as a follow-up. The imperative
families are unaffected — `ignore X` is not false when the carrier is about X,
only harder to obey, which is a finding rather than a confound.

**Instrument.** All 50 `SENTENCES_PAPER` embedded against all 50
`CONCEPT_WORDS_PAPER` with the model under test: bare text, no chat template,
mean-pooled `resid_post` over the item's own tokens, **BOS excluded**, each pool
centred by its own per-layer mean. Concepts use the `word_tokens` extraction
variant. Screening runs against **all 50** concepts, not the held-out subset, so
the rule does not depend on the concept split.

**Layer 43**, not the readout layer 40. Selected on a known-pairs diagnostic
(eight hand-identified contaminated carriers ranked against all 62 layers)
before any threshold was chosen: layers 42–54 recover the true concept at
ceiling; layer 40 recovers 2 of 3 non-lexical pairs and ranks one known
contaminant 45th of 50; layer 31 is at chance (mean rank 20.3 of 50 against a
chance baseline of 25.5). The screen and the readout answer different questions,
and the screen uses no SAE so it is not restricted to the Gemma Scope layers.
Per-stimulus similarity **at layer 40** is recorded in `stimuli.csv` as a
covariate regardless.

**Gate.** For each (carrier, concept) pair the centred cosine at layer 43 is
standardised two ways — within carrier across the 50 concepts (`z_carrier`) and
within concept across the 50 carriers (`z_concept`). A carrier is excluded if any
concept exceeds **z = 2.0 on either marginal**. The union is required because
neither marginal suffices: `Lightning` is caught only by `z_concept` (5.94),
`Oceans` only by `z_carrier` (2.98). See `NOTES.md` for why.

τ = 2.0 was fixed by two independent criteria agreeing, both applied before
selection: flagged pairs remain individually plausible on inspection down to 2.0,
and the statistic saturates immediately below it (25 sentences flagged at 2.00,
48 at 1.75). All eight known-contaminated pairs are caught.

**Manual exclusions.** The gate cannot detect part-whole or entailment overlap
(an orchestra contains trumpets; a forest is trees; an antique vase is a
treasure) and buries broadly-evocative concepts. Twelve survivors were removed by
reading them, enumerated with per-sentence reasons in
`rk_scripts/screen_carriers.py::MANUAL_EXCLUSIONS`. Manual exclusion **only ever
removes** candidates, is listed in full, and was fixed before any experimental
data existed.

**Attrition.** 50 → 19 after the gate → **7** after manual review.

**k = 7 carriers, fixed at Stage 1.** In a fully crossed design the
carrier main effect cancels in the paired contrast, so only carrier × condition
interaction contributes, and it enters the standard error only as `σ_within²/k`
added to `σ_between²`. Relative to ten carriers, seven inflates the contrast SE
by roughly 2% even when the interaction equals the between-concept SD. Seven
uncontaminated carriers dominate ten containing two known contaminations.

**The set**, in fixed order (`screen_carriers.py::SELECTED_CARRIERS_V1`):

1. The train arrived precisely on schedule.
2. The basketball bounced off the rim.
3. The chef garnished the plate with fresh herbs.
4. The cat jumped onto the windowsill to watch birds.
5. The air conditioner hummed quietly in the background.
6. The book fell open to page 217.
7. Fragrant lilacs bloomed along the garden fence.

Order is fixed so that truncating to fewer carriers, if run time requires it, is
a pre-registered choice rather than a post-hoc one. Carriers span 8–12 tokens.

**Evidence committed** as `carrier_similarity.csv` (50 × 50, repo root — *not*
`artifacts/`, which is gitignored).

**Declared limitation.** The gate does not detect implicit entailment, which is
why the manual list exists. Residual similarity of each selected carrier to each
concept is carried into `stimuli.csv` as `max_similarity` and
`similarity_to_this_concept` and analysed as an exploratory covariate.

---

## Gate

**Q0 (A > T1 > T7) is a hard gate.** If the basic ordering does not reproduce on
the held-out set, the confirmatory contrasts Q1–Q10 are **not run**, and the
result is reported as a failed replication of the base effect on this model with
this readout. No further analysis is attempted on that data.

---

## Confirmatory contrasts

Run in this order, corrected as one family.

| # | Contrast | Prediction |
|---|---|---|
| Q0 | A > T1 > T7 | **Gate.** Ordering holds. |
| Q1 | L vs T1 vs T7 | L below T1. Whether L is suppressed / dampened / not primed is reported, not predicted. |
| Q2 | G vs T1 | No significant difference. |
| Q3 | **I vs G** | **Primary.** Direction not predicted — the four hypotheses in `PLAN.md` §5 predict different signs. |
| Q3b | C vs A | Direction not predicted. |
| Q4 | K vs M | Direction not predicted. |
| Q5 | G/I/K/M | Frame × negation interaction. |
| Q5b | I vs J; K vs L | Direction not predicted. |
| Q5c | C vs D; E vs F | Direction not predicted. |
| Q6 | H vs G | Direction not predicted. |
| Q7 | I vs N/P vs Q/R/S | Direction not predicted. |
| Q8 | T1 vs T2; T4 vs T5 | Direction not predicted. |
| Q9 | L1 vs T6 | L1 significantly below T6. **Sanity check — must pass.** |
| Q10 | within-cell variance | Descriptive. |

Q3 is the primary contrast. The others are secondary and corrected within the
same family.

**Terminology is fixed** per `PLAN.md` §2 and used strictly: *suppressed* =
below T7; *dampened* = below T1 but above T7; *not primed* = indistinguishable
from T7; *primed* = above T7. *Rebound* means above T1, not merely elevated.

**Interpretation is fixed in advance** per the prediction matrix in `PLAN.md` §5:
negation-breaks-instructions, frame-type, mental-verbs-unactionable, and
away-instructions-specifically-hard each predict a distinct signature across
G / I / K / M / C. The observed pattern is matched against those four; if it
matches none, that is reported as such rather than fitted post hoc.

---

## Derived at Stage 2 — rules fixed now, values recorded later

**Analysis layer.** Chosen on the pilot as the layer maximising the A-vs-T7
separation, restricted to layers with a published Gemma Scope 2 SAE
(16 / 31 / 40 / 53). Single layer for confirmatory tests. Full layer curve
reported as a secondary figure.

*Prior expectation, recorded so it cannot be claimed after the fact:* the
carrier screen found bare-text concept geometry legible over layers 42–54,
marginal at 40 and at chance at 31. That is a different instrument from SAE
latent separation and does not constrain the choice above, but if the pilot is
ambiguous between 40 and 53 it is weak converging evidence for 53.

**Pooling rule.** Chosen on the pilot from exactly three candidates: mean over
response tokens; mean over the top-k token positions by activation; activation at
positions where the concept is a plausible next token. Whichever maximises
A-vs-T7 separation on the pilot.

**SAE width.** 16k for the pilot. 262k for the confirmatory run *if and only if*
its Neuronpedia index is confirmed available (see `CLAUDE.md`); otherwise 16k
throughout. Decided before the held-out run, recorded below.

---

## Declared exploratory

Reported as exploratory, not corrected, not treated as confirmatory:

- **Mass vs plural concepts** (14 / 26 held out), which tracks abstract vs
  concrete almost exactly. Concrete nouns may yield cleaner SAE latents.
- **`juggle` as a possible outlier within N**, given its idiomatic attentional
  sense ("juggling priorities").
- **`disattend` as a frequency probe** — if it patterns with N or Q rather than
  its J cellmates, that locates real-but-unfamiliar verbs on the frequency axis.
- **`overlook`-style intentionality** — not currently in the grid; noted only.
- **Residual carrier similarity** as a continuous covariate, using
  `max_similarity` and `similarity_to_this_concept` at layer 40.
- **Dilution diagnostics**: second-concept readout, raw projection vs cosine,
  reduction vs `n_prompt_tokens`, and family N as the dilution control.
- **The J-lens arm**, if attempted.
- Any layer other than the one fixed at Stage 2.

---

## Coding decisions recorded in advance

- Negation is coded by whether the stem is a free morpheme in modern English.
  `ignore` is therefore coded **negation-free** despite Latin *in-* + *gnarus*.
- `dismiss` was considered for J and **dropped** before data collection:
  `dismiss` does not decompose as `disregard` does.
- `defenestrate` and `vanquish` were considered for N and **rejected** before
  data collection as away-coded.
- Concept words are stored capitalised and lowercased into prompts.
- The prompt scaffold uses a **colon**, not a comma, before the frame: a comma
  leaves bare-noun conditions ungrammatical while imperatives stay fine, so
  grammaticality would otherwise vary with condition.

---

## Amendments

Revisions to this file after it is committed. Stage 2 values go here, each
dated, with a reason. Anything recorded below post-dates the commit above.

| Date | Change | Reason |
|---|---|---|
| | | |
