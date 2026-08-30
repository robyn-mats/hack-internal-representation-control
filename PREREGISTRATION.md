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
- **Leak rate** — the concept appears in the completion. Two tiers are recorded
  per trial, both scored against all 50 concepts and both matched on word
  boundaries, case-insensitively:

  - **Strict (primary).** Inflectional forms of the concept's lemma —
    `dust / dusts / dusting / dusted`, `snow / snowed / snowing`. Frozen in the
    `forms` column of `irc/concepts.csv`.
  - **Loose (sensitivity check, never primary).** Strict plus hand-pruned WordNet
    derivational forms — `bloody`, `snowy`, `rubberize`, `pacify`. Frozen in
    `forms_derived`.

  **Leak is scored on the completion only, never on the prompt.** The prompt
  contains the concept by construction in every condition except T7, so scoring
  the prompt would mark essentially every trial as a leak and measure nothing.

  Declared limitations. Neither tier disambiguates sense: a completion using
  "dust" in an unrelated sense counts. Neither catches multiword or suppletive
  forms. The loose tier is secondary because WordNet relates lemmas by string,
  so polysemous concepts import derivations from unintended senses — the three
  clearest (`Phones`→phonetic, `Deserts`→desertion, `Information`→inform) are
  dropped in `rk_scripts/gen_concepts_csv.py`, but the tier remains broader than
  the experiment's claim.

**Pre-registered expectation about deviation (directional).** Deviation rates are
expected to be **highest in the imperative families that ask for an action on the
concept** — N (`juggle X`, `braise X`), P (`do not laminate X`), Q (`glorf X`) and
R (`do not vusk X`) — and lowest in the declarative families (B, K, L, M) and the
bare baselines (T1–T5). Reasoning: a non-mental imperative asks the model to *do
something to* the concept, and the most available way to comply is to write
something other than the carrier sentence, whereas a declarative asserts
something that requires no action. A mental imperative (G, `do not think about
X`) should sit between — actionable in principle, not externally realisable.

This is a prediction about a first-class outcome, not a nuisance. If it holds,
deviation is tracking how *actionable* an instruction is, which bears directly on
Q7's coherence gradient. Recorded before any generation run beyond a 5-trial
smoke test.

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
| Q5d | A/B/C/E | Frame × negation interaction on the **focus** side — the mirror of Q5. |
| Q6 | H vs G | Direction not predicted. |
| Q7 | I vs N/P vs Q/R/S | Direction not predicted. |
| Q8 | T1 vs T2; T1 vs T3; T3 vs T4; T4 vs T5 | Direction not predicted. |
| Q9 | L1 vs T6 | L1 significantly below T6. **Sanity check — must pass.** |
| Q10 | within-cell variance | Descriptive. |

Q3 is the primary contrast. The others are secondary and corrected within the
same family.

**Q5d and the extension of Q8 close two gaps in the original list.** Every other
cell entered a confirmatory contrast; B (`focus_decl_none`) and T3
(`base_filler_none`) did not, so both were being generated and then used only in
Q10's descriptive variance check.

- **Q5d (A/B/C/E)** is the focus-side frame × negation 2×2 — imperative/none,
  declarative/none, imperative/syntactic, declarative/syntactic — exactly
  mirroring Q5 on the away side. Without it the design tested the interaction on
  one side only, and `PLAN.md` §5's prediction matrix makes claims about C that
  have no declarative counterpart to check them against.
- **T3** is the un-negated filler baseline that both T4 and T5 are defined as
  negating. Testing T4 vs T5 without it establishes that arrangement matters but
  not that either differs from the filler alone. `T1 vs T3` asks whether the
  filler does anything by itself; `T3 vs T4` is the negation test with filler
  present, parallel to `T1 vs T2` without it.

Q8 absorbs its additions rather than becoming new family members, so the
correction family grows by one (Q5d) — 12 tests, moving the primary contrast's
detectable dz from 0.58 to 0.59.

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

**Prompt scaffold.** The scaffold in `CLAUDE.md` is the default and is used
unless the pilot triggers the rule below. It is *not* free to re-choose after
seeing results.

**Always reported, whether or not the trigger fires:** per-cell deviation and
leak rates from the pilot under the default scaffold. These are results about
instruction-following, not diagnostics, and they are reported from the pilot
where they were measured.

**Trigger**, evaluated on the 10 pilot concepts only: any cell entering a
confirmatory contrast exceeds **50%** deviation, or pooled deviation exceeds
**25%**. Fifty percent is the point at which the modal trial in a cell is no
longer the intended stimulus, so that cell's mean activation is dominated by
whatever else was written; twenty-five percent pooled is where the
pre-registered compliant-only robustness check stops retaining enough trials to
be a meaningful comparison.

The trigger applies to N, P, Q and R **as well**, even though high deviation
there is the registered expectation. Q7 needs those cells to be interpretable:
if `juggle X` deviates on 60% of trials, `I vs N/P vs Q/R/S` measures output
text rather than internal representation. The deviation finding is preserved by
reporting it from the pilot; exempting those cells would protect the finding at
the cost of the contrast it exists to feed.

**Candidates.** All four keep `While you write it: {frame}.` as the final line,
so the distance from the concept tokens to the response tokens is identical
across candidates. This is the constraint that matters: the readout is on
response tokens, attention decays with distance, and any candidate placing text
after the frame would push the concept further away — reducing measured
activation for positional reasons unrelated to the instruction's meaning, and
biasing the selection rule below against it for the wrong cause. Only the copy
instruction and any constraint placed *before* the frame may vary.

    S1 (current)
      Write exactly this sentence and nothing else:
      "{carrier}"
      While you write it: {frame}.

    S2 (stronger copy instruction)
      Repeat the following sentence exactly. Output only that sentence -- no
      commentary, no explanation, no preamble.
      "{carrier}"
      While you write it: {frame}.

    S3 (explicit no-addition line, before the frame)
      Write exactly this sentence and nothing else:
      "{carrier}"
      Do not add anything before or after it.
      While you write it: {frame}.

    S4 (S2 and S3 combined)
      Repeat the following sentence exactly. Output only that sentence -- no
      commentary, no explanation, no preamble.
      "{carrier}"
      Do not add anything before or after it.
      While you write it: {frame}.

These target the observed failure mode: in the timing sample the one non-exact
completion reproduced the carrier correctly and *then* appended a comment about
the instruction. Every candidate therefore strengthens the "nothing else"
constraint rather than altering the frame, which is the manipulation and must
not change.

**Selection.** Not a minimisation. A qualifying test:

1. **Compliance floor** — pooled deviation below 25%. Necessary, not sufficient:
   a scaffold could reach full compliance by making the model ignore the third
   line entirely, which is perfect copying and no experiment.
2. **Manipulation check** — `A > T1 > T7` must hold on the pilot concepts. This
   is Q0's own gate applied to scaffold selection, and it is what catches the
   failure in (1): a scaffold whose frame has gone inert fails here.
3. **Among candidates passing both**, take the one maximising A-vs-T7 separation
   on the pilot — the same rule already used to choose the analysis layer and
   the pooling rule, licensed by the same thing: it runs on the 10 pilot
   concepts, and the 40 held-out are untouched until the choice is frozen.

**Leak rate is a disqualifier, never an objective.** Leakage is the concept
appearing in the output and concept activation is the dependent variable, so
selecting the scaffold with the lowest leak rate risks selecting the one that
suppresses the quantity being measured — shrinking the effect before the
experiment starts. Overall leak level is therefore not optimised. A candidate is
rejected, whatever its deviation rate, if its leak rate varies across the
contrast families of interest (G / I / K / L / M) by more than the other
candidates' spread: *differential* leakage is a confound rather than a nuisance.

**Disclosure.** If the scaffold changes, the readout for the affected cells is
reported under **both** scaffolds on the pilot concepts. A new scaffold that
suppresses deviation may also suppress the internal effect that produced it —
the two could share a cause — and that would otherwise look like a clean
improvement. Pilot data under the default scaffold already exists, so this costs
nothing.

Changing the scaffold invalidates `stimuli.csv`, which is regenerated by
`rk_scripts/01_generate_stimuli.py`; the regenerated row count and the resulting
`n_prompt_tokens` range are recorded in the amendment, since prompt length is
itself a pre-registered dilution covariate.

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
| 2026-08-30 | Leak rate specified as two tiers — strict (inflectional, primary) and loose (plus pruned WordNet derivational, sensitivity). Forms frozen in `irc/concepts.csv` as `forms` and `forms_derived`. | The original wording said "an inflected form", but the implementation carried only number inflection, so `dusting`, `snowed` and `milked` would not have counted as leaks of Dust, Snow and Milk. LemmInflect supplies the full inflectional paradigm at no precision cost — its occasional non-words (`lightninged`) cannot produce false positives because such strings do not occur in text. Derivational forms are kept separate rather than merged, because WordNet's lemma-string linking imports unrelated senses. No experimental data existed at the time of this change. |
| 2026-08-30 | A1/A2 templates shortened: `concentrate on {concept} while you write` → `concentrate on {concept}`, and likewise A2. No change to cells, factors or counts (still 67 phrasings, 25 cells). | The scaffold already supplies the temporal anchor ("While you write it:"), so A1/A2 rendered it twice. The redundancy made A2 (31 tokens) *longer* than its own negation G2 (30), inverting the length relationship on a pair Q0 and Q2 both rest on and confounding it with `n_prompt_tokens`, a pre-registered dilution check. Now 28 vs 30, differing by exactly the negation. No experimental data existed at the time of this change. |
