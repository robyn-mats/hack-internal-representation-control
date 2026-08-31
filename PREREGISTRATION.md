# Pre-registration

**Committed before any experimental data exists.** Nothing in this file may be
revised after results are collected. Revisions before Stage 2 are permitted and
must be recorded as dated amendments at the bottom, with reasons.

- Repository: fork of Kamp (2026) *Intentional Control of Internal States in Gemma 3 27B*
- Design rationale: `PLAN.md`. Operating conventions: `CLAUDE.md`.
- Incidental findings and methodological notes: `NOTES.md`
- Committed: **2026-08-30 00:01:42 EDT** (`2026-08-30T04:01:42+00:00`),
  git commit **`99533c1`**
- First experimental data of any kind: `2026-08-30T17:46:34Z`
  (`artifacts/runs/smoke/generated/invocations.jsonl`), 13h 45m later. Every run
  directory records its own start time and git commit, so the ordering is
  checkable rather than asserted.
- Amendments since that commit are listed at the bottom, each dated with its
  reason. All of them predate any held-out measurement.

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
a standardized paired difference of **dz ≈ 0.46 uncorrected**. The confirmatory
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
same prompts, and for the held-out run it is **restricted to the trials that
deviate** (amended 2026-08-31; the pilot runs it in full). On a compliant trial
the forced and generated activations are bit-identical — same prompt, same
tokens, deterministic forward pass, verified at max|diff| = 0 — so the forced
pass carries unique information only where the model did not write the carrier
verbatim. Under the tagged scaffold that is 2 of 4,627 pilot trials, so forcing
the full held-out grid would cost ~9.4 h to learn about ~12 trials. The trials
are selected by `rk_scripts/10_deviant_stimuli.py` after the generated pass
completes. Two quantities are reported per condition:

- **Forced-token surprisal** — mean over the carrier's own tokens.
- **Stop surprisal** — the surprisal of `<end_of_turn>` immediately after the
  forced carrier, i.e. how reluctant the model is to stop there, together with
  the token it would rather have written.

The second is the load-bearing one, and it is not a refinement. Forcing the
carrier is unsurprising for any model that would have written it anyway: in the
pilot smoke test N1 (`juggle X`) reproduced the carrier *correctly* and only then
appended a comment, so all seven forced tokens scored near zero while the entire
deviation sat one token past the end. Forced-token surprisal alone would have
reported that trial as indistinguishable from a fully compliant one.

Flat stop surprisal across conditions means teacher-forcing is near-neutral;
stop surprisal tracking condition means it is not, and the forced comparison is
confounded with reluctance to stop.

**Compliance.** Exactness is recorded, not used as a filter. Two rates are
reported per cell as first-class outcomes:

- **Deviation rate** — the completion is not an exact match to the target
  carrier sentence.
- **Leak rate** — the concept appears in the completion. Two tiers are recorded
  per trial, both scored against all 50 concepts and both matched on word
  boundaries, case-insensitively:

  - **Strict (primary).** Inflectional forms of the concept's lemma —
    `dust / dusts / dusting / dusted`, `snow / snowed / snowing` — **plus emoji
    that depict the concept** (`trumpets` → 🎺, `oceans` → 🌊). Frozen in the
    `forms` and `forms_emoji` columns of `irc/concepts.csv`. A concept reaching
    the output pictorially has surfaced as much as one reaching it lexically,
    and unlike the derivational forms there is no sense ambiguity, so emoji
    count under both tiers. Words are matched on word boundaries; emoji
    literally, since `\b` never matches around a non-word character.
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
X`) should sit between — actionable in principle, not externally realizable.

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

**Statistics.** Paired comparisons across concepts. Effect size: standardized
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
centered by its own per-layer mean. Concepts use the `word_tokens` extraction
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

**Gate.** For each (carrier, concept) pair the centered cosine at layer 43 is
standardized two ways — within carrier across the 50 concepts (`z_carrier`) and
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
`similarity_to_this_concept` and analyzed as an exploratory covariate.

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
| Q3c | A vs I; B vs K | **Direction**, frame and negation held fixed. Direction not predicted. |
| Q4 | K vs M | Direction not predicted. |
| Q5 | G/I/K/M | Frame × negation interaction. |
| Q5b | I vs J; K vs L | Direction not predicted. |
| Q5c | C vs D; E vs F | Direction not predicted. |
| Q5d | A/B/C/E | Frame × negation interaction on the **focus** side — the mirror of Q5. |
| Q5e | L vs M | Morphological vs syntactic negation, declarative held fixed. Direction not predicted. |
| Q5f | A vs D; B vs F | Double negation against **no** negation — composition. Direction not predicted. |
| Q6 | H vs G | Direction not predicted. |
| Q7 | I vs N/P vs Q/R/S vs T1 | Direction not predicted. |
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

**Q3c, Q5e and Q5f close gaps found by enumerating every pair of cells differing
in exactly one factor and checking it against the contrast list.**

- **Q3c (A vs I; B vs K)** — **direction had no direct test.** It is one of the
  three factors, yet no contrast compared a toward cell with an away cell at
  matched frame and negation; the ordering was only inferable transitively from
  `A > T1` (Q0) and `L < T1` (Q1). These are the maximal toward/away contrasts in
  the design.
- **Q5e (L vs M)** — the question `PLAN.md` §5 poses as Q5b, *"does morphological
  negation behave like syntactic, or like none?"*, was only approached via `K`
  (Q5b: `K vs L`; Q4: `K vs M`). This compares the two negation types directly,
  and it is also the `irrelevant`-vs-`not relevant` comparison that motivates the
  fork.
- **Q5f (A vs D; B vs F)** — Q5c compares double negation against *single*
  negation only. "Composed" means `D ≈ A`, so composition cannot be assessed
  without the un-negated baseline.

**Q7 additionally takes T1 as the floor of its gradient**, testing whether an
incoherent instruction does anything beyond bare mention: if `glorf X ≈ T1`,
nonsense instructions contribute nothing; if `glorf X > T1`, they do.

Q7 and Q8 absorb their additions rather than becoming new family members, so the
correction family is **15 tests** (Q0–Q10 plus Q3b, Q3c, Q5b–Q5f), moving the
primary contrast's detectable dz from 0.58 to **0.60**.

Single-factor pairs deliberately left untested, to protect power: direction at
syntactic negation (`C vs G/H/P/R`, `E vs M`), which is redundant once Q3c tests
it without negation; anything against `T6`, which Q9 anchors and which is a floor
control rather than a comparison target; `T3 vs T7`, transitively covered; and
`H vs I/J/M`, since H has two phrasings and Q6 pairs it with its matched
comparator G.

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

**Analysis layer.** Chosen on the pilot as the layer maximizing the A-vs-T7
separation, restricted to layers with a published Gemma Scope 2 SAE
(16 / 31 / 40 / 53). Single layer for confirmatory tests. Full layer curve
reported as a secondary figure.

*Prior expectation, recorded so it cannot be claimed after the fact:* the
carrier screen found bare-text concept geometry legible over layers 42–54,
marginal at 40 and at chance at 31. That is a different instrument from SAE
latent separation and does not constrain the choice above, but if the pilot is
ambiguous between 40 and 53 it is weak converging evidence for 53.

**Pooling rule.** Chosen on the pilot by A-vs-T7 separation. Three candidates
were pre-registered; **`top-k by activation` is dropped, and this is recorded
before any readout was computed.** It selects positions *by the quantity being
measured*, so a condition with higher variance scores higher at equal true mean —
and conditions are precisely what the contrasts compare, which makes it a
confound rather than a nuisance. `max` is retained as a descriptive companion,
not a candidate, having the same flaw in sharper form.

Two candidates remain eligible:

- **`token_mean`** — mean over the captured response tokens.
- **`plausible`** — weighted mean, weighting each position by P(the concept's
  first token | prefix). Computed by `rk_scripts/08_plausible_positions.py`,
  which needs a forward pass per trial because the runner stores activations
  rather than logits. Weighting by an *independent* quantity is what
  distinguishes it from the dropped candidate.

**Prompt scaffold — RESOLVED at Stage 2, 2026-08-31.** The trigger fired and the
comparison was run; the outcome is recorded here and in the amendment table. The
scaffold is now the tagged form in `CLAUDE.md` and is frozen.

*What was pre-registered:* a trigger (any confirmatory cell above 50% deviation,
or 25% pooled) and four candidate scaffolds, to be compared on the pilot
concepts, selected by a compliance floor, then the `A > T1 > T7` manipulation
check, then maximum A-vs-T7 separation.

*What happened, and where the pre-registration was wrong.* The trigger fired on
T3 (90.0%) and T4 (91.4%); pooled deviation was 6.8%, well under 25%. But the
pilot also diagnosed the cause, and **none of the four candidates addressed it**.
All four strengthened the *output constraint* ("output only the sentence"). The
actual failure was the **copy boundary** — the model reproducing the entire third
line, scaffold prefix included, because it could not tell where the text to copy
ended. Strengthening "nothing else" is the wrong lever for that, so running the
pre-registered comparison would have been a formality with a foreseeable null.

Two further pre-registration errors, recorded rather than quietly fixed:

- The selection rule implied re-running the 4,627-trial pilot per candidate,
  roughly 10 hours for four. What was run instead is a reduced comparison set:
  the cells the manipulation check needs plus the cells that actually deviate.
- The trigger was written before it was known that deviation depends heavily on
  which carrier is being copied, so no single run's absolute rate is the cell's
  true rate.

*What was compared.* Two rounds, both on pilot concepts. First, five wordings x
five conditions x 5 concepts x 2 carriers (210 trials) to find a tagged form that
does not echo its own delimiters. Then the winner against the incumbent over 14
phrasings x 10 concepts x 2 carriers (524 trials), using the real templates from
`irc/conditions.csv`.

*Result.* Deviation **27.5% → 0.4%** (72/262 → 1/262), improving every one of the
14 phrasings and regressing none. Leak rate 24.0% → 0.4%. Zero delimiter echoes.
The clean controls (G1, I5, L1) stayed clean, so the strengthened constraint does
not collide with instructions that ask the model to act — N1 (`juggle X`) went
50% → 0%.

*Honest limits.* T3 is made rare, not impossible: 1 trial in 20 still appended
its instruction line. The comparison used 2 of 7 carriers, and deviation varies
with carrier, so these absolute rates do not transfer. The naive tagged wording
was **worse than the incumbent** — it broke T1 from 0% to 100% by echoing markup
— so the result belongs to this specific wording, not to delimiters in general.

**SAE width.****SAE width.** 16k for the pilot. 262k for the confirmatory run *if and only if*
its Neuronpedia index is confirmed available; otherwise 16k throughout.

**Condition met, verified 2026-08-31.** Probing the Neuronpedia API at each of
`SAE_LAYERS`, 262k is the only width returning an explanation at all four
(16k and 65k both have gaps; 1m 404s at every layer despite a registry entry).
The confirmatory run therefore uses **262k**; the pilot stays at 16k as written.
Evidence in `CLAUDE.md`.

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
| 2026-08-31 | **STAGE 2 VALUE: analysis layer = 40.** Ranked on each layer's own usable concepts (the rule as originally written): L40 0.711 (n=9), L53 0.662 (n=10), L31 0.581, L16 0.483. Layer 53 is **excluded for failing the pilot Q0 ordering** (A 157.0, T1 157.3, T7 2.6). L40 passes cleanly (A 173.1 > T1 24.7 > T7 1.8) and is the only layer where the readout is informative for every measurable concept (9 of 9; L31 6 of 10, L53 6 of 10, L16 3 of 10). Held-out n = 37 of 40, detectable dz ~0.62. **Two clauses of the earlier amendment today are superseded:** (a) the common-concept restriction, and (b) coverage as a general tiebreak, replaced by *passing pilot Q0 as an eligibility precondition* applied before the dz criterion. The k=0-is-missing clause stands. | The common-set restriction was added the same morning to stop a layer ranking high by shedding its hardest concepts, and it does the opposite. It excludes any concept that **any** layer cannot measure, so a layer that measures `vegetables` and reads exactly 0.0 (L16, L31, L53) gets to drop it and inflate its own dz, while L40 — which has k=0 there and never had it — gains nothing. Every layer's dz rose under the restriction except L40's, which could not, and the argmax flipped from 40 to 53. Correct accounting: measurable-but-silent is **data** (delta = 0, kept), no-instrument is **missing** (dropped). Q0 becomes a precondition rather than a tiebreak because it is already a hard gate in this file: a layer the pilot shows failing it cannot support Q1–Q10 at all, so choosing it would guarantee no confirmatory analysis. **Disclosure of sequence:** pilot dz was computed before this revision, so the defect was found after seeing that the superseded rule selected 53. The defect is mechanical and demonstrable rather than a preference, and both dz tables are printed by `rk_scripts/11_stage2_choose.py` and stored in `stage2_values.json`. The pre-recorded 40-vs-53 prior expectation favoring 53 was conditioned on the pilot being *ambiguous*; it is not, once dz is computed on the registered concept sets. |
| 2026-08-31 | **Positive control run on all 50 concepts** (`rk_scripts/12_latent_positive_control.py`, `latent_positive_control.csv`). Held-out numbers therefore exist, and were used **only** as a binary usable/not count. | The pilot readout is exactly 0.0 for many (concept, layer) cells in every condition, which is ambiguous between a dead selection and a genuine null. The control measures the selected latents in the context they were selected from (the four `WORD_TEMPLATES_V1` prompts, concept's own tokens). Result: **zero SELECTION_DEAD at every layer** — the latents fire at 3,000–16,000 there and at 0.0 in the task, so the zeros are real measurements, not artifacts, and 92% zeros at the best layer is a finding about the task rather than a broken instrument. The control uses no carrier, no condition and no generation, and yields one number per (concept, layer) that is constant across all 25 conditions, so it cannot differentially favor any condition or bias a within-concept paired contrast. Continuous held-out control magnitudes were deliberately **not** used to rank layers, since activation scale could correlate with effect size; only the binary was. |
| 2026-08-31 | **Held-out latent coverage: rule fixed before any dz was computed.** Because the selection is ragged, usable held-out concepts vary by layer — L16: 40, L31: 37, L40: 37, L53: 38 (affected concepts: Bags, Deserts, Oceans, Snow, Trees; none lacks coverage at every layer). Three decisions: (a) the criterion stays **max A-vs-T7 dz** — coverage does not override it; (b) **tiebreak** — within the 0.10 dz band this file already calls ambiguous, prefer the layer with more usable held-out concepts; (c) a k=0 concept is **dropped from contrasts at that layer, never imputed**, and the per-layer n is reported with every result. The stated detectable dz of 0.60 at n=40 becomes **~0.62 at n=37–38**, by 1/sqrt(n) rescaling of the stated figure rather than a fresh power calculation. | Choosing the layer on the pilot could otherwise silently cost 3 of 40 held-out concepts and quietly falsify the registered power statement. Coverage is derived from `select_latents`, which sees only concept words and their template prompts — never a carrier, condition or generation — so it is instrument metadata, and reading it does not inspect held-out results. Recorded and committed **before** the pilot dz values were computed, so the rule cannot have been fitted to the ranking it decides; the commit ordering is checkable in git history. A full-coverage requirement was considered and rejected: it would force layer 16 (26% depth, where the carrier screen found concept geometry at chance) and let an instrument artifact choose the analysis depth. Note the confirmatory run is pre-registered at 262k, where coverage is expected to differ; this table is 16k. |
| 2026-08-31 | **Ragged SAE latent selection.** k=5 is a ceiling, not a count: 92 of 200 (concept, layer) cells have fewer than 5 selected latents and **9 have k=0**. Two consequences are fixed here. (a) Where k=0 the readout **does not exist** and no row is emitted — it is missing data, not a zero. (b) The analysis layer is ranked on the concepts paired at **every** layer, with each layer's own-coverage dz reported beside it. The criterion is unchanged (A-vs-T7 paired dz); only the concept set it is evaluated on is specified. | `sum` over an empty latent index returns a clean `0.0`, which would enter the primary readout as a genuine measurement of "concept not active" for 9 cells. And because coverage differs by layer (0 missing concepts at 16, four at 40), unrestricted per-layer dz values describe different concept sets, so a layer could rank higher partly by having dropped its hardest concepts. The pre-registration specified k=5 and did not anticipate ragged selection, so it prescribes no handling. In the pilot this is live rather than hypothetical: `Vegetables` has k=0 at layer 40, so the layer comparison runs on 9 of 10 concepts. Recorded before the layer was chosen; the numbers behind it are in NOTES.md 2026-08-31. |
| 2026-08-31 | **Pooling rule fixed at `token_mean`.** The third candidate, weighting positions by P(concept is the next token), is **degenerate for this task** and is dropped. | Measured on the pilot before any effect size was computed: median P(concept next) is 6.3e-23, the highest of 42,965 positions is 2.9e-06, no position exceeds 1e-4, and the within-trial max/min weight ratio is 7e+16. The model copies a fixed sentence under an explicit output-only instruction and complies on 99.96% of trials, so P(concept next) is ~0 at every position by construction. A weighted mean over weights spanning sixteen vanishing orders of magnitude selects far-tail softmax numerics rather than weighting anything. With `topk_mean` already dropped for selecting on the measured quantity, `token_mean` is the only non-degenerate candidate. Determined on pilot data only; held-out was not measured. Consequence: held-out rule 3 is not run. |
| 2026-08-31 | Held-out **generation** started before the Stage 2 values were committed. Held-out data is **not measured, inspected or analyzed** until Stage 2 is committed. | Stage 2 fixes the analysis layer and pooling rule. Neither changes what is captured: the run stores raw `resid_post` at all four SAE layers, and both choices are applied afterwards to stored activations. The protection the split provides is against held-out results influencing analysis choices, which generating-without-measuring cannot do. Recorded in advance rather than explained afterwards. The ordering is practical: the pilot's rule-3 pass and the held-out generation both need the single GPU, and generation is the ~11 h job. |
| 2026-08-31 | Pooling candidate `top-k by activation` **dropped**; `max` demoted to descriptive. `token_mean` and `plausible` remain eligible. | It selects positions by the quantity being measured, so a condition with higher variance scores higher at equal true mean — a confound, since conditions are what the contrasts compare. Recorded **before any readout was computed**, so it cannot be a response to which rule won. |
| 2026-08-31 | Teacher-forcing for the **held-out** run restricted to deviating trials; the pilot still runs it in full. | Forced and generated activations are bit-identical on compliant trials (verified, max abs diff = 0), so the forced pass is informative only where the model did not write the carrier verbatim — 2 of 4,627 under the tagged scaffold. Forcing the full grid costs ~9.4 h to learn about ~12 trials. `p_stop_soon` is thereby available only for deviating trials in held-out; it appears in no confirmatory contrast, and the pilot's full forced pass characterizes it across every cell. |
| 2026-08-31 | **Prompt scaffold amended** to the tagged form: `The tags below mark a sentence. Output that sentence alone -- no tags, no commentary.` / `<sentence>{carrier}</sentence>` / `While you write it: {frame}.` `stimuli.csv` regenerated, 23,450 rows and 23,107 unique prompts unchanged; `n_prompt_tokens` 18-40 → 32-54. `irc/conditions.csv` untouched — T3/T4/T5 keep their PLAN.md templates. | The Stage 2 trigger fired on T3 (90.0%) and T4 (91.4%). Deviation across 14 phrasings falls 27.5% → 0.4% (72/262 → 1/262), leak 24.0% → 0.4%, no cell regressed, no delimiter echoes. The tagged scaffold also fixes T3 with its **original** dot filler, which no filler substitution achieved — so this is one amendment rather than two and the condition grid is unchanged. Held-out data does not exist. `artifacts/runs/pilot1/` was collected under the previous scaffold and is a pre-amendment record, not confirmatory data; the pilot is being re-run. |
| 2026-08-30 | Leak rate additionally counts emoji depicting the concept, frozen in a `forms_emoji` column. 21 of 50 concepts have one. | The generated pilot produced 30 completions containing 🎺 for the concept `trumpets` — e.g. `The train arrived precisely on schedule. 🎺` under `do not skip past trumpets` — and the word-form detector caught 3 of them. The concept was reaching the output pictorially, which the pre-registered measure could not see. Recovers 27 trials, moving the pilot leak rate 5.66% → 6.25%. Derived from Unicode character names and hand-pruned (`fountains` → 🖋 is a pen; `information` → 💁 is a person; `deserts` → 🏝 is an island), same procedure as the derivational forms. Held-out data does not exist; the pilot is recomputable from stored completions without re-running. |
| 2026-08-30 | Leak rate specified as two tiers — strict (inflectional, primary) and loose (plus pruned WordNet derivational, sensitivity). Forms frozen in `irc/concepts.csv` as `forms` and `forms_derived`. | The original wording said "an inflected form", but the implementation carried only number inflection, so `dusting`, `snowed` and `milked` would not have counted as leaks of Dust, Snow and Milk. LemmInflect supplies the full inflectional paradigm at no precision cost — its occasional non-words (`lightninged`) cannot produce false positives because such strings do not occur in text. Derivational forms are kept separate rather than merged, because WordNet's lemma-string linking imports unrelated senses. No experimental data existed at the time of this change. |
| 2026-08-30 | A1/A2 templates shortened: `concentrate on {concept} while you write` → `concentrate on {concept}`, and likewise A2. No change to cells, factors or counts (still 67 phrasings, 25 cells). | The scaffold already supplies the temporal anchor ("While you write it:"), so A1/A2 rendered it twice. The redundancy made A2 (31 tokens) *longer* than its own negation G2 (30), inverting the length relationship on a pair Q0 and Q2 both rest on and confounding it with `n_prompt_tokens`, a pre-registered dilution check. Now 28 vs 30, differing by exactly the negation. No experimental data existed at the time of this change. |
