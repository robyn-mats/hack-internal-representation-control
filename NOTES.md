# Working notes and incidental findings

Chronological record of things discovered while building the experiment that are
not themselves experimental results. Nothing here is confirmatory. Findings that
bear on pre-registered choices are cross-referenced from `PREREGISTRATION.md`.

---

## 2026-08-30 — The paper's carrier set is largely unusable for this design

**Finding: 43 of Anthropic's 50 carrier sentences are contaminated by at least
one of the 50 concept words. Seven survive.**

This is a fork-specific problem, and worth stating plainly because it is not a
criticism of the prior art. Lindsey (2025) and Kamp (2026) pair each carrier
with each concept and ask whether an instruction changes the concept's internal
activation. Overlap between carrier and concept adds variance but does not
invalidate the manipulation.

This fork's declarative conditions make it invalidating. `X is irrelevant to
this task` is a **false statement** when the carrier is about X, and a false
statement is a different manipulation from an irrelevance instruction. That
applies to families B, E, F, K, L, M, S and T6 — roughly half the grid. The
imperative families (`ignore X`, `set X aside`) are unaffected: they are not
false when the carrier is about X, only harder to obey, which is a finding
rather than a confound.

Attrition, screening at layer 43 (see below):

| stage | removed | remaining |
|---|---|---|
| all paper carriers | — | 50 |
| automated gate, z > 2.0 on either marginal | 31 | 19 |
| manual review of survivors | 12 | **7** |

The automated gate caught the obvious cases. The manual pass caught what it
structurally cannot see (below). Both are enumerated in `screen_carriers.py`
(`MANUAL_EXCLUSIONS`) with per-sentence reasons.

**Implication for anyone reusing this stimulus set:** the paper's sentence pool
and its concept pool were not constructed to be independent. Several sentences
name a concept outright (`Snowflakes …`/Snow, `Fresh bread …`/Bread,
`Lightning flashed …`/Lightning), and many more entail one without naming it
(`… through the forest`/Trees, `sandcastles at the water's edge`/Oceans,
`The orchestra tuned their instruments`/Trumpets). If a design needs carrier and
concept to be unrelated, the pool must be screened or replaced.

---

## 2026-08-30 — Three failure modes of embedding-based contamination screening

Recorded because each one produced a wrong answer before being caught, and each
would recur in any similar screen.

**1. Layer choice is not free, and the readout layer is not the right one.**
Ranking the true concept for eight known-contaminated pairs across all 62
layers: layers 42–54 place it first essentially always; layer 40 (the
experiment's SAE readout layer) manages 2 of 3 non-lexical pairs and ranks
`Lightning flashed across the night sky.` **45th of 50** on suspiciousness —
it would have selected a transparently contaminated sentence as one of the
cleanest. Layer 31 is at chance (mean rank 20.3 of 50, chance 25.5).

The screen and the readout answer different questions — "what is this sentence
about" versus "what is the model representing while it writes" — and the screen
is not bound to the Gemma Scope layers because it uses no SAE. Screening at 43.

**2. Within-sentence z-scoring is blind to cluster contamination.** Standardizing
cosine within a sentence across the 50 concepts asks whether one concept stands
out. When several *related* concepts are elevated together they raise that
sentence's own mean and sd, suppressing every individual z.
`The orchestra tuned their instruments before the concert.` scored the **lowest**
max-z of ten selected carriers while being the most contaminated of them
(Trumpets, Harmonies and Xylophones are all concepts).

**3. Generic attractors.** A handful of low-frequency, morphologically heavy
nouns — Contraptions, Dirigibles, Monoliths, Xylophones, Amphitheaters,
Masquerades — rank in the top five for *any* bland sentence. Six unrelated
carriers (piano practice, furniture assembly, chef garnishing, crossword,
seashells, comedian) shared nearly the same top five. Within-sentence z therefore
measures how generic a sentence is, not what it is about.

Standardizing within *concept* (across sentences) removes attractors, but buries
broadly-evocative concepts: Oceans is close to many sentences, so its closeness
to `Waves crashed against the rocky shoreline.` is unremarkable *for Oceans* and
collapses from z 2.98 to 1.14. Neither marginal is sufficient alone. The gate
takes their **union**; `min` of the two ranks the known pairs 1–7 of 2500 and is
reported, but is not the gate.

**4. What no statistic caught.** Part-whole and entailment overlap is invisible
to all three variants: an orchestra contains trumpets, a forest is trees, an
antique vase is a treasure. These were removed by reading the survivors. Manual
exclusion only removes candidates, is enumerated in full, and was fixed before
any experimental data existed.

---

## 2026-08-30 — Activation capture was 87% of generation runtime

**Finding: a 66x speedup in the capture pass, bit-identical output. The full
grid drops from ~49 hours to ~7.**

Timing 20 real prompts showed generation at 1.00 s per trial and the
activation-capture forward pass at 6.61 s — for the same sequence, on a pass
that does strictly less work. Two causes in `irc/model.py::ResidualCapture`:

1. **The hook did `.float().cpu()` per layer.** With all 62 layers hooked that is
   62 synchronising GPU→CPU transfers inside one forward pass, each a pipeline
   stall, and the upcast doubled the bytes moved before the transfer.
2. **No `torch.no_grad()` around the forward.** The pass built an autograd graph
   across every layer. `model.generate()` applies `no_grad` internally, which is
   exactly why generation looked fast next to an explicit `model(...)` call.

Fix: `ResidualCapture(..., to_cpu=False)` keeps activations on-device in their
native dtype; the caller slices to the response tokens and makes **one**
transfer, inside `torch.no_grad()`.

| | before | after |
|---|---|---|
| capture | 6.56 s | **0.10 s** |
| per trial | 7.55 s | **1.09 s** |
| pilot (4,690) | 9.8 h | **1.4 h** |
| full grid (23,107) | 48.5 h | **7.0 h** |

Verified bit-identical, not merely close: `max|diff|` = 0 over a (62, 11, 5376)
tensor, and completions match. The slow path goes bf16 → fp32 → bf16 and fp32
represents every bf16 value exactly, so anything short of exact equality would
have meant the fast path was wrong rather than less precise.

The old behavior remains the default so upstream's pipeline is unaffected.
Note that `irc/pipeline.py::_generate_and_capture` still uses the slow path and
would benefit identically.

Generation is now 92% of per-trial time, so further gains would have to come
from batching the greedy decode.

---

## 2026-08-30 — Compliance is far better than upstream's

Upstream reports roughly one third of completions exact. On 20 prompts sampled
across all 67 phrasings, **19 of 20** were exact.

The single deviation is the interesting kind rather than a failure to copy: J3
(`de-emphasize`) produced the target sentence *and then* a meta-comment about
"the prompt's odd request". That is the behavior families N/P/Q/R/S exist to
probe, and this fork records it as a deviation-rate outcome rather than
discarding the trial.

Caveat: n=20, one carrier-length regime, and no nonce conditions were sampled.
Superseded by the pilot below, which put pooled deviation at 6.8% over 4,627
trials — still far better than upstream's, but the per-cell picture is where the
interest turned out to be.

## 2026-08-30 — Teacher forcing: four things that were wrong, in order

The teacher-forced pass looked correct at every stage and was wrong at three of
them. Each error produced output that was plausible on its face, which is why
they had to be found by deliberate checks rather than inspection.

### 1. Near-zero forced surprisal is ambiguous

Compliant trials returned surprisal of zero to float32 precision. That is equally
the signature of a correct implementation on a very confident model **and** of an
off-by-one where the model is scored on a token it has already seen. Reading the
indexing does not settle it; both stories predict the same output.

Forcing an unrelated sentence against an identical prompt separates them:

| forced target | mean surprisal |
|---|---|
| the correct carrier | 0.0000 nats/token |
| `Purple hexagons fermented the quarterly ledger.` | **20.05** nats/token (p ≈ 2e-9) |

Indexing correct. Gemma reproduces the carrier with near-certainty when the
source sits ~20 tokens back — induction doing what induction does.
`--verify-surprisal` re-runs this.

### 2. float32 cannot represent the result

`log_softmax` in float32 cannot express log(p) between 0 and about -1.2e-7, so
any p above ~0.99999988 reads as exactly zero. Two compliant conditions with
genuinely different surprisal both report `0.0000` and are indistinguishable.
Now computed in float64. The real floor is the bf16 logits upstream, so
differences far below ~1e-3 nats are not trustworthy either way.

### 3. Forced-token surprisal is blind to the deviation it exists to detect

N1 (`juggle X`) reproduced the carrier **correctly** and only then appended
*"...and yes, the satellites are definitely being juggled"*. All seven forced
tokens therefore scored ~0 and the trial was indistinguishable from a fully
compliant one. The entire deviation sat one token past the end of what was being
scored.

Fix: append `<end_of_turn>` to the forced sequence and score it. Reluctance to
**stop** is where deviation shows up under forcing.

### 4. Stop surprisal conflates formatting with deviation

Measured across all 67 phrasings (before the capture fix below):

| stop surprisal | preferred token | trials | reading |
|---|---|---|---|
| 12.1, 5.5, 5.3 | `' '` | T3, N1, T4 | **real** — all three deviated under free generation |
| 1.0 – 3.4 | `'\n'` | 11 trials | **formatting** — Gemma's trailing-newline habit |
| < 0.7 | `<end_of_turn>` | the rest | stop is the top choice |

The middle band is not deviation. Those eleven wanted the newline they habitually
emit before stopping, and every one of them was `exact_match=True`.

### 5. Excluding whitespace inverts the answer

The obvious fix — mass on continuing with something that is neither the stop
token nor whitespace — is wrong, and wrong in the direction that matters. N1
scored `P(content) = 0.0000`, the *lowest* possible, while being one of the three
trials that genuinely continued.

**Whitespace is a prefix to continuation, not an alternative to stopping.**

    compliant:  carrier -> '\n'  -> <end_of_turn>
    N1:         carrier -> ' '   -> '\n\n' -> '*(...and yes, the satellites'

Both put their mass on whitespace at the stop position. One token of lookahead
cannot separate them, and excluding whitespace scores the continuing trial at
zero.

Fix: walk the greedy path forward while it stays whitespace (at most 3 steps),
accumulating P(stop) along the way. `p_stop_soon` answers "would this have ended
the turn"; `p_stop_direct` is the immediate P(stop); `ws_path` and
`after_ws_token` record the route taken and what it wanted to write next. One
extra forward pass in the common case.

### What the stop measure is actually for

All 67 generated completions begin with the carrier **verbatim** — token ids
compared, not stripped strings. So on the captured span, teacher forcing imposes
nothing the model would not have written, and the pre-registered neutrality check
on the activations passes trivially and correctly.

The stop measure is therefore not testing that. It is a **graded** measure of
deviation propensity where `exact_match` is binary: a trial 60% likely to
continue that happened to stop is indistinguishable from one 1% likely, under
exactness alone. `p_stop_soon` separates them, which is what makes per-cell
deviation rates informative rather than merely countable.

### Cosmetic, but it is what exposed the above

The verbose line printed `P(stop)=0.8158; prefers '<end_of_turn>' p=0.816` —
the same number at two precisions, with wording implying the model preferred
something other than stopping when stop *was* its top choice. Chasing that
apparent discrepancy is what surfaced the whitespace problem underneath.

---

## 2026-08-30 — `exact_match` hid a trailing token that was diluting the readout

**Finding: 11 of 67 compliant trials wrote 8 tokens, not 7. Activation capture
averaged a trailing newline into the dependent variable for those 11 and not for
the other 53 — and which trials is condition-dependent.**

Same root cause as the stop-surprisal contamination above, but this one damages
the DV rather than a robustness check. Exactness is scored on
`completion.strip() == target`, so a trial writing the carrier plus a trailing
`\n` is recorded as exact while having produced an extra token. `n_resp_tokens`
counts everything up to `<end_of_turn>`, and capture used that span:

| generated `n_resp` | teacher-forced `n_resp` | count | stored act shape |
|---|---|---|---|
| 7 | 7 | 53 | `(4, 7, 5376)` |
| **8** | 7 | **11** (all `exact_match=True`) | `(4, 8, 5376)` |
| 21–23 | 7 | 3 (the real deviations) | `(4, 21..23, 5376)` |

Five distinct shapes where there should be one.

1. **Condition-correlated dilution.** The pooling rule is "mean over response
   tokens". For 11 trials that mean includes a semantically empty newline's
   activation; for 53 it does not. The affected trials are a specific subset
   (C1, C4, D1, D3, I4, J3, N3, S1, S2, S3, T7), so the dilution varies with
   condition — the one thing a nuisance must not do.
2. **The two passes were not comparable.** Generated captured 8 vectors where
   teacher-forced captured 7 for the same stimulus.

Fix: capture exactly `min(n_resp, len(target_tokens))` positions in both passes.
The full completion and `n_resp_tokens` are still recorded, with
`n_capture_tokens` alongside. After the fix every acts file should be
`(4, 7, 5376)` — one shape, both passes, no exceptions.

**Generalizable lesson:** `.strip()` in a correctness check discards exactly the
tokens whose presence changes the measurement. Compare token ids, not stripped
strings, wherever the token count feeds the readout.

## 2026-08-30 — The pilot: deviation is concentrated, and the registered prediction was half right

Generated pass: 4,627 trials, 2.43 h, **315 non-exact (6.8%)**. Teacher-forced
pass: 4,627 trials, 2.30 h. Pooled deviation is far under the 25% scaffold
trigger; the interest is entirely in the per-cell spread.

| cell | dev% | | cell | dev% |
|---|---|---|---|---|
| **T4** `. . . . not X` | **91.4** | | P `do not laminate X` | 3.1 |
| **T3** `. . . . . X` | **90.0** | | Q `glorf X` | 2.9 |
| **C** `do not ignore X` | **22.1** | | I `set X aside` | 2.0 |
| **N** `juggle X` | **14.0** | | B, R | 1.9, 1.4 |
| T1 `X` | 7.1 | | E F G H K L M S T2 T5 T6 T7 | 0.0 |
| J, A, D | 6.8, 5.7, 5.0 | | | |

**The registered prediction (N/P/Q/R highest) was half right, and the wrong half
is the finding.** N is elevated at 14.0% as predicted, but P (3.1%), Q (2.9%) and
R (1.4%) are not. **Negation suppresses deviation on incoherent instructions**:
N 14.0% → P 3.1%, Q 2.9% → R 1.4%. `do not juggle X` is trivially satisfiable by
doing nothing, while `juggle X` invites an attempt:

    The train arrived precisely on schedule.
    *(...and yes, the satellites are definitely being juggled. It

So the mechanism was right — non-mental imperatives invite action — but only for
un-negated ones. G (`do not think about X`), predicted to sit in the middle, is
at 0.0%.

**C at 22.1% was unpredicted and is the highest non-baseline cell.** `do not
ignore X` makes the model *demonstrate* not-ignoring:

    The basketball bounced off the rim.
    (Acknowledging the constant, silent presence of numerous satellites orbi...

Within-cell variation is large — C3 (`do not set X aside`) is 0.0% while C1
(`do not ignore X`) is 38%. That is exactly what Q10 exists to detect, and it
means the cell-level number hides most of the story.

---

## 2026-08-30 — Concepts leak as emoji, which the word-form detector cannot see

30 completions contained 🎺 for the concept `trumpets`; the leak detector caught
**3**.

    The train arrived precisely on schedule. 🎺        <- do not skip past trumpets

The concept was reaching the output pictorially. Spans C, D, B, T1, N and I —
not confined to one condition. `irc/concepts.csv` gains a `forms_emoji` column,
21 of 50 concepts populated, derived from Unicode character names and hand-pruned
for the same reason the WordNet links needed pruning: `fountains` → 🖋 is a pen,
`information` → 💁 is a person, `deserts` → 🏝 is an island. Recovers exactly the
27 missed, no false positives. Pilot leak rate 5.66% → 6.25%.

Note `\b` never matches around an emoji — it is defined between a word and a
non-word character — so `\b🎺\b` finds nothing. Words keep boundary matching;
emoji match literally.

Nothing needed re-running: leak is derived from the stored completion.

---

## 2026-08-30 — T3/T4 was a copy-boundary failure, and the fix is the scaffold

**The symptom.** T3 and T4 deviate on 90%+ of trials by reproducing the entire
third line, scaffold prefix included:

    The train arrived precisely on schedule.
    While you write it: . . . . . satellites.

T5 (`not . . . . X`) deviates on **0%**. The difference is that T5's line starts
with a word.

**First hypothesis — the filler token.** Six fillers x three arrangements x five
concepts x three carriers, 270 trials:

| filler | T3 | T4 | T5 |
|---|---|---|---|
| `.` (control) | 100% | 93% | 0% |
| `-` | 100% | 100% | 0% |
| `blah` | 57% | **0%** | 0% |
| `XXX` | 71% | 7% | 0% |
| `x` | 71% | 21% | 0% |
| `hmm` | 50% | 29% | 0% |

The control reproduced the pilot, so the test is valid. Dash fails identically to
dot, so it is punctuation generally rather than ellipsis specifically. **T4 is
fixed by a word filler; T3 is not fixed by any of them.**

**Why.** T4 and T5 both contain `not` — a real function word. T3 is pure filler
plus a noun, with no instruction-like element at all, so there is no word to swap
in. The failures show the model narrating rather than copying, paraphrasing the
instruction into the past tense:

    (While I wrote it: blah blah blah blah blah amphitheaters

That is what a model does when nothing in the line is actionable. Consistent with
T1 (bare `X`) deviating only 7.1% — a bare concept is fine; padding it with five
filler tokens is what breaks it.

**Second hypothesis — the copy boundary.** If the model cannot tell where the
text to reproduce ends, explicit delimiters should help. Four tag wordings
against the plain scaffold, 210 trials:

| | plain | tag1 | tag2 | tag3 | **tag4** |
|---|---|---|---|---|---|
| total deviation | 38% | 95% | 95% | 83% | **0/42** |
| tag echoes | 0 | 40 | 36 | 0 | **0** |
| T3 with dots | 100% | 100% | 100% | 100% | **0%** |

The naive tagged scaffold is **catastrophically worse** — it breaks cells the
plain scaffold handles perfectly (T1: 0% → 100%) by echoing the delimiters:

    '<sentence>The train arrived precisely on schedule.</sentence>'

The sentence is exact; only the markup leaks. That still destroys the
measurement, since capture takes the first N response tokens and those would be
markup.

The wordings differ by one clause. tag3 (`Do not include the tags themselves`)
stops the echoing completely and still deviates 83%, appending the instruction
line or commentary. tag4 names **both** failure modes:

    The tags below mark a sentence. Output that sentence alone -- no tags, no commentary.
    <sentence>{carrier}</sentence>
    While you write it: {frame}.

**tag4 fixes T3 with the original dot filler, 100% → 0%** — which no filler
achieved. If it holds, that is one amendment (the scaffold) rather than two, and
T3/T4/T5 keep their `PLAN.md` templates untouched.

**Confirmed and adopted (2026-08-31).** `scaffold2` ran both scaffolds over 14
phrasings x 10 pilot concepts x 2 carriers, 524 trials, using the real templates
from `conditions.csv`:

| phrasing | plain | tag4 | | phrasing | plain | tag4 |
|---|---|---|---|---|---|---|
| T3 | 95% | **5%** | | J3 | 25% | **0%** |
| T4 | 90% | **0%** | | C1 | 20% | **0%** |
| N1 | 50% | **0%** | | C4, N3 | 15% | **0%** |
| A1 | 30% | **0%** | | T1, D1 | 10% | **0%** |
| **TOTAL** | **27.5%** | **0.4%** | | G1 I5 L1 | 0% | **0%** |

Every phrasing improved, none regressed, no delimiter echoes. Leak rate
24.0% → 0.4%. The failure I was watching for did not appear: `no commentary` does
not collide with instructions that ask the model to act, and N1 (`juggle X`) went
50% → 0%.

Three honest limits, all recorded in `PREREGISTRATION.md`:

- **T3 is rare, not fixed.** 1 trial in 20 still appended its instruction line.
- **Deviation depends on the carrier.** The plain scaffold's rates here differ
  from the full pilot in both directions (A1 30% vs 5.7%, C1 20% vs 38%) because
  this used 2 of 7 carriers. The paired comparison within a run is sound; the
  absolute rates are not the cell's true rate.
- **The result belongs to this wording, not to delimiters.** The naive tagged
  form was *worse than the incumbent*.

Adopting it costs a full pilot re-run — every activation in `pilot1` was captured
under the old scaffold — and moves `n_prompt_tokens` from 18-40 to 32-54, which
is itself a pre-registered dilution covariate. `irc/conditions.csv` is untouched:
T3/T4/T5 keep their `PLAN.md` templates, so the filler work stands as a
documented dead end rather than a change.


## Open items

- `carrier_similarity.csv` and `stimuli.csv` are not yet generated.
- `PREREGISTRATION.md` is committed but its date and commit-hash header are
  placeholders until the run begins.
- **The T3/T4 scaffold decision is open**, pending `scaffold2`. If tag4 holds,
  the amendment is the scaffold alone and the condition templates are untouched;
  if it fixes only the T-cells, C and N stay as findings rather than defects.
- **The pre-registered scaffold A/B is now mis-specified.** Its four candidates
  all strengthen the output constraint, which the pilot showed is the wrong
  lever — the failure is the copy boundary. Its selection rule also implies
  re-running the 4,627-trial pilot per candidate (~10 h for four); a reduced
  comparison set is what actually happened and what should be written down.
- **The measure stage does not exist yet**, and it has two requirements the
  runner has already created: take the LAST record per `prompt_group` (re-runs
  append rather than replace), and recompute leak from stored completions rather
  than trusting the `leaked_concepts` field, which predates emoji support in
  `pilot1`.
- `irc/conditions.csv` and `irc/concepts.csv` live under `irc/`, while
  `CLAUDE.md`'s file table implies the repo root. Harmonise the paths or the doc.
- SAE width: `CLAUDE.md` records 16k as "the only variant Neuronpedia indexed",
  but the SAELens registry lists 65k / 262k / 1m at `l0_medium`. Unverified.
- ~~Concept split size disagrees between documents.~~ **Resolved 2026-08-30:
  10 pilot / 40 held out, all 50 concepts used.** `PLAN.md` §5 and `CLAUDE.md`
  amended from 10/30. Reason: n=10 reserve detects only dz > 0.9 and cannot serve
  as a confirmatory set, while costing 15% of power on the headline contrast
  (Holm-corrected detectable dz 0.58 at n=40 vs 0.67 at n=30). Split verified
  stratified by grammatical number: pilot 3 mass / 7 plural, held out 14 / 26.
