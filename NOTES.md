# Working notes and incidental findings

Chronological record of things discovered while building the experiment that are
not themselves experimental results. Nothing here is confirmatory. Findings that
bear on pre-registered choices are cross-referenced from `PREREGISTRATION.md`.

---

## 2026-08-30 — The paper's carrier set is largely unusable for this design

**Finding: 43 of Anthropic's 50 carrier sentences are contaminated by at least
one of the 50 concept words. Seven survive.**

**Which prior art this criticizes: none of it.** Two earlier versions of this
note got the reasoning wrong in opposite directions, so the resolution is worth
stating carefully. The key fact is that **three different stimulus sets are
involved.**

| | carriers | concepts | conditions |
|---|---|---|---|
| Lindsey (2025) | the 50 sentences | the 50 concepts | think / don't-think |
| Gurnee et al. (2026) §3.2 | their own | citrus fruits, `3² − 2`, line widths | focus / ignore / no-instruction |
| **this fork** | **7 of Lindsey's 50** | **Lindsey's 50** | **Gurnee's grid, decomposed** |

`irc/words_paper.py` is transcribed from **Lindsey**, whose intentional-control
experiment is think / don't-think. Neither instruction makes a claim that can be
false, so carrier–concept overlap adds variance without invalidating anything.
Lindsey never needed the two lists independent.

**Gurnee's stimuli are their own and are not contaminated** — their protocol
copies a sentence about a crooked painting while holding *citrus fruits* in mind,
and they verify the point directly: the no-instruction baseline rate is
approximately zero, which they describe as confirming the prompt context alone
does not put the target concept in the readout. Since the context does not evoke
the concept, "X is irrelevant to this task" is *true* when they assert it, so
there is no false-statement confound in their design either.

**Gurnee do use the declarative.** Their `ignore` condition's canonical phrasing
is `X is irrelevant to this task` — stated that way both in the main text and in
the phrasing appendix. Section 3.2's body writes it as "ignore X", which is
shorthand for the condition rather than the template; the appendix is
unambiguous. So the fork's family L is the same *form* as their headline
condition, which is what makes the comparison meaningful at all.

**But their stimuli are different and, as far as we can tell, unpublished.** The
paper gives illustrative examples only — a carrier sentence about a crooked
painting, with *citrus fruits*, `3² − 2`, or a line width as the target — across
three task families. Not published, and not locatable from the page:

- the carrier sentences and target concepts actually used across the many trials;
- the five to eight phrasing templates per condition (they sit behind hover
  tooltips absent from the page source);
- even the trial count for Figure 10 — other figures state theirs (n=100, n=90,
  n=24, n=8), that one does not.

There is no supplementary-data, release, or repository link anywhere in the page.

Two consequences worth carrying into the writeup. **Their contamination claim
cannot be independently checked** — the ~0 baseline is good evidence, but it is
their measurement of stimuli nobody else can inspect. And **phrasings cannot be
aligned**: this fork's 67 phrasings were written from the paper's canonical forms,
so overlap with their actual templates is unknown, and any per-phrasing
comparison to their spread is approximate.

**The exposure is created by the recombination.** This fork puts Gurnee's
relevance declaratives onto Lindsey's carriers, and those carriers overlap the
concept list because nothing ever required them not to. Neither source paper was
careless; the problem is new, and it belongs to the fork.

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


## 2026-08-31 — The scaffold amendment, validated at full scale

`pilot2` re-ran the whole pilot under the tagged scaffold: 4,627 trials, 2.70 h,
**2 non-exact**. Against `pilot1`'s 315 under the plain scaffold, on identical
stimuli and concepts.

| | pilot1 (plain) | pilot2 (tagged) |
|---|---|---|
| deviation | 315/4,627 = **6.81%** | 2/4,627 = **0.04%** |
| leak | 262/4,627 = **5.66%** | 1/4,627 = **0.02%** |
| T4 | 91.4% | **0.0%** |
| T3 | 90.0% | **1.4%** |
| C | 22.1% | **0.0%** |
| N | 14.0% | **0.0%** |

Every other cell is at exactly 0.0%. The 524-trial `scaffold2` comparison
predicted this and understated it.

**The two survivors are different failure modes.**

`T3` (`. . . . . vegetables`, carrier 2) is the known residual — appending the
filler line. It is the *same trial* that failed under tag4 in `scaffold2`, same
concept and carrier, so it is reproducible rather than random.

`J3` (`de-emphasize satellites`, carrier 3) is new and is an **omission**:

    target: 'The chef garnished the plate with fresh herbs.'
    got:    'The chef garnished the plate with herbs.'

Every failure chased so far has been *addition* — commentary, echoed
instructions, markup. This dropped a word. It registers as a deviation and
produces no leak. Whether `de-emphasize` causing an adjective to vanish means
anything is not answerable at n=1, but it is worth grepping the held-out run for
completions that are a strict subsequence of their target.

---

## 2026-08-31 — The trailing-token trim was preventing a confound, not noise

The capture-alignment fix trims tokens beyond the carrier's own span. At pilot
scale it fired on **383 of 4,627 trials (8.3%)**, and the distribution is what
matters:

| cell | trials with a trailing token trimmed |
|---|---|
| T3 | 67.1% |
| T4 | 50.0% |
| **A** (`concentrate on X`) | **45.7%** |
| J | 16.8% |
| M, S, L | ~1% |
| H, T2, T5, T7 | **0%** |

A 45-point spread across conditions. Without the fix, cell A would have carried
an extra semantically-empty newline's activation in its mean on 46% of trials
while H, T2, T5 and T7 carried it on none.

**A is the focus condition and `A > T1 > T7` is Q0's hard gate** — the contrast
that blocks every other one if it fails. A dilution hitting A on 46% of trials
and T7 on 0% pushes directly against the ordering the gate exists to test. That
is the difference between a nuisance and a confound, and it landed exactly where
it would have done most damage.

The smoke test showed 11 of 67 with no visible structure. At 4,627 the structure
is unmistakable. Verified at scale: `capture count is constant within each
carrier` — every trial of a given carrier now contributes the same number of
activation vectors (7, 9, 10 or 11 by carrier length).


## 2026-08-31 — The analysis layer is chosen at one SAE width and applied at another

`PREREGISTRATION.md` fixes the pilot at 16k and the confirmatory run at 262k, and
separately fixes the analysis layer as whichever of 16/31/40/53 maximizes A-vs-T7
separation **on the pilot**. Those two commitments interact in a way neither
states: **the layer is selected using 16k separation and then applied to held-out
data measured with 262k SAEs.**

That assumes the best layer at 16k is also the best layer at 262k. Probably safe
— A-vs-T7 separation is a coarse property of where in the network a concept is
legible, and it should not flip between two decompositions of the same residual
stream. But it is an assumption, not a derivation, and the pilot cannot check it
because the pilot never sees 262k.

Worth a line in the writeup rather than passing unremarked. Cheap ways to firm it
up, if GPU time allows later:

- re-run the pilot's SAE readout at 262k for the chosen layer plus its nearest
  rival, and confirm the ordering holds (needs 262k selection at two layers
  rather than one, so ~2x that download);
- or report the full layer curve at 16k alongside the held-out result, so a
  reader can see how flat or peaked the choice was — a flat curve makes the
  assumption nearly free, a sharply peaked one makes it load-bearing.

Related sequencing, recorded because it is easy to get wrong: the SAE **readout**
(encoding stored activations) is a small matmul and runs fine on CPU. The SAE
**latent selection** needs the GPU, because it runs template prompts through the
model. So held-out measurement is not GPU-bound; only the 262k selection that
precedes it is.


## 2026-08-31 — Held-out compliance holds, and `information` is a special case

Held-out generation: 18,487 prompts, **33 non-exact (0.18%)** against pilot2's
0.043%. That 4.2x gap is almost entirely **one concept**:

| | non-exact | rate |
|---|---|---|
| all held-out | 33/18,487 | 0.18% |
| **excluding `information`** | **10/18,187** | **0.055%** |
| pilot2 (10 concepts) | 2/4,627 | 0.043% |

So the tagged scaffold generalized to 40 unseen concepts. 23 of the 33 deviations
are `information`, and they split into two mechanisms.

**1. The instruction is applied to the carrier rather than to a mental concept.**

    I4  'leave information out'     -> 'The train arrived.'
    J3  'de-emphasize information'  -> 'The train arrived on schedule.'

The model obeyed — by deleting words from the sentence it was copying.
`information` is **meta-linguistic**: the carrier *is* information, so
"leave information out" has a coherent reading as an instruction about the
copying task itself, and the away-instruction lands on the text instead of on a
concept.

This is the same mechanism as pilot2's single J3 deviation, which dropped "fresh"
from "the plate with fresh herbs". That was recorded as possibly idiosyncratic;
it is not. **Omission is a real failure mode**, and it inverts every other one
chased in this project — commentary, echoed instructions, markup were all
*additions*.

**2. Incongruent verbs invite explanation of the verb.** N4 (`braise
information`) and N5 (`centrifuge information`) append encyclopedic definitions
of braising and centrifugation. That is the registered N-family prediction, with
the "action" being to explain rather than to perform.

### What follows for the analysis

**No post-hoc exclusion.** `information` sits in the held-out 40, so this was
invisible from the pilot — none of the 10 pilot concepts is meta-linguistic.
Dropping it now would be excluding on an outcome after seeing held-out data,
which `PREREGISTRATION.md` forbids and which the pilot/held-out split exists to
prevent. It is reported.

**A hypothesis for the writeup, not a filter.** Concepts whose word has a
meta-linguistic reading in an instruction context — `information` most acutely,
plausibly `secrecy`, `illusions`, `memories` — are ambiguous between "hold this
in mind" and "this is about the text in front of you". That ambiguity is a
property of the concept list, not of the manipulation, and it is checkable: the
prediction is that away-directed instructions on such concepts produce
*omissions* from the carrier, which no other concept should show.

**Detection.** Omission passes `exact_match` as a deviation but produces no leak
(12 of 33 held-out deviations leaked; the omissions are among the 21 that did
not). A completion that is a strict subsequence of its target is the signature
worth grepping for, in both runs.


## 2026-08-31 — What the readouts and pooling actually do, and why rule 3 is degenerate

Recorded partly as plain-language reference, because the pipeline has enough
stages that the *purpose* of each is easy to lose, and partly because writing it
out is what exposed that one pre-registered choice cannot work here.

### What is stored, and what the measure stage does with it

For every trial, the run saved the model's internal state while it wrote the
carrier: at 4 layers, for each of the ~7-11 tokens produced, 5,376 numbers.
Nothing in that says "satellites" — individual dimensions are not interpretable,
because concepts are smeared across many at once.

The measure stage translates, two independent ways:

- **Concept vectors** (secondary readout). A single "satellites direction" in
  5,376-dimensional space, built by averaging the model's state when reading the
  word in the four `WORD_TEMPLATES_V1` sentences and subtracting a baseline. Per
  token, take the angle between the model's actual state and that arrow.
- **SAE latents** (primary readout). The sparse autoencoder re-expresses the
  5,376 numbers as which of 16,384 named features are firing. The 5 most
  concept-selective are pre-picked per concept — Satellites layer 16 latent
  16085 is labelled "GPS and satellites" — and the measure asks how hard those
  fired.

Either gives one number per (trial, layer, token).

### Why pooling exists

Contrasts need one number per **trial**. The concept is unlikely to be equally
active across all seven tokens of `The train arrived precisely on schedule.`, so
if it spikes at one position and is absent at six, a flat average dilutes it
toward nothing. Three rules were pre-registered; the collapse rule is the choice.

### Rule 3 is degenerate for a copying task

Rule 3 reads the concept out where it could plausibly have surfaced, weighting
each position by P(the concept's first token | prefix). Measured across the whole
pilot:

| | |
|---|---|
| median P(concept next) | **6.3e-23** |
| 99.9th percentile | 1.0e-08 |
| **highest of 42,965 positions** | **2.9e-06** |
| within-trial max/min weight ratio | **7e+16** |
| positions with P > 1e-4 | **0** |

The model is copying a fixed sentence under an explicit "output that sentence
alone" instruction and complies on 99.96% of trials, so the probability of
emitting the concept next is essentially zero **by construction, at every
position**. The weights span sixteen orders of magnitude within a single trial,
all vanishing — so the "weighted mean" is not weighting anything. It selects
whichever position happened to have the least-astronomically-small value, and at
1e-23 those differences are softmax far-tail numerics, not signal.

**The rule assumed a task in which the concept might plausibly appear. This
design forbids it.** That is a property of the design, not of the model or the
data.

### Consequences

- **Pooling is `token_mean`** — the only non-degenerate eligible rule, `topk_mean`
  having been dropped earlier for selecting on the quantity being measured.
- **Held-out rule 3 is unnecessary**, saving ~28 min of GPU.
- Rule 3's per-trial cost, for the record, was **0.09 s/trial** — 4,627 trials in
  7 minutes against a 2.2 h estimate. The estimate came from the teacher-forced
  pass at 2.08 s/trial, which also captured activations, ran three whitespace
  lookaheads, and wrote 300 KB per trial. Rule 3 does one forward pass and
  appends to a dict. Worth remembering that per-trial estimates transfer badly
  between stages that look similar.

This is the pilot working as designed: a pre-registered option eliminated on
principled grounds, **before any effect size was computed and without touching
held-out data**. A stronger position than "token_mean won a comparison" — the
claim is that `plausible` is undefined for a copying task.


## 2026-08-31 — The T7 baseline was collapsing to one concept out of fifty

The measure stage ran for the first time today and hit three bugs (below). This
one was not a crash — it would have produced a complete, plausible-looking
parquet with the baseline silently missing.

### What T7 is

T7 (`base_absent`) is the do-nothing condition: the prompt says write this
sentence, and stops. No third line at all. It answers "how present is this
concept when nothing whatsoever drew attention to it?"

Because its prompt never names a concept, the prompt does not depend on which
concept we are asking about. Seven carriers means exactly **seven distinct T7
prompts**, and the run stores seven activation tensors.

But measurement asks a per-concept question — "how strongly is *satellites*
represented in this activation?" — and that question can be asked of the same
neutral activation once per concept. Same tensor, fifty different questions.
That is what makes it a baseline: one neutral state, compared against each
concept in turn. `stimuli.csv` books this correctly as 350 rows (7 carriers ×
50 concepts) sharing 7 prompts.

### What went wrong

`load_stimuli` built a lookup keyed by prompt (`prompt_group`) with
`setdefault` — first row wins. Of the 50 concepts attached to each T7 prompt,
**49 were discarded.** T7 came out measured against one arbitrary concept per
carrier: 7 readout values where there should be 350 (70 within the pilot split).

For 9 of the 10 pilot concepts there was no T7 baseline at all.

### Why it mattered

The pre-registration derives *both* Stage 2 values — analysis layer and pooling
rule — from "A-vs-T7 separation" on the pilot. That comparison needs an A value
and a T7 value for the same concept. It was not computable. Neither were the
confirmatory contrasts that include T7: **Q0, the gate** (`A > T1 > T7`), plus
Q1 and Q7.

Nothing would have complained. The parquet would have had ~60,000 rows and the
7 stray T7 rows would have looked like a small cell, not a broken one.

### The fix, and why it is nearly free

Keep every row per prompt and loop over them. The activation is loaded once and
SAE-encoded once per (trial, layer); only the concept vector or latent set
applied to it changes. So expanding T7 fiftyfold costs 50 column gathers, not
50 SAE encodes — the encode is hoisted above the concept loop deliberately.

### The general lesson

Any condition whose prompt omits the concept is shared across concepts, and a
prompt-keyed lookup will collapse it. Upstream has the same structure and
handles it: its `no_mention` is shared across words per sentence, and the viewer
ships "word-independent `no_mention` null bands, deduplicated". The design was
right; only the fork's measure code collapsed it.

Note the contrast with **T6** (`floor_control`), which names a *partner* concept
(`fountains are irrelevant to this task` while the target is `dust`). Its prompt
*does* vary by concept, so it has 350 rows over 350 prompt_groups and was never
affected. Shared-ness follows from whether the concept appears in the prompt,
not from whether the condition is a baseline.

### The other two bugs in the same stage

- **Capitalization.** The concept-vector bank keys words as
  `irc/words_paper.py` stores them (`"Satellites"`); `stimuli.csv` carries the
  lowercased form used in prompts. Every lookup missed, and the stage wrote a
  **0-row parquet and reported success.** The SAE half already lowercased —
  that side had been fixed and this one missed. There is now a guard that raises
  rather than writing an empty file.
- **The tail summary** rebuilt its DataFrame from the concept-vector rows
  whichever readout had run, so an empty vector half crashed the summary for a
  working SAE half, and `--readout sae` alone would have raised
  `UnboundLocalError`. Each readout now summarizes itself.

Two of the three were silent-wrong rather than loud-wrong, which is the argument
for running the stage on 20 trials and *reading the row counts* before running
it on 27,674.

## 2026-08-31 — The latent selection is ragged, and 9 cells are empty

Discovered by a guard, not by inspection. Vectorizing the SAE readout meant
gathering all of a trial's concepts in one indexing operation, which assumes
every concept has the same number of selected latents. It asserted that and
died:

    latent selections have unequal k at layer 16 ([1, 2, 3, 4, 5])

`select_latents` takes "top-k concept-selective latents" with k=5, but k=5 is a
**ceiling, not a count** — a latent has to pass the contrastive score and the
control-word exclusion to be kept, and often fewer than five do.

### The distribution

Selected latents per concept, all 50 concepts, latents_v2 at 16k:

| layer | k=0 | k=1 | k=2 | k=3 | k=4 | k=5 | mean k |
|---|---|---|---|---|---|---|---|
| 16 | 0 | 1 | 1 | 9 | 2 | 37 | 4.46 |
| 31 | **3** | 8 | 11 | 5 | 7 | 16 | 3.06 |
| 40 | **4** | 3 | 4 | 2 | 9 | 28 | 3.86 |
| 53 | **2** | 5 | 4 | 6 | 6 | 27 | 3.80 |

**92 of 200 (concept, layer) cells have k < 5.** Nine have **k = 0** — no
latent survived selection at all, so the SAE readout for that concept at that
layer does not exist.

### Two things this breaks

**1. k=0 must be missing, not zero.** `sum` over an empty index gives `0.0` and
`max` over an empty axis raises. A zero would enter the analysis as a genuine
measurement of "this concept is not active", when the truth is "we have no
instrument for this concept here". The readout now emits **no row** for those
cells and prints which they are.

**2. Layers cover different concept sets, so their dz values are not
comparable.** The pre-registered rule picks the analysis layer by A-vs-T7
separation across layers. But if layer 40 is missing four concepts and layer 16
is missing none, the two dz values describe different concepts — and a layer can
rank higher partly by having dropped its hardest concepts. The pre-registration
did not anticipate ragged selection, so it specifies no handling.

Handled by ranking the layers on the **concepts paired at every layer**, so the
four numbers describe one set, and reporting each layer's own-coverage dz beside
it. Logged as an amendment; the criterion (A-vs-T7 dz) is unchanged, only the
concept set it is evaluated on.

### It hits the pilot

Pilot concepts' k by layer — `Vegetables` has **k=0 at layer 40**:

    concept         L16   L31   L40   L53
    Amphitheaters   5     2     4     5
    Frosts          5     5     5     5
    Kaleidoscopes   5     5     5     5
    Rubber          3     1     5     4
    Satellites      5     4     4     5
    Secrecy         5     5     5     5
    Silver          3     1     4     5
    Trumpets        5     5     5     5
    Vegetables      5     2     0     3     <-- no readout at layer 40
    Xylophones      5     5     5     5

So the pilot layer comparison runs on 9 concepts, not 10. Layer 40 is a leading
candidate on depth grounds (~65%), which makes this the awkward case rather than
a hypothetical one.

### Converging evidence for the 262k width

`CLAUDE.md` argues for 262k on the grounds that 16,384 latents decompose too
coarsely to separate a target concept from 49 others. This is that argument
showing up as data: at 16k the selection cannot find even one concept-selective
latent for 9 of 200 cells, and averages 3.06 of a possible 5 at layer 31. The
`latent_sum` readout is also not comparable across concepts when k varies 1 to 5
— within-concept pairing protects the registered contrasts, but any
cross-concept statement about absolute latent activation is confounded by k.

This was recorded before the confirmatory run was configured, so the 262k switch
remains the pre-registered conditional it already was, with one more reason
behind it.

### Why the guard mattered

Both of these were silent failure modes. A rectangular gather would have thrown
on ragged input, so bug 1 was going to surface — but `sum` over an empty
selection returns a clean `0.0`, and nothing anywhere would have flagged that
nine cells of the primary readout were fabricated zeros.

## 2026-08-31 — The zeros are real: a positive control on the SAE readout

The pilot readout is exactly `0.0` for a lot of (concept, layer) cells — 86% to
98% of all readouts depending on layer — including in cell A, the strongest
instruction to think about the concept. That number is uninterpretable on its
own, because it is ambiguous between two opposite things:

- **instrument insensitivity** — the model represents the concept fine, but the
  five latents we selected do not detect it here, so `0.0` is a
  non-measurement and must not be read as "concept not activated";
- **a genuine null** — the latents are valid detectors and the concept really is
  not activated while the model copies an unrelated carrier sentence.

Nothing in the experiment distinguishes them. `rk_scripts/12_latent_positive_control.py`
does: it measures the same latents in the context they were **selected** in —
the four `WORD_TEMPLATES_V1` prompts, at the concept's own token positions. If a
latent will not fire there, it will not fire anywhere.

### Result: zero SELECTION_DEAD at every layer

Every selected latent fires in the selection context, and fires hard:

| concept | control L40 | task L40 | control L53 | task L53 |
|---|---|---|---|---|
| silver | 9649 | 496 | **9965** | **0.0** |
| kaleidoscopes | 4081 | 247 | 3116 | **0.0** |
| rubber | 6270 | 162 | 3149 | **0.0** |
| trumpets | 11330 | 122 | 16028 | 1331 |
| secrecy | 8223 | 1139 | 6209 | 999 |
| vegetables | 0 (k=0) | — | 4149 | **0.0** |

`Silver`'s latents fire at ~10,000 on "Tell me about silver" and at exactly
`0.0` in every one of the 25 experimental conditions at layer 53.

**So the zeros are measurements, and the 92%-zero rate at the best layer is a
finding about the task, not a broken instrument.** Instructing the model to
focus on a concept while it copies an unrelated sentence largely fails to
activate that concept's SAE latents. That is a substantive result and it bears
directly on the fork's question — but it also means the readout has a severe
floor effect, which is a power problem rather than a validity problem.

Layer 40 is the exception: it is the only layer whose readout is informative for
every concept it can measure (9 of 9, against 6 of 10 at layers 31 and 53 and
3 of 10 at layer 16).

### Corollary: the coverage refinement was a no-op

Because nothing is SELECTION_DEAD, tightening the definition of "usable" from
`k>0` to "actually fires" changes no held-out count: 40 / 37 / 37 / 38 either
way. The worry that a concept might have three selected latents that never fire
was well-formed and simply false here.

---

## 2026-08-31 — A common-concept restriction that did the opposite of its purpose

Worth recording as a methodological trap, because it was introduced *to protect*
the comparison and silently inverted it.

The SAE selection is ragged, so layers cover different concept sets. The obvious
worry: an unrestricted per-layer dz would let a layer rank high partly by having
dropped its hardest concepts. The obvious fix: restrict the comparison to the
concepts paired at **every** layer. That was committed the same morning.

It is wrong, and by exactly the mechanism it was meant to prevent.

The common set excludes any concept that **any** layer cannot measure.
`Vegetables` has `k=0` at layer 40 — no instrument. But at layers 16, 31 and 53
it *is* measurable and reads `0.0 / 0.0 / 0.0` across A, T1 and T7. So the
restriction lets those layers **shed a concept they measure and get no signal
from**, while layer 40 — which never had it — gains nothing:

| layer | dz, own concepts | dz, common set |
|---|---|---|
| 16 | 0.483 (n=10) | 0.513 (n=9) |
| 31 | 0.581 (n=10) | 0.621 (n=9) |
| **40** | **0.711 (n=9)** | 0.711 (n=9) |
| 53 | 0.662 (n=10) | **0.713 (n=9)** |

Every layer's dz rises except layer 40's, which cannot rise because the concept
was already absent. **The argmax flips from 40 to 53.** The restriction
systematically rewards degenerate layers.

The correct accounting distinguishes two kinds of nothing:

- **measurable but silent** → **data**. Δ = 0, keep the concept.
- **no instrument (k=0)** → **missing**. Drop the concept.

That is what the rule did before the "fix", and it is the literal reading of the
pre-registration.

### The layer choice, and the honest sequence

Under the literal rule layer 40 wins (0.711 vs 0.662). Layer 40 also passes the
pilot Q0 ordering cleanly (A 173.1 > T1 24.7 > T7 1.8) while **layer 53 fails
it** (A 157.0, T1 157.3 — inverted). Since Q0 is already a hard gate in
`PREREGISTRATION.md`, passing it on the pilot is now an **eligibility
precondition** applied before the dz criterion, rather than coverage acting as a
general tiebreak: a layer the pilot shows failing Q0 cannot support Q1–Q10 at
all, so selecting it would guarantee no confirmatory analysis.

Layers 16 and 31 also pass Q0, so 53 is the only failure — and it was the one
the superseded rule selected, by 0.003.

**Sequence, stated because it matters:** pilot dz was computed *before* this
revision. The defect was found after seeing that the superseded rule picked 53.
What makes reverting it something other than fishing is that the defect is
mechanical and demonstrable — three of four layers' dz rose, and the one that
could not rise is the one that lost — rather than a preference about the answer.
Both dz tables are printed by `rk_scripts/11_stage2_choose.py` and stored in
`stage2_values.json`, and the amendment records all of it.

Layer 16 is a reminder to read the means and not just the test: it "passes" Q0
with A 1.75, T1 1.72, T7 1.57 — an ordering that holds on numbers too small to
mean anything, with 3 of 10 concepts informative.

## 2026-09-05 — The zeros are not spread evenly, and that decides what the analysis can test

The positive control (2026-08-31) established that exact-`0.0` readouts are real
measurements, not a dead instrument. This is the follow-on question, and the one
that matters for the analysis: **which contrasts still have anything left to
compare?** Measured on the **pilot only**, at the committed analysis layer 40,
`latent_sum`, `token_mean`.

    per-trial exactly zero            92.1%
    per-concept-cell exactly zero     65.8%  (148 of 225)
    concepts with any nonzero cell    9 of 9

The marginal rate understates the problem, because the zeros are **not spread
evenly across cells**. A paired contrast is uninformative for a concept when
*both* its cells read zero — the difference is then structurally 0, contributing
nothing while still consuming an n:

| contrast | informative concepts (of 9) |
|---|---|
| Q0 `A − T7` | 9 |
| Q3c `A − I` | 9 |
| **Q3 `I − G`** — the PRIMARY | **5** |
| Q5b `I − J` | 5 |
| Q1 `T1 − L` | 3 |
| Q4 `K − M` | 3 |
| **Q5e `L − M`** | **2** |

The pattern is coherent rather than random: cells that direct attention *toward*
the concept (A, and T1's bare mention) activate its latents; the *away* and
*declarative* cells largely do not. So the readout is most informative exactly
where the design needs it least — establishing that mention works — and least
informative where the fork's actual question lives.

**Q5e is `irrelevant` vs `not relevant`**, the morphological-versus-syntactic
comparison that motivates the whole fork, and it has two informative concepts of
nine.

Scaling to the held-out n=37 optimistically, the away-side contrasts would carry
perhaps 8-12 informative concepts, which moves detectable dz from the registered
~0.62 to somewhere above 1.0. Nothing in the design was powered for that.

### The interpretive trap this sets

Most contrasts will come back non-significant, and the available reading is
*"instruction framing does not change internal representation."* That reading is
not supported. The alternative — *"the readout is at floor and could not have
detected a difference"* — predicts the same p-values, and the analysis as
registered cannot separate them. A null from 2 informative concepts is not
evidence of absence; it is absence of evidence, and the two must not be reported
in the same voice.

### Does 262k fix it?

Probably not, and it is worth writing the prediction down before the money is
spent. The positive control showed the selected 16k latents fire at 3,000-16,000
on the concept's own tokens, so the 16k SAE **can** represent these concepts. The
zeros therefore mean the concept is not active during the copying task, not that
the decomposition is too coarse to see it. A finer decomposition catches weaker
activation below the 16k threshold, so 262k may raise sensitivity at the margin —
but it cannot manufacture activation that is not there. Expect an improvement in
degree, not a fix.

The nonzero values are also badly skewed — median 13.8, max 781, mean/median 3.9
— so the surviving comparisons rest on a few large values and the paired t's
standard deviation is set by two or three concepts. The BCa intervals will be
wide, which is the honest outcome rather than a defect.

### Consequence

This has to be resolved **before** the held-out set is unblinded, since any rule
chosen after seeing the contrast results is worthless. Options are under
discussion; whatever is chosen goes in `PREREGISTRATION.md` as an amendment
first. All numbers above are pilot-only.

## 2026-09-05 — Independent review of the two flagged implementation decisions

`14_confirmatory.py`'s docstring flags two decisions as unregistered implementation
choices needing sign-off: the family-combination rule (`FAMILY_RULE`) and
`MIN_INFORMATIVE = 15`. Reviewed before the held-out set was touched. Both were
accepted as sound in their core approach, each had one gap worth fixing, and
both fixes below are now implemented in `14_confirmatory.py` and amended into
`PREREGISTRATION.md` — still before the held-out set is unblinded, so neither
required re-deriving anything already committed.

### The combination rule can hide a sign-flipped interaction

`paired_mean_of` (Q3c, Q5b, Q5c, Q5f) tests the *average* of two per-concept
differences — mathematically the marginal main effect when the two comparisons
are the same factor contrast at two levels of another factor (e.g. Q5c: negation
composition, averaged over imperative vs. declarative frame). That is a standard
and correct way to test a main effect, and every sub-comparison is already
printed individually.

The gap: averaging is exactly the operation that **cancels an interaction**. If,
say, `C vs D` (Q5c, imperative) and `E vs F` (declarative) are each individually
significant but in *opposite* directions, the family-level Q5c number — the one
that is Holm-corrected and gets called significant/null — can come back null by
cancellation, while the two sub-comparisons underneath tell a genuinely
different story. Q5 and Q5d already carry an explicit interaction contrast
(`paired_interaction`) for exactly this reason; Q3c, Q5b, Q5c and Q5f do not,
and there is no principled reason the four "mean" members should be exempt from
the same check the two "omnibus" members get.

**Fix, implemented:** `paired_interaction` added for Q3c, Q5b, Q5c and Q5f,
reported alongside the existing sub-comparisons as a diagnostic. It is
explicitly **not** added to `FAMILY` — the Holm-corrected family stays at 15
members, untouched — so this closes the gap without reopening anything already
committed. Smoke-tested on `pilot2` (n=9, `--ignore-gate`): all four now print
their interaction line, e.g. Q5f's `interaction_(A-D)-(B-F)` at dz=1.191,
p=0.007 — consistent with its two sub-comparisons (dz 1.154 and 0.380) pointing
the same direction rather than canceling, which is the reassuring case, not the
one the fix was written to catch.

### `MIN_INFORMATIVE = 15`: the arithmetic is right, the approximation is mildly optimistic

Checked by hand: solving `0.60 * sqrt(40/n) = 1.0` gives n ≈ 14.4, so 15 is
exactly the smallest integer at which the stated rule — rescaling the
registered detectable-dz figure by `1/sqrt(n)` — crosses from "cannot detect
even a dz-1.0 effect" (n=14, required dz ≈ 1.014) to "can" (n=15, required dz ≈
0.980). The threshold is not a round-number guess; it is the correct integer
boundary for the formula as written.

The formula itself, though, is a normal (z) approximation to what is really a
small-sample (noncentral-t) power problem. At n=15 the relevant t-distribution
has only 14 degrees of freedom, where its tails are noticeably fatter than the
normal approximation assumes — so the *true* required dz to detect a "large"
effect at n=15 is probably somewhat above 1.0, meaning the real crossover n is
likely a couple of concepts higher than 15 (rough estimate: 17-18, not derived
exactly here). Since the entire purpose of this rule is to avoid overclaiming a
null under the floor effect, an approximation that is mildly anti-conservative
works against the rule's own goal.

**Fix, implemented:** the `1/sqrt(n)` rescaling is replaced with an exact
noncentral-t power calculation (`scipy.stats.nct` + `brentq`), at the same
Holm-worst-case alpha (0.05/15) and 80% power. The rough estimate above was
close: the exact crossover is **n = 19**, not 15. Sanity checks against the
approximation, both confirming the predicted direction (exact ≥ approximate,
gap widening as n shrinks):

| n | approx (1/sqrt(n) of 0.60 @ n=40) | exact |
|---|---|---|
| 40 | 0.600 | 0.632 |
| 19 | 0.870 | 0.986 |
| 15 | 0.980 | 1.155 |

`MIN_INFORMATIVE` is now derived, not hardcoded — `next(n for n in range(3, 100)
if detectable_dz(n) <= 1.0)` — so it cannot silently drift out of sync with
`detectable_dz`'s own formula again. This can only move the threshold up
relative to the old value, never down, so it is a strictly more conservative
version of the same rule, not a new one. Stated explicitly in the writeup,
independent of the exact cutoff: clearing the threshold licenses "no effect at
least this large," never "no effect," since `detectable_dz` is reported for
every contrast regardless of verdict.

Both fixes are recorded as a dated amendment in `PREREGISTRATION.md`
(2026-09-05, alongside this note) before the held-out set is unblinded, per the
project's own rule that methodology decisions are committed before the data
that would be affected by them. Neither touches the Q0 gate or any text already
committed there.

## 2026-09-05 — `--hybrid-compliant`: a third variant, built from data already on disk

Clarified the intent behind restricting held-out teacher-forcing to deviating
trials (2026-08-31 amendment): the surprisal quantities it was originally
written down for (forced-token surprisal, stop surprisal) are a method-validity
check and feed no confirmatory contrast, but the *activations* captured during
that same forced pass were meant to double as substitutes for the trials
`--compliant-only` would otherwise drop — this was never actually implemented
or written down as its own mode.

**No new measurement needed.** Checked `artifacts/runs/heldout1/teacher_forced/results/`:
both readout parquets (SAE, concept-vector) already exist, covering all 33
deviant `prompt_group`s at all four layers and both poolings — confirmed by
cross-referencing against the generated pass that every one of those 33 was
`exact_match = False` there and `True` in the teacher-forced pass. The measure
stage is CPU-only (reads stored activations, no model), so this is a pure merge
of two already-computed files, no GPU or pod required.

**Implemented** as `--hybrid-compliant` in `14_confirmatory.py`: for each
`prompt_group`, keep the generated row unless it deviated, in which case swap
in the matching teacher-forced row (falling back to the generated,
non-compliant value with a printed warning if no substitute exists at that
layer/pooling/readout — did not occur on held-out, all 33 have one). Mutually
exclusive with `--compliant-only`; requires the default `--pass generated`.
Smoke-tested on `pilot2` (`--ignore-gate`, since pilot's n=9 can't power the
gate): 1 trial substituted, 0 left without a match, ran cleanly end to end.

Distinct from `--compliant-only` in the direction that matters for the
`informative_n` fix below: `--compliant-only` *drops* trials and can shrink a
concept out of a cell entirely if every trial for that cell happened to
deviate, which is exactly the scenario where the counting mismatch could bite.
`--hybrid-compliant` *substitutes* instead of dropping, so no concept is ever
lost and that scenario cannot arise within this mode. Recorded as a
`PREREGISTRATION.md` amendment (2026-09-05) before the held-out set is
unblinded; reported as a second robustness check alongside `--compliant-only`,
not a replacement for the primary (all-trials) analysis.

## 2026-09-05 — `informative_n` fixed to match the test it describes

The counting bug flagged in review (a concept missing one side of a comparison
could still be counted "informative" if its other side was present and
nonzero, because `informative()` filled the missing side with `0` before
checking) is fixed, ahead of running `--compliant-only` on held-out where it
was live rather than hypothetical: unlike the primary readout (where a
concept's missingness is uniform across every cell, so the bug was inert),
`--compliant-only` filters per (concept, cell) independently and can leave one
side of a pair present while the other is missing.

**Fix:** `informative()` now requires a complete case across the cells being
compared (`dropna()`, no `fillna`), exactly matching what `paired()`'s
`(w[hi] - w[lo]).dropna()` and `omnibus()`'s `w[have].dropna()` already
require — so `informative_n` can no longer exceed the n the underlying test
actually used.

**Verified with a synthetic case** before touching any real data: concept X
present on one side only (5.0, NaN), Y present and nonzero on both (3.0, 2.0),
Z present and exactly zero on both (0.0, 0.0). Old code counted X and Y as
informative (2) — wrongly including X, which `paired()` drops entirely since
subtracting through its NaN gives NaN. New code counts only Y (1), the single
concept that is both present everywhere and actually informative; Z is
correctly excluded (present, but a real zero difference) while still counting
toward the test's own n of 2 the way the pre-registration's `informative_n`
concept always intended.

**Re-ran all four pilot smoke tests after the fix** (primary, `--compliant-only`,
`--hybrid-compliant`, `concept_vector`): every `informative_n` and `detectable_dz`
printed identically to before the fix. Expected and reassuring, not a null
result about the fix — pilot's SAE missingness is entirely concept-level
(`k=0` at latent selection, never per-cell), so the bug had no live case to
correct there; it was always specifically a `--compliant-only`-on-held-out risk,
which is exactly why it's being closed before that run rather than after.

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
  `CLAUDE.md`'s file table implies the repo root. Harmonize the paths or the doc.
- SAE width: `CLAUDE.md` records 16k as "the only variant Neuronpedia indexed",
  but the SAELens registry lists 65k / 262k / 1m at `l0_medium`. Unverified.
- ~~Concept split size disagrees between documents.~~ **Resolved 2026-08-30:
  10 pilot / 40 held out, all 50 concepts used.** `PLAN.md` §5 and `CLAUDE.md`
  amended from 10/30. Reason: n=10 reserve detects only dz > 0.9 and cannot serve
  as a confirmatory set, while costing 15% of power on the headline contrast
  (Holm-corrected detectable dz 0.58 at n=40 vs 0.67 at n=30). Split verified
  stratified by grammatical number: pilot 3 mass / 7 plural, held out 14 / 26.
