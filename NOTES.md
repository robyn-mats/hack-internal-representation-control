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

**2. Within-sentence z-scoring is blind to cluster contamination.** Standardising
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

Standardising within *concept* (across sentences) removes attractors, but buries
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

The old behaviour remains the default so upstream's pipeline is unaffected.
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
"the prompt's odd request". That is the behaviour families N/P/Q/R/S exist to
probe, and this fork records it as a deviation-rate outcome rather than
discarding the trial.

Caveat: n=20, one carrier-length regime, and no nonce conditions were sampled.
Not a substitute for the per-cell rates the run will produce.

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

**Generalisable lesson:** `.strip()` in a correctness check discards exactly the
tokens whose presence changes the measurement. Compare token ids, not stripped
strings, wherever the token count feeds the readout.

## Open items

- `carrier_similarity.csv` and `stimuli.csv` are not yet generated.
- `PREREGISTRATION.md` is committed but its date and commit-hash header are
  placeholders until the run begins.
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
