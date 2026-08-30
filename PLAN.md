# Instruction-frame ablation — design

Why `X is irrelevant to this task` reduces a concept's internal activation while
`do not think about X` does nothing.

Operating conventions and hard rules live in `CLAUDE.md`. This file is the
rationale: what the question is, why each cell exists, and what the analysis
commits to in advance.

---

## 1 · The finding this starts from

Gurnee et al. (2026), *Verbalizable Representations Form a Global Workspace in
Language Models*, appendix A.10. Three Claude models, three task families,
five to eight phrasings per condition, read out with the Jacobian lens.

| Condition | Instruction | Effect on the concept |
|---|---|---|
| focus | "concentrate on X while you write" | Highest — but only *modestly* above mention |
| mention | X named, no instruction attached | Already high. Mere presence does most of the work. |
| don't-think | "whatever you do, do not think about X" | Roughly the mention rate. **No reduction.** |
| ignore | "X is irrelevant to this task" | **Well below mention, on every model.** |
| *absent* | *concept never in the prompt* | **Not measured — no such condition.** |

Two things follow.

**The affirmative question is settled.** Bare mention primes nearly as strongly
as an explicit focus instruction, so "does the verb *think about* do any work?"
is answered, negatively. That was the original plan for this project; it is
no longer the question.

**The interesting result is undecomposed.** `don't-think` and `ignore` request
the same outcome and one works. Anthropic describe the contrast — *"the model
can modulate the J-space downward when the instruction frames the concept as
task-irrelevant rather than as forbidden"* — but do not explain it, and the two
phrasings differ on at least four dimensions simultaneously.

### Why their comparison cannot attribute the effect

`do not think about X` is imperative, syntactically negated, and uses a
mental-state verb. `X is irrelevant to this task` is declarative, has no
mental verb, and — note — is **not negation-free**: `irrelevant` carries
morphological negation. The two sit two moves apart in frame × negation-type,
and the negation-free imperative cell was never run at all.

### Candidate explanations, all separable

1. **Negation.** Zhou et al. (2026) find models build *constructive*
   representations for negated concepts rather than suppressing them, which
   would predict exactly this failure.
2. **Relevance framing.** "Irrelevant to this task" may engage a task-relevance
   mechanism that prohibition does not touch.
3. **Mental-state verb.** Instructions about cognition may simply be less
   actionable than instructions about task structure.
4. **Prohibitive force.** "Whatever you do, do not…" is emphatic; emphasis may
   backfire — the ironic-rebound story.

---

## 2 · Terminology

`ignore < mention` is consistent with four different mechanisms and only one is
suppression. The concept-absent baseline (`T7`) is what distinguishes them, and
it is the control A.10 lacks. Use these terms and mean them:

| Term | Definition |
|---|---|
| **suppressed** | significantly below `T7` (concept absent) |
| **dampened** | significantly below `T1` (mention) but above `T7` |
| **not primed** | statistically indistinguishable from `T7` |
| **primed** | significantly above `T7` |

"Not primed" is the deflationary reading and it is invisible without `T7`:
`ignore` may not suppress anything, merely fail to prime. Kamp's Gemma result —
`don't-think` "barely different from no mention" — is **not primed**, not
"failed to suppress".

Likewise **rebound** in the strict (Wegner) sense means *above* mention, not
merely elevated. By that standard A.10's finding is failure-to-suppress, not
rebound.

---

## 3 · Design

Factors: **direction** (toward / away / neutral) × **frame** (imperative /
declarative) × **negation** (none / morphological / syntactic / both).

Negation is coded by whether the stem is a free morpheme in modern English:
`dis+regard`, `dis+count`, `ir+relevant`, `un+related`, `im+material` all
decompose. `ignore` does not — there is no `*gnore` — so despite its Latin
etymology it is coded negation-free.

`cell_id` is the unit of analysis. `phrasing_id` is a replicate within it.
Anthropic report per-phrasing precision spanning 14–56% within one condition,
so single-phrasing cells measure wording noise.

### The grid

**A · focus, imperative**

- `A1` concentrate on X while you write
- `A2` think about X while you write

**B · focus, declarative**

- `B1` X is/are relevant to this task
- `B2` X is/are related to this task
- `B3` X is/are the point here

**C · negated focus, imperative**

- `C1` do not ignore X — negates `I5`
- `C2` do not overlook X
- `C3` do not set X aside — negates `I1`
- `C4` do not skip past X — negates `I2`

**D · negated focus, imperative, double**

- `D1` do not disregard X — negates `J1`
- `D2` do not discount X — negates `J2`
- `D3` do not de-emphasize X — negates `J3`
- `D4` do not disattend X — negates `J4`

**E · negated focus, declarative**

- `E1` X is/are not beside the point here — negates `K1`
- `E2` X is/are not extraneous here — negates `K2`
- `E3` X does/do not belong to a different task — negates `K3`

**F · negated focus, declarative, double**

- `F1` X is/are not irrelevant to this task — negates `L1`
- `F2` X is/are not unrelated to what you are writing — negates `L2`
- `F3` X is/are not immaterial to this task — negates `L3`

**G · prohibitive, mental — **Anthropic `don't-think`, fails****

- `G1` whatever you do, do not think about X
- `G2` do not think about X
- `G3` do not consider X

**H · prohibitive, output-directed**

- `H1` do not mention X
- `H2` do not write X

**I · away, negation-free imperative — **the key cell****

- `I1` set X aside
- `I2` skip past X
- `I3` pass over X
- `I4` leave X out
- `I5` ignore X

**J · away imperative, morphological negation**

- `J1` disregard X
- `J2` discount X
- `J3` de-emphasize X
- `J4` disattend X

**K · relevance declarative, negation-free**

- `K1` X is/are beside the point here
- `K2` X is/are extraneous here
- `K3` X belongs/belong to a different task

**L · relevance declarative, morphological — **Anthropic `ignore`, works****

- `L1` X is/are irrelevant to this task
- `L2` X is/are unrelated to what you are writing
- `L3` X is/are immaterial to this task

**M · relevance declarative, syntactic**

- `M1` X is/are not relevant to this task — negates `B1`
- `M2` X does/do not matter here

**N · incongruent real verb**

- `N1` juggle X
- `N2` laminate X
- `N3` notarize X
- `N4` braise X
- `N5` centrifuge X

**P · incongruent, negated**

- `P1` do not juggle X — negates `N1`
- `P2` do not laminate X — negates `N2`
- `P3` do not notarize X — negates `N3`
- `P4` do not braise X — negates `N4`
- `P5` do not centrifuge X — negates `N5`

**Q · nonce verb**

- `Q1` glorf X
- `Q2` flarn X
- `Q3` vusk X

**R · nonce, negated**

- `R1` do not glorf X — negates `Q1`
- `R2` do not flarn X — negates `Q2`
- `R3` do not vusk X — negates `Q3`

**S · nonce, declarative**

- `S1` X is/are glorfy to this task
- `S2` X is/are flarny to this task
- `S3` X is/are vusky to this task

**T · baselines and controls**

- `T1` X
- `T2` not X — negates `T1`
- `T3` . . . . . X
- `T4` . . . . not X — negates `T3`
- `T5` not . . . . X — negates `T3`
- `T6` Y is/are irrelevant to this task
- `T7` *(no third line)*
### What this grid does that A.10 cannot

- **I isolates negation from framing.** One move from `G`, holding direction and
  frame fixed. If `set X aside` reduces the concept like `X is irrelevant` does,
  negation is the culprit; if it fails like `do not think about X`, framing is.
  `K` vs `M` asks the same from the declarative side.
- **C tests whether negation is symmetric.** A negated *focus* instruction should
  fail to elevate if negation is a general instruction-breaker. If damage is
  confined to away-commands, negation is not the mechanism.
- **N/P and Q/R/S give a coherence gradient.** `glorf` has no semantics to fail
  at; `juggle` has real semantics that do not apply. Does the instruction need to
  make sense, or only to be well-formed?
- **T1 vs T2 is the purest negation test** — negation with no instruction at all.
  **T4 vs T5 is a permutation control**: identical tokens, identical length,
  only arrangement differs, testing whether negation must bind locally.

### Stimulus construction

Prompt scaffold, carriers, and agreement handling are specified in `CLAUDE.md`.

**Carrier screening.** `X is irrelevant to this task` is *false* when the carrier
is about X — the model then receives a contradiction, not a suppression
instruction. All 50 of Anthropic's sentences are embedded against all 50
concepts; carriers are filtered on max cosine, then k selected matched on token
count. Both stages and the threshold are pre-registered. `max_similarity` and
`similarity_to_this_concept` are carried into `stimuli.csv` as covariates — even
among survivors there is a range, and it is a miniature of the relevance-truth
experiment.

This narrows the claim to *the effect of `irrelevant` when it is true*, which is
the right narrow claim, and leaves the crossed version as the follow-up.

---

## 4 · Measurement

**Primary readout: SAE latents.** Kamp found cosine on difference-in-means
concept vectors is the weakest of the available readouts (paired deltas ~0.015–0.025,
largely inside baseline noise) while SAE latents separate cleanly. Concept
vectors are reported alongside for comparability with Lindsey and Kamp, not
relied on.

**Generate as primary, teacher-force as robustness check.** Neither is clean:

| Method | Confound |
|---|---|
| Generate | Outputs differ across conditions → activation differences confounded with *what was written* |
| Teacher-force | Forced text is off-distribution **asymmetrically** → differences confounded with *surprisal* |

The second is the reason teacher-forcing cannot be primary: if the model deviates
on 60% of `G` trials and 5% of `L` trials, forcing pushes `G` further
off-distribution, and the between-condition difference partly measures surprise
at imposed text — an artifact correlated with the manipulated factor.

Record per-condition surprisal of the forced tokens. Flat surprisal means
teacher-forcing is near-neutral; surprisal tracking condition means it is not.

### Ruling out dilution

A reduced readout need not mean anything acted on the concept:

- **Norm competition.** Cosine is normalized, so added content lowers similarity
  even at constant absolute contribution.
- **Attention budget.** Softmax sums to one; instruction tokens take attention
  from X's tokens.
- **Feature-slot competition.** At L0 ≈ 60 roughly sixty latents fire by
  construction. X's feature can lose a competition for slots rather than being
  suppressed. This hazard is specific to a sparse readout.

Four checks, three of them re-reads of activations already collected:

1. **Read out a second, unrelated concept under every condition.** Dilution is
   indiscriminate; suppression is selective.
2. **Report raw projection alongside cosine.** Normalization is where norm
   competition bites.
3. **Check reduction against `n_prompt_tokens`.** Dilution scales with added content.
4. **Family N is the control.** Directionally neutral but adds instruction
   content. If N reduces the target as much as L does, dilution leads.

---

## 5 · Analysis

**Unit of analysis is the concept**, n=40 held out. Average over carriers and
over phrasings within a cell, then paired tests across concepts. Bootstrap CIs
over concepts. Holm correction across the contrast family.

**Pilot/held-out split.** 10 pilot concepts choose the analysis layer and the
pooling rule; the other 40 are untouched until the rule is committed. With ~62
layers, 25 cells and several pooling choices, post-hoc selection would find
significance whether or not an effect exists. The split is drawn once, stratified
by grammatical number (pilot 3 mass / 7 plural; held out 14 / 26).

All 50 concepts are used. An earlier draft held out 30 and left 10 unused as a
reserve; that was dropped because n=10 detects only dz > 0.9, so the reserve
cannot serve as a confirmatory replication set, while the ten concepts cost 15%
of the power on the headline contrast (Holm-corrected detectable dz 0.58 at
n=40 versus 0.67 at n=30). Carriers and phrasings, not concepts, are the slack
if compute binds: a carrier costs ~1% of contrast SE, a concept far more.

### Contrasts

| # | Contrast | Question |
|---|---|---|
| Q0 | A > T1 > T7 | Does the basic ordering reproduce? **Hard gate.** |
| Q1 | L vs T1 vs T7 | Does the `ignore` reduction replicate — suppressed, dampened, or not primed? |
| Q2 | G ≈ T1 | Does the `don't-think` null replicate? |
| Q3 | **I vs G** | **Negation, direction and frame held fixed. Headline.** |
| Q3b | C vs A | Is negation symmetric — does it break focus instructions too? |
| Q4 | K vs M | Negation, declarative held fixed. Converging evidence. |
| Q5 | G/I/K/M | Full frame × negation interaction on the away side. |
| Q5b | I vs J, K vs L | Does morphological negation behave like syntactic, or like none? |
| Q5c | C vs D, E vs F | Double negation: composed, or worse? |
| Q6 | H vs G | Output versus cognition. |
| Q7 | I vs N/P vs Q/R/S | Coherence gradient. |
| Q8 | T1/T2/T4/T5 | Negation with no instruction; locality of binding. |
| Q9 | L1 vs T6 | Floor check. Must pass or nothing else means anything. |
| Q10 | var within cell | Do phrasings within a cell agree? |

### Pre-registered predictions

Four hypotheses, four distinct signatures:

| If the driver is… | G | I | K | M | C |
|---|---|---|---|---|---|
| Negation breaks instructions | fails | reduces | reduces | fails | fails to elevate |
| Frame type (declarative wins) | fails | fails | reduces | reduces | elevates |
| Mental verbs unactionable | fails | reduces | reduces | reduces | elevates |
| Away-instructions specifically hard | fails | ? | reduces | fails | elevates |

`C` separates row 1 from row 4: if negation is a general instruction-breaker it
should damage a negated *focus* command too.

### Also pre-register

- Whether the activation analysis uses all trials or compliant-only (the other
  reported as robustness). Compliant-only conditions on an outcome — a collider.
- `juggle` may pattern apart from its cellmates given its idiomatic attentional
  sense ("juggling priorities"). Report N per-phrasing.
- `disattend` is rare outside psychology. If it patterns with N or Q rather than
  its J cellmates, that locates real-but-unfamiliar verbs on the frequency axis.
- Mass/plural tracks abstract/concrete almost exactly (17 vs 33). Concrete nouns
  tend to have cleaner SAE features. Declare as an exploratory split.

---

## 6 · Limitations

- **Different instrument from A.10.** SAE latents on Gemma versus J-lens on
  Claude, and a writing task versus copying. A disagreement is not decisive.
  A J-lens arm would close this — `github.com/anthropics/jacobian-lens` fits the
  lens on open-weight HF decoders.
- **No temporal dimension.** Measurement is concurrent with the instruction in
  context. Rebound *after* a suppression attempt ends (Mann et al.'s paradigm)
  is not tested.
- **Carriers screened for distance**, so the claim is about `irrelevant` when
  true.
- **Teacher-forced activations are counterfactual** — a trajectory the model
  might not have produced.
- **The effect may not exist at this scale.** Kamp's Gemma effect was much weaker
  than Anthropic's Claude effect, and his `don't-think` barely differed from no
  mention. Q0 is a hard gate for exactly this reason.

---

## 7 · References

- Gurnee, Sofroniew, Pearce et al. (2026). *Verbalizable Representations Form a
  Global Workspace in Language Models.* arXiv:2607.15495 — **appendix A.10**.
- Lindsey (2025). *Emergent Introspective Awareness in Large Language Models.*
  transformer-circuits.pub/2025/introspection
- Kamp (2026). *Intentional Control of Internal States in Gemma 3 27B.*
  LessWrong — the upstream of this fork.
- Zhou, Zhou, Jia & May (2026). *How Language Models Process Negation.* ICML.
  arXiv:2605.03052
- Ramnauth & Scassellati (2026). *The Attentional White Bear Effect in
  Transformer Language Models.* arXiv:2605.28639
- Mann, Saxena, Tandon et al. (2025). *Don't Think of the White Bear.*
  arXiv:2511.12381
- Macar, Yang, Wang et al. (2026). *Mechanisms of Introspective Awareness.*
  arXiv:2603.21396