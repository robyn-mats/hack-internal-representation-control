"""Deviation and leak rates per cell -- first-class outcomes, not filters.

Upstream excludes non-exact completions from measurement. This fork keeps them
and reports the rates, because excluding them discards ~2/3 of trials and, per
upstream's own note, may discard exactly the trials where the model engaged with
the concept hardest.

Reported per cell, and per (cell, split):

  deviation rate   the completion is not an exact match to the target carrier
  leak rate        the concept appears in the completion. Two tiers, both scored
                   against all 50 concepts, word boundaries, case-insensitive:
                     strict (primary)  inflectional forms + concept emoji
                     loose  (secondary) strict + pruned WordNet derivational
  self-leak        the leaked concept is the trial's OWN target concept
  other-leak       some other concept leaked (a carrier-contamination signal)

Leak is scored on the **completion only**, never the prompt: the prompt contains
the concept by construction in every condition but T7.

Also tests the pre-registered directional prediction that deviation is highest
in imperative families that ask for an action on the concept (N, P, Q, R),
lowest in the declaratives (B, K, L, M) and bare baselines (T1-T5), with the
mental imperative G in between. Tested at the registered unit of analysis: a
per-concept rate within each group, then paired across concepts.

No GPU, no model, no activations -- reads generations.jsonl and stimuli.csv.

    python3 rk_scripts/15_compliance.py --run-id heldout1
    python3 rk_scripts/15_compliance.py --run-id pilot2 --pass teacher_forced
"""

from irc import env  # noqa: F401

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from irc.paths import RUNS

REPO_ROOT = Path(__file__).resolve().parents[1]

# Registered grouping for the deviation prediction. Letters are phrasing
# prefixes; the cells they map to are in irc/conditions.csv.
GROUPS = {
    "action_imperative": ["incong_imp_none", "incong_imp_syn",
                          "nonce_imp_none", "nonce_imp_syn"],   # N, P, Q, R
    "mental_imperative": ["mental_imp_syn"],                     # G
    "declarative": ["focus_decl_none", "relevance_decl_none",
                    "relevance_decl_morph", "relevance_decl_syn"],  # B,K,L,M
    "bare_baseline": ["base_bare_none", "base_bare_syn", "base_filler_none",
                      "base_filler_adjacent", "base_filler_detached"],  # T1-T5
}
PREDICTED_ORDER = ["action_imperative", "mental_imperative",
                   "declarative", "bare_baseline"]


def leak_patterns(path: Path) -> tuple[dict, dict]:
    """(strict, loose) -> {concept: compiled pattern}.

    Words match on word boundaries; emoji match literally, because `\\b` never
    matches around a non-word character. Longer forms first so the alternation
    prefers the most specific.
    """
    strict, loose = {}, {}
    for row in csv.DictReader(path.open()):
        forms = [f for f in (row["forms"] or "").split("|") if f]
        emoji = [e for e in (row.get("forms_emoji") or "").split("|") if e]
        derived = [d for d in (row.get("forms_derived") or "").split("|")
                   if d and d != "nan"]

        def build(words, emo):
            parts = []
            if words:
                parts.append(r"\b(?:%s)\b" % "|".join(
                    re.escape(w) for w in sorted(words, key=len, reverse=True)))
            if emo:
                parts.append("|".join(re.escape(e) for e in emo))
            return re.compile("|".join(parts), re.IGNORECASE) if parts else None

        s = build(forms, emoji)
        l = build(forms + derived, emoji)
        if s:
            strict[row["concept"]] = s
        if l:
            loose[row["concept"]] = l
    return strict, loose


def load(run_dir: Path, stimuli: pd.DataFrame) -> pd.DataFrame:
    """One row per trial, with factor coding joined from stimuli.csv."""
    recs = []
    with (run_dir / "generations.jsonl").open() as fh:
        for line in fh:
            r = json.loads(line)
            recs.append({"prompt_group": r["prompt_group"],
                         "concept": r.get("concept"),
                         "completion": r.get("completion") or "",
                         "target": r.get("target") or "",
                         "exact_match": bool(r.get("exact_match"))})
    df = pd.DataFrame(recs).drop_duplicates("prompt_group", keep="last")
    # For T7 the prompt is shared across concepts, so stimuli has many rows per
    # prompt_group; the record's own `concept` is the authoritative one.
    cols = ["prompt_group", "cell_id", "phrasing_id", "split", "direction",
            "frame_type", "negation", "carrier_order"]
    st = stimuli[cols].drop_duplicates("prompt_group")
    return df.merge(st, on="prompt_group", how="left")


def score_leaks(df: pd.DataFrame, strict: dict, loose: dict) -> pd.DataFrame:
    """Add leak columns, scored on the completion only."""
    out = {"leak_strict": [], "leak_loose": [], "self_leak": [],
           "other_leak": [], "n_concepts_leaked": []}
    for _, r in df.iterrows():
        txt = r.completion or ""
        hits_s = [c for c, p in strict.items() if p.search(txt)]
        hits_l = [c for c, p in loose.items() if p.search(txt)]
        own = (r.concept or "").lower()
        out["leak_strict"].append(bool(hits_s))
        out["leak_loose"].append(bool(hits_l))
        out["self_leak"].append(own in [h.lower() for h in hits_s])
        out["other_leak"].append(any(h.lower() != own for h in hits_s))
        out["n_concepts_leaked"].append(len(hits_s))
    for k, v in out.items():
        df[k] = v
    return df


def rate_table(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    g = df.groupby(by, observed=True)
    t = pd.DataFrame({
        "n": g.size(),
        "deviation": 1.0 - g.exact_match.mean(),
        "leak_strict": g.leak_strict.mean(),
        "leak_loose": g.leak_loose.mean(),
        "self_leak": g.self_leak.mean(),
        "other_leak": g.other_leak.mean(),
    })
    return t.sort_values("deviation", ascending=False)


def group_of(cell: str) -> str | None:
    for name, cells in GROUPS.items():
        if cell in cells:
            return name
    return None


def test_prediction(df: pd.DataFrame) -> None:
    """Per-concept deviation rate by group, then paired across concepts.

    The registered unit of analysis is the concept, so a rate is computed within
    each (concept, group) and the groups are compared pairwise across concepts.
    Trial-level tests would treat 7 carriers x several phrasings as independent
    and badly understate the standard error.
    """
    d = df.copy()
    d["group"] = d.cell_id.map(group_of)
    d = d[d.group.notna() & d.concept.notna()]
    if d.empty:
        print("  (no cells in the registered groups)")
        return

    per = (d.groupby(["concept", "group"], observed=True)
             .apply(lambda g: 1.0 - g.exact_match.mean(), include_groups=False)
             .rename("dev").reset_index())
    w = per.pivot_table(index="concept", columns="group", values="dev")

    print("\n  mean per-concept deviation rate by group "
          f"(n={w.shape[0]} concepts):")
    for gname in PREDICTED_ORDER:
        if gname in w.columns:
            v = w[gname].dropna()
            print(f"    {gname:<20} {v.mean():>7.2%}  "
                  f"(sd {v.std(ddof=1):.2%}, n={len(v)})")
        else:
            print(f"    {gname:<20} absent from this run")

    present = [g for g in PREDICTED_ORDER if g in w.columns]
    print("\n  observed ordering (highest deviation first): "
          + " > ".join(w[present].mean().sort_values(ascending=False).index))
    print("  predicted ordering:                           "
          + " > ".join(PREDICTED_ORDER))
    ok = list(w[present].mean().sort_values(ascending=False).index) == present
    print(f"  ordering as predicted: {'YES' if ok else 'NO'}")

    print("\n  adjacent paired comparisons (higher - lower, predicted > 0):")
    for a, b in zip(present, present[1:]):
        dd = (w[a] - w[b]).dropna()
        if len(dd) < 3:
            print(f"    {a} - {b}: n={len(dd)}, too few")
            continue
        x = dd.to_numpy(float)
        t, p = stats.ttest_1samp(x, 0.0)
        sd = x.std(ddof=1)
        dz = x.mean() / sd if sd > 0 else np.nan
        print(f"    {a:<20} - {b:<18} diff={x.mean():>+7.2%}  "
              f"dz={dz:>6.3f}  p={p:.4g}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-id", default="heldout1")
    ap.add_argument("--pass", dest="pass_", default="generated")
    ap.add_argument("--stimuli", type=Path, default=REPO_ROOT / "stimuli.csv")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    run_dir = RUNS / args.run_id / args.pass_
    if not (run_dir / "generations.jsonl").exists():
        raise SystemExit(f"no generations.jsonl in {run_dir}")

    stimuli = pd.read_csv(args.stimuli)
    strict, loose = leak_patterns(REPO_ROOT / "irc" / "concepts.csv")
    print(f"leak patterns: {len(strict)} strict, {len(loose)} loose")

    df = score_leaks(load(run_dir, stimuli), strict, loose)
    print(f"{args.run_id}/{args.pass_}: {len(df):,} trials, "
          f"{df.concept.nunique()} concepts, {df.cell_id.nunique()} cells")

    print("\n" + "=" * 78)
    print("OVERALL")
    print("=" * 78)
    print(f"  deviation   {1 - df.exact_match.mean():.3%}  "
          f"({(~df.exact_match).sum():,} of {len(df):,})")
    print(f"  leak strict {df.leak_strict.mean():.3%}  "
          f"({df.leak_strict.sum():,})")
    print(f"  leak loose  {df.leak_loose.mean():.3%}")
    print(f"  self-leak   {df.self_leak.mean():.3%}   "
          f"other-leak {df.other_leak.mean():.3%}")

    print("\n" + "=" * 78)
    print("PER CELL (sorted by deviation)")
    print("=" * 78)
    t = rate_table(df, ["cell_id"])
    print(t.to_string(formatters={
        "deviation": "{:.2%}".format, "leak_strict": "{:.2%}".format,
        "leak_loose": "{:.2%}".format, "self_leak": "{:.2%}".format,
        "other_leak": "{:.2%}".format}))

    nonzero = t[t.deviation > 0]
    print(f"\n  cells with any deviation: {len(nonzero)} of {len(t)}")

    print("\n" + "=" * 78)
    print("REGISTERED DEVIATION PREDICTION")
    print("=" * 78)
    test_prediction(df)

    if args.out:
        t.to_csv(args.out)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
