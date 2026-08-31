"""Derive the Stage 2 values from the PILOT split, per PREREGISTRATION.md.

ONE value is chosen here: the **analysis layer** -- the layer maximizing
A-vs-T7 separation, restricted to the published Gemma Scope 2 layers
(16 / 31 / 40 / 53).

The pooling rule is already settled and is NOT chosen here. Of the three
pre-registered candidates, `top-k by activation` was dropped before any readout
was computed and `plausible` (rule 3) was found degenerate on the pilot
(commit 61ab4f2: weights span 1e-23 to 3e-06, so the weighted mean selects
far-tail softmax numerics rather than weighting anything). `token_mean` is the
only non-degenerate candidate and is fixed.

Both dropped rules are still printed, marked ineligible: `plausible` because
seeing its numbers next to `token_mean` is the check that the degeneracy finding
still holds on the full pilot readout, and `max` because a large gap between it
and `token_mean` is worth knowing about. Neither can win.

Separation is the paired effect size dz = mean(delta) / sd(delta) over concepts,
which follows the registered unit of analysis: average over carriers, then over
phrasings within a cell, then pair across concepts. The pilot has 10 concepts,
so these dz values are estimated on n=10 and are not themselves evidence about
any hypothesis -- they only rank instruments.

Runs on ALL trials, not compliant-only (PREREGISTRATION.md: conditioning on
compliance conditions on an outcome and opens a collider path). The
compliant-only grid is printed as the registered robustness check.

Reads the pilot parquets written by 09_measure.py. No GPU, no model.

    python3 rk_scripts/11_stage2_choose.py --run-id pilot2
    python3 rk_scripts/11_stage2_choose.py --run-id pilot2 --write-json
"""

from irc import env  # noqa: F401

import argparse
import json
import subprocess
import time
from pathlib import Path

import pandas as pd

from irc.constants import SAE_LAYERS
from irc.paths import RUNS

REPO_ROOT = Path(__file__).resolve().parents[1]

# The cells the choice is made on. A is the strongest toward-directed
# instruction in the design; T7 is the concept-free baseline, whose prompt never
# names a concept and is therefore shared across concepts (see NOTES.md
# 2026-08-31 on the T7 expansion).
CELL_A = "focus_imp_none"    # A1, A2  -- "concentrate on dust"
CELL_T1 = "base_bare_none"   # T1      -- bare mention, the Q0 middle term
CELL_T7 = "base_absent"      # T7      -- no third line at all

PRIMARY_READOUT = "latent_sum"     # PREREGISTRATION.md: SAE latent activation
# Fixed by the 2026-08-31 amendment, not chosen here. See the module docstring.
POOLING_RULE = "token_mean"
INELIGIBLE_POOLINGS = ("max", "plausible")


def per_concept(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse to one value per (readout, layer, pooling, concept, cell).

    The registered order: mean over carriers within a phrasing, then mean over
    phrasings within a cell. Two steps rather than one grand mean, because
    phrasings have unequal carrier counts only if a trial is missing, and an
    unweighted phrasing mean is what "cells, not phrasings, are the unit"
    means.
    """
    keys = ["readout", "layer", "pooling", "concept", "cell_id"]
    by_phrasing = (df.groupby(keys + ["phrasing_id"], observed=True)
                     .value.mean().reset_index())
    return by_phrasing.groupby(keys, observed=True).value.mean().reset_index()


def separation(pc: pd.DataFrame, cell_hi: str, cell_lo: str) -> pd.DataFrame:
    """Paired dz for cell_hi - cell_lo, per (readout, layer, pooling)."""
    keys = ["readout", "layer", "pooling"]
    hi = pc[pc.cell_id == cell_hi].set_index(keys + ["concept"]).value
    lo = pc[pc.cell_id == cell_lo].set_index(keys + ["concept"]).value
    delta = (hi - lo).dropna().rename("delta").reset_index()
    if delta.empty:
        return delta
    g = delta.groupby(keys, observed=True).delta
    out = pd.DataFrame({"n": g.size(), "mean_delta": g.mean(),
                        "sd_delta": g.std(ddof=1)}).reset_index()
    out["dz"] = out.mean_delta / out.sd_delta
    return out.sort_values("dz", ascending=False)


def q0_by_layer(pc: pd.DataFrame, readout: str, pooling: str) -> pd.DataFrame:
    """Pilot Q0 ordering (A > T1 > T7) per layer, and hence layer eligibility.

    Q0 is a HARD GATE in PREREGISTRATION.md: if the ordering does not reproduce
    on held-out, Q1-Q10 are not run at all. A layer the pilot shows failing it
    therefore cannot support the confirmatory analysis, so passing Q0 on the
    pilot is a precondition for eligibility -- applied before the dz criterion
    rather than as a tiebreak after it.
    """
    sub = pc[(pc.readout == readout) & (pc.pooling == pooling)]
    rows = []
    for layer, d in sub.groupby("layer", observed=True):
        w = d.pivot_table(index="concept", columns="cell_id", values="value")
        for c in (CELL_A, CELL_T1, CELL_T7):
            if c not in w:
                w[c] = float("nan")
        a, t1, t7 = w[CELL_A].mean(), w[CELL_T1].mean(), w[CELL_T7].mean()
        nz = int((w[[CELL_A, CELL_T1, CELL_T7]].fillna(0).abs().sum(axis=1) > 0).sum())
        rows.append({"layer": int(layer), "mean_A": a, "mean_T1": t1,
                     "mean_T7": t7, "informative": nz, "n": len(w),
                     "q0_passes": bool(a > t1 > t7)})
    return pd.DataFrame(rows).set_index("layer")


def heldout_coverage(latents_version: str = "v2") -> pd.DataFrame:
    """Usable held-out concepts per layer, from the latent selection alone.

    This is INSTRUMENT METADATA, not held-out data: `select_latents` builds its
    selection from the concept words and their template prompts, and never sees
    a carrier, a condition or a generation. Reading it therefore does not
    inspect held-out results and does not compromise the split -- "how many
    latents does concept X have at layer L" is a different question from "how
    active is concept X in condition C".

    k=0 means no latent survived selection, so that concept has no readout at
    that layer and drops out of every contrast there.
    """
    import json

    from irc.paths import ARTIFACTS

    st = pd.read_csv(REPO_ROOT / "stimuli.csv")
    split = {c.capitalize(): g.split.iloc[0] for c, g in st.groupby("concept")}
    rows = []
    for f in sorted((ARTIFACTS / f"latents_{latents_version}").glob("*.json")):
        j = json.loads(f.read_text())
        rows.append({"concept": f.stem, "split": split.get(f.stem, "?"),
                     **{l: len(j["layers"][str(l)]) for l in SAE_LAYERS}})
    d = pd.DataFrame(rows)
    held = d[d.split == "held_out"]
    n_total = len(held)
    out = []
    for l in SAE_LAYERS:
        usable = int((held[l] > 0).sum())
        # The pre-registration states detectable dz = 0.60 at n=40 for the
        # 15-test Holm family. Rescaled as 1/sqrt(n) -- an adjustment of the
        # stated figure, not a fresh power calculation.
        out.append({"layer": l, "heldout_usable": usable,
                    "heldout_missing": n_total - usable,
                    "detectable_dz": round(0.60 * (40 / usable) ** 0.5, 3)})
    return pd.DataFrame(out).set_index("layer")


def coverage(pc: pd.DataFrame, readout: str, pooling: str) -> pd.DataFrame:
    """Concepts with a usable readout at each layer, for the paired cells.

    The SAE latent selection is ragged: k ranges 0..5 over (concept, layer), and
    where k=0 the readout does not exist, so 09_measure.py emits no row. Layers
    therefore cover different concept sets, and an unrestricted per-layer dz
    would be computed on different subsets -- a layer could rank higher partly
    by having dropped its hardest concepts.
    """
    sub = pc[(pc.readout == readout) & (pc.pooling == pooling)
             & (pc.cell_id.isin([CELL_A, CELL_T7]))]
    g = (sub.groupby(["layer", "cell_id"], observed=True)
            .concept.nunique().unstack(fill_value=0))
    paired = (sub.pivot_table(index=["layer", "concept"], columns="cell_id",
                              values="value", aggfunc="mean")
                 .dropna().reset_index()
                 .groupby("layer").concept.nunique().rename("paired"))
    return g.join(paired, how="outer").fillna(0).astype(int)


def common_concepts(pc: pd.DataFrame, readout: str, pooling: str) -> set[str]:
    """Concepts paired at EVERY layer, so layers can be ranked on one set."""
    sub = pc[(pc.readout == readout) & (pc.pooling == pooling)
             & (pc.cell_id.isin([CELL_A, CELL_T7]))]
    per_layer = []
    for layer, d in sub.groupby("layer", observed=True):
        w = d.pivot_table(index="concept", columns="cell_id", values="value",
                          aggfunc="mean").dropna()
        per_layer.append(set(w.index))
    return set.intersection(*per_layer) if per_layer else set()


def show_grid(sep: pd.DataFrame, title: str) -> None:
    print(f"\n{title}")
    if sep.empty:
        print("  (no paired data -- is the T7 baseline present per concept?)")
        return
    print(f"  {'readout':<15} {'layer':>5} {'pooling':<11} {'n':>3} "
          f"{'mean_d':>10} {'sd':>10} {'dz':>7}  eligible")
    for _, r in sep.iterrows():
        elig = "yes" if r.pooling == POOLING_RULE else "NO (dropped)"
        print(f"  {r.readout:<15} {int(r.layer):>5} {r.pooling:<11} "
              f"{int(r.n):>3} {r.mean_delta:>10.4f} {r.sd_delta:>10.4f} "
              f"{r.dz:>7.3f}  {elig}")


def ordering_check(pc: pd.DataFrame, layer: int, pooling: str,
                   readout: str) -> None:
    """The Q0 ordering A > T1 > T7, on the pilot. Diagnostic, not confirmatory.

    Q0 is a confirmatory contrast on the held-out split. Running it on the
    pilot is what the pilot is for: if the ordering fails here, the instrument
    is not measuring what the design assumes and that is worth knowing before
    the held-out set is touched.
    """
    sub = pc[(pc.layer == layer) & (pc.pooling == pooling)
             & (pc.readout == readout)]
    print(f"\nPilot Q0 ordering at layer {layer}, {pooling}, {readout} "
          f"(diagnostic -- confirmatory Q0 runs on held-out):")
    means = {}
    for name, cell in (("A ", CELL_A), ("T1", CELL_T1), ("T7", CELL_T7)):
        v = sub[sub.cell_id == cell].value
        means[name] = v.mean()
        print(f"  {name} {cell:<18} n={len(v):>3}  mean={v.mean():>10.4f}  "
              f"sd={v.std(ddof=1):>9.4f}")
    ok = means["A "] > means["T1"] > means["T7"]
    print(f"  ordering A > T1 > T7: {'HOLDS' if ok else 'DOES NOT HOLD'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-id", default="pilot2")
    ap.add_argument("--pass", dest="pass_", default="generated")
    ap.add_argument("--variant", default="word_tokens")
    ap.add_argument("--latents-version", default="v2")
    ap.add_argument("--split", default="pilot",
                    help="must stay 'pilot' -- Stage 2 is derived on the pilot")
    ap.add_argument("--write-json", action="store_true",
                    help="write stage2_values.json (the committed record)")
    args = ap.parse_args()

    res = RUNS / args.run_id / args.pass_ / "results"
    paths = {
        "sae": res / f"readout_sae_{args.latents_version}.parquet",
        "concept_vector": res / f"readout_concept_vector_{args.variant}.parquet",
    }
    frames = []
    for name, p in paths.items():
        if p.exists():
            frames.append(pd.read_parquet(p))
            print(f"read {p.name}: {len(frames[-1]):,} rows")
        else:
            print(f"MISSING {p} -- run 09_measure.py first")
    if not frames:
        raise SystemExit("no readout parquets found")
    df = pd.concat(frames, ignore_index=True)

    df = df[df.split == args.split]
    print(f"\nsplit={args.split}: {len(df):,} rows, "
          f"{df.concept.nunique()} concepts, {df.cell_id.nunique()} cells")
    for cell in (CELL_A, CELL_T1, CELL_T7):
        d = df[df.cell_id == cell]
        print(f"  {cell:<18} {len(d):>7,} rows  "
              f"{d.concept.nunique():>2} concepts  "
              f"{d.prompt_group.nunique():>4} prompt_groups")
    if df[df.cell_id == CELL_T7].concept.nunique() < df.concept.nunique():
        raise SystemExit(
            f"T7 covers only {df[df.cell_id == CELL_T7].concept.nunique()} of "
            f"{df.concept.nunique()} concepts. The A-vs-T7 separation is not "
            f"computable. 09_measure.py must expand concept-free prompt_groups "
            f"over every concept sharing the prompt -- see NOTES.md 2026-08-31.")

    pc = per_concept(df)
    sep = separation(pc, CELL_A, CELL_T7)
    show_grid(sep, "A-vs-T7 separation, ALL trials (registered basis for the choice):")

    compliant = separation(per_concept(df[df.exact_match]), CELL_A, CELL_T7)
    show_grid(compliant, "Same, COMPLIANT trials only (registered robustness check):")

    cov = coverage(pc, PRIMARY_READOUT, POOLING_RULE)
    print(f"\nConcept coverage at {PRIMARY_READOUT}/{POOLING_RULE} "
          f"(k=0 in the latent selection means no readout exists):")
    print(cov.to_string())

    # Ranked on EACH LAYER'S OWN usable concepts -- the literal registered rule.
    #
    # An earlier amendment (same day) restricted this to concepts paired at
    # every layer, intending to stop a layer ranking high by dropping its
    # hardest concepts. It does the opposite. The common set excludes a concept
    # that ANY layer cannot measure, so a layer which measures it and reads
    # exactly 0.0 gets to shed it -- inflating that layer's dz -- while the
    # layer that merely lacks an instrument for it gains nothing. On the pilot
    # every layer's dz rose under the restriction except layer 40's, which
    # could not rise because the concept was already absent there, and the
    # argmax flipped. Superseded; both tables are printed.
    #
    # The correct accounting: measurable-but-silent is DATA (delta = 0, kept),
    # no-instrument is MISSING (dropped).
    sep_own = separation(pc, CELL_A, CELL_T7)
    show_grid(sep_own[sep_own.pooling == POOLING_RULE],
              "A-vs-T7 separation on EACH LAYER'S OWN concepts "
              "(the registered rule, basis for the choice):")

    common = common_concepts(pc, PRIMARY_READOUT, POOLING_RULE)
    dropped = sorted(set(pc.concept.unique()) - common)
    sep_common = separation(pc[pc.concept.isin(common)], CELL_A, CELL_T7)
    show_grid(sep_common[(sep_common.pooling == POOLING_RULE)
                         & (sep_common.readout == PRIMARY_READOUT)],
              f"Same on the {len(common)} common concepts (SUPERSEDED rule, "
              f"reported for transparency; drops {dropped}):")

    q0 = q0_by_layer(pc, PRIMARY_READOUT, POOLING_RULE)
    print("\nPilot Q0 ordering per layer -- eligibility precondition "
          "(Q0 is a hard gate):")
    print(q0.to_string())

    hc = heldout_coverage(args.latents_version)
    print("\nHeld-out coverage per layer (instrument metadata -- from the latent "
          "selection,\nnot from held-out results; used only as a binary "
          "usable/not count):")
    print(hc.to_string())

    cand = sep_own[(sep_own.readout == PRIMARY_READOUT)
                   & (sep_own.pooling == POOLING_RULE)]
    eligible = [int(l) for l in cand.layer if q0.loc[int(l), "q0_passes"]]
    excluded = [int(l) for l in cand.layer if not q0.loc[int(l), "q0_passes"]]
    if excluded:
        print(f"\n  layers excluded for failing the pilot Q0 gate: {excluded}")
    cand = cand[cand.layer.isin(eligible)].sort_values("dz", ascending=False)
    if cand.empty:
        raise SystemExit(
            "no layer passes the pilot Q0 ordering. The base effect does not "
            "reproduce at any layer on this readout -- report as a failed "
            "replication rather than choosing a layer.")

    best = cand.iloc[0]
    if len(cand) > 1:
        top = cand[cand.dz >= float(cand.iloc[0].dz) - 0.10]
        if len(top) > 1:
            ranked = top.assign(
                usable=[int(hc.loc[int(r.layer), "heldout_usable"])
                        for _, r in top.iterrows()]
            ).sort_values(["usable", "dz"], ascending=False)
            if int(ranked.iloc[0].layer) != int(best.layer):
                print(f"\n  TIEBREAK among Q0-eligible layers "
                      f"{sorted(int(l) for l in top.layer)}: preferring layer "
                      f"{int(ranked.iloc[0].layer)} on held-out coverage.")
            best = ranked.iloc[0]
    layer = int(best.layer)

    print(f"\n{'=' * 68}\nSTAGE 2 VALUE (from {args.run_id}, split={args.split}, "
          f"readout={PRIMARY_READOUT})\n{'=' * 68}")
    print(f"  analysis layer : {layer}")
    print(f"  pooling rule   : {POOLING_RULE}  (fixed by amendment, not chosen here)")
    print(f"  dz             : {best.dz:.3f}  "
          f"(mean delta {best.mean_delta:.4f}, sd {best.sd_delta:.4f}, n={int(best.n)})")
    print(f"  Q0 on pilot    : PASSES  "
          f"(A {q0.loc[layer, 'mean_A']:.1f} > T1 {q0.loc[layer, 'mean_T1']:.1f} "
          f"> T7 {q0.loc[layer, 'mean_T7']:.1f})")
    print(f"  informative    : {int(q0.loc[layer, 'informative'])} of "
          f"{int(q0.loc[layer, 'n'])} pilot concepts")
    print(f"  held-out n     : {int(hc.loc[layer, 'heldout_usable'])} of 40 "
          f"({int(hc.loc[layer, 'heldout_missing'])} have k=0 here)")
    print(f"  detectable dz  : {hc.loc[layer, 'detectable_dz']} "
          f"(0.60 at n=40, rescaled)")
    print("\n  eligible layer ranking (own concepts, Q0-passing only):")
    for _, r in cand.iterrows():
        print(f"    layer {int(r.layer):>2}  dz {r.dz:>7.3f}  "
              f"n={int(r.n)}")

    ordering_check(pc, layer, POOLING_RULE, PRIMARY_READOUT)

    if args.write_json:
        commit = subprocess.run(["git", "rev-parse", "HEAD"],
                                capture_output=True, text=True).stdout.strip()
        out = {
            "analysis_layer": layer,
            "pooling_rule": POOLING_RULE,
            "pooling_rule_source": "fixed by the 2026-08-31 amendment (commit "
                                   "61ab4f2), not derived here",
            "primary_readout": PRIMARY_READOUT,
            "criterion": f"paired dz, {CELL_A} minus {CELL_T7}, all trials",
            "dz": float(best.dz),
            "mean_delta": float(best.mean_delta),
            "sd_delta": float(best.sd_delta),
            "n_concepts": int(best.n),
            "ranking_rule": "each layer's own usable concepts (literal "
                            "registered rule); the same-day common-set "
                            "restriction was superseded",
            "q0_eligible_layers": eligible,
            "q0_excluded_layers": excluded,
            "concepts_dropped_from_common_set": dropped,
            "coverage_by_layer": {str(k): v for k, v in
                                  cov.to_dict(orient="index").items()},
            "heldout_coverage": {str(k): v for k, v in
                                 hc.to_dict(orient="index").items()},
            "heldout_usable": int(hc.loc[layer, "heldout_usable"]),
            "detectable_dz": float(hc.loc[layer, "detectable_dz"]),
            "derived_from": {"run_id": args.run_id, "pass": args.pass_,
                             "split": args.split},
            "ineligible_poolings": list(INELIGIBLE_POOLINGS),
            "sae_layers_considered": list(SAE_LAYERS),
            "full_grid": sep.to_dict(orient="records"),
            "git_commit": commit,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        path = REPO_ROOT / "stage2_values.json"
        path.write_text(json.dumps(out, indent=1))
        print(f"\nwrote {path}")
        print("Commit this BEFORE the held-out readouts are committed, so the "
              "ordering is checkable in git history.")


if __name__ == "__main__":
    main()
