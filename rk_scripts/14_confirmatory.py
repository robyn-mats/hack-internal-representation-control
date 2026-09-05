"""Confirmatory analysis: the 15-test family, on the HELD-OUT split.

Implements PREREGISTRATION.md exactly where it is explicit, and flags the one
place it is not (see FAMILY_RULE below).

Fixed by the pre-registration and by the Stage 2 commits:

  unit of analysis   the concept -- mean over carriers, then over phrasings
                     within a cell, then paired across concepts. Cells, not
                     phrasings, are compared.
  readout            SAE latent activation, `latent_sum` (primary).
                     `concept_vector` is run too, as the registered secondary.
  layer              40                      (commit b54ee96)
  pooling            token_mean              (commit 61ab4f2)
  trials             ALL, not compliant-only. Compliant-only is the registered
                     robustness check (--compliant-only).
  statistics         paired t, alpha = .05 two-sided; dz = mean(d)/sd(d);
                     BCa bootstrap CI over concepts, 10,000 resamples;
                     Holm across the 15-member family.
  gate               Q0 (A > T1 > T7): adjacent steps ordered by sign, and the
                     A-vs-T7 span significant at alpha two-sided (amendment
                     2026-08-31, fixed before held-out was tested). If it fails,
                     Q1-Q10 are NOT run and the result is a failed replication
                     of the base effect. This script enforces that and exits.

Outside the correction family: Q0 (gate), Q9 (sanity check), Q10 (descriptive).
That is what makes the family 15 rather than 18.

    python3 rk_scripts/14_confirmatory.py --run-id heldout1
    python3 rk_scripts/14_confirmatory.py --run-id heldout1 --compliant-only
    python3 rk_scripts/14_confirmatory.py --run-id heldout1 --readout concept_vector
"""

from irc import env  # noqa: F401

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from irc.paths import RUNS

REPO_ROOT = Path(__file__).resolve().parents[1]

# letter -> cell_id, from irc/conditions.csv. Note the phrasing ids Q1/Q2/Q3
# (nonce_imp_none) are a DIFFERENT namespace from the contrast names Q0-Q10.
CELL = {
    "A": "focus_imp_none",       "B": "focus_decl_none",
    "C": "focus_imp_syn",        "D": "focus_imp_both",
    "E": "focus_decl_syn",       "F": "focus_decl_both",
    "G": "mental_imp_syn",       "H": "output_imp_syn",
    "I": "away_imp_none",        "J": "away_imp_morph",
    "K": "relevance_decl_none",  "L": "relevance_decl_morph",
    "M": "relevance_decl_syn",   "N": "incong_imp_none",
    "P": "incong_imp_syn",       "Q": "nonce_imp_none",
    "R": "nonce_imp_syn",        "S": "nonce_decl_none",
    "T1": "base_bare_none",      "T2": "base_bare_syn",
    "T3": "base_filler_none",    "T4": "base_filler_adjacent",
    "T5": "base_filler_detached", "T6": "floor_control",
    "T7": "base_absent",
}

# How a family member with several comparisons yields ONE p-value for Holm.
#
# NOT SPECIFIED by the pre-registration, which lists e.g. Q3c as "A vs I; B vs K"
# and says Q7/Q8 "absorb their additions rather than becoming new family
# members" -- fixing the family at 15 without saying how to combine within a
# member. Two rules are used, chosen to match the structure of each member, and
# every sub-comparison is reported individually regardless:
#
#   "mean"     the member's comparisons are directionally parallel (the same
#              factor contrast at two levels of another factor), so the family
#              test is a single paired contrast on the AVERAGE of the per-concept
#              differences. Q3c, Q5b, Q5c, Q5f.
#   "omnibus"  the member's cells form a gradient or a set of heterogeneous
#              questions that must not be averaged, so the family test is a
#              repeated-measures one-way test across those cells (Friedman,
#              which needs no sphericity assumption). Q5, Q5d, Q7, Q8.
#
# THIS NEEDS SIGN-OFF before the numbers are treated as confirmatory.
FAMILY_RULE = "documented in the module docstring; see FAMILY_RULE comment"

N_BOOT = 10_000
ALPHA = 0.05


# ----------------------------------------------------------------- helpers ---

def per_concept(df: pd.DataFrame) -> pd.DataFrame:
    """One value per (concept, cell): mean over carriers, then over phrasings."""
    by_phrasing = (df.groupby(["concept", "cell_id", "phrasing_id"],
                              observed=True).value.mean().reset_index())
    return (by_phrasing.groupby(["concept", "cell_id"], observed=True)
                       .value.mean().reset_index())


def per_concept_phrasing(df: pd.DataFrame) -> pd.DataFrame:
    """One value per (concept, phrasing): mean over carriers. For Q9's L1."""
    return (df.groupby(["concept", "phrasing_id"], observed=True)
              .value.mean().reset_index())


def wide(pc: pd.DataFrame) -> pd.DataFrame:
    return pc.pivot_table(index="concept", columns="cell_id", values="value")


def _boot_stat(b: np.ndarray, kind: str) -> np.ndarray:
    """Statistic over the last axis of a (resamples, n) matrix, vectorized."""
    if kind == "mean":
        return b.mean(axis=-1)
    if kind == "dz":
        sd = b.std(axis=-1, ddof=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(sd > 0, b.mean(axis=-1) / np.where(sd > 0, sd, 1.0),
                            np.nan)
    raise ValueError(f"unknown statistic {kind!r}")


def bca_ci(x: np.ndarray, kind: str = "mean", alpha: float = ALPHA,
           n_boot: int | None = None, rng=None) -> tuple[float, float]:
    """BCa bootstrap CI over the concept axis, for "mean" or "dz".

    Takes a statistic NAME rather than a callable so resampling is one
    vectorized numpy operation instead of a Python loop -- the family runs ~30
    paired comparisons, each needing a bootstrap and a jackknife, and the loop
    version cost minutes per contrast at the registered 10,000 resamples.

    `n_boot` defaults to the module-level N_BOOT at CALL time rather than at
    definition time, so --n-boot actually takes effect.
    """
    n_boot = n_boot or N_BOOT
    rng = rng or np.random.default_rng(0)
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 3:
        return (np.nan, np.nan)
    theta = float(_boot_stat(x[None, :], kind)[0])
    if not np.isfinite(theta):
        return (np.nan, np.nan)

    boots = _boot_stat(x[rng.integers(0, n, size=(n_boot, n))], kind)
    boots = boots[np.isfinite(boots)]
    if len(boots) < 100:
        return (np.nan, np.nan)

    prop = float(np.mean(boots < theta))
    prop = min(max(prop, 1e-6), 1 - 1e-6)
    z0 = stats.norm.ppf(prop)

    jmat = np.stack([np.delete(x, i) for i in range(n)])
    jack = _boot_stat(jmat, kind)
    jack = jack[np.isfinite(jack)]
    if len(jack) < 3:
        return (np.nan, np.nan)
    dev = jack.mean() - jack
    denom = 6.0 * (float(np.sum(dev ** 2)) ** 1.5)
    a = float(np.sum(dev ** 3) / denom) if denom > 0 else 0.0

    out = []
    for q in (alpha / 2, 1 - alpha / 2):
        z = stats.norm.ppf(q)
        adj = z0 + (z0 + z) / max(1 - a * (z0 + z), 1e-12)
        out.append(float(np.percentile(boots, 100 * stats.norm.cdf(adj))))
    return (out[0], out[1])


def informative(w: pd.DataFrame, cells: list[str]) -> tuple[int, float]:
    """(concepts with any signal across `cells`, detectable dz at that n).

    A concept whose readout is exactly 0.0 in EVERY cell of a contrast has a
    structurally zero difference: it consumes an n while carrying no
    information. With the SAE readout at 92% zeros those concepts dominate some
    contrasts -- on the pilot, Q5e (`irrelevant` vs `not relevant`) had 2 of 9
    informative -- so a non-significant result there is absence of evidence, not
    evidence of absence.

    detectable dz rescales the registered figure (0.60 at n=40, Holm across 15)
    as 1/sqrt(n). It is an adjustment of a stated number, not a fresh power
    calculation.
    """
    have = [c for c in cells if c in w.columns]
    if not have:
        return 0, float("nan")
    sub = w[have].dropna(how="all")
    n = int((sub.fillna(0).abs().sum(axis=1) > 0).sum())
    return n, (round(0.60 * (40 / n) ** 0.5, 3) if n else float("nan"))


def paired(w: pd.DataFrame, hi: str, lo: str, rng=None) -> dict:
    """Paired comparison of two cells across concepts."""
    if hi not in w.columns or lo not in w.columns:
        return {"n": 0, "note": f"missing cell ({hi} or {lo})"}
    d = (w[hi] - w[lo]).dropna()
    if len(d) < 3:
        return {"n": len(d), "note": "fewer than 3 paired concepts"}
    x = d.to_numpy(float)
    t, p = stats.ttest_1samp(x, 0.0)
    sd = x.std(ddof=1)
    lo_ci, hi_ci = bca_ci(x, "mean", rng=rng)
    dz_lo, dz_hi = bca_ci(x, "dz", rng=rng)
    inf_n, inf_dz = informative(w, [hi, lo])
    return {"n": len(x), "mean_diff": float(x.mean()), "sd_diff": float(sd),
            "dz": float(x.mean() / sd) if sd > 0 else np.nan,
            "t": float(t), "p": float(p),
            "ci_lo": lo_ci, "ci_hi": hi_ci,
            "dz_ci_lo": dz_lo, "dz_ci_hi": dz_hi,
            "informative_n": inf_n, "detectable_dz": inf_dz}


def paired_mean_of(w: pd.DataFrame, pairs: list[tuple[str, str]],
                   rng=None) -> dict:
    """One paired test on the AVERAGE of several per-concept differences."""
    cols = []
    for hi, lo in pairs:
        if hi in w.columns and lo in w.columns:
            cols.append(w[hi] - w[lo])
    if not cols:
        return {"n": 0, "note": "no usable pairs"}
    d = pd.concat(cols, axis=1).mean(axis=1).dropna()
    if len(d) < 3:
        return {"n": len(d), "note": "fewer than 3 paired concepts"}
    x = d.to_numpy(float)
    t, p = stats.ttest_1samp(x, 0.0)
    sd = x.std(ddof=1)
    lo_ci, hi_ci = bca_ci(x, "mean", rng=rng)
    inf_n, inf_dz = informative(w, [c for pr in pairs for c in pr])
    return {"n": len(x), "mean_diff": float(x.mean()), "sd_diff": float(sd),
            "dz": float(x.mean() / sd) if sd > 0 else np.nan,
            "t": float(t), "p": float(p), "ci_lo": lo_ci, "ci_hi": hi_ci,
            "informative_n": inf_n, "detectable_dz": inf_dz}


def paired_interaction(w: pd.DataFrame, a: str, b: str, c: str, d: str,
                       rng=None) -> dict:
    """Paired test of the 2x2 interaction ((a - b) - (c - d)) across concepts.

    For a frame x negation cell block: a/b are the two negation levels at one
    frame, c/d the same two levels at the other. A non-zero contrast means the
    effect of negation differs by frame, which is the interaction the design
    asks about -- distinct from the omnibus across the four cells, which is
    driven mostly by main effects.
    """
    need = [a, b, c, d]
    if any(x not in w.columns for x in need):
        missing = [x for x in need if x not in w.columns]
        return {"n": 0, "note": f"missing cells {missing}"}
    d_ = ((w[a] - w[b]) - (w[c] - w[d])).dropna()
    if len(d_) < 3:
        return {"n": len(d_), "note": "fewer than 3 paired concepts"}
    x = d_.to_numpy(float)
    t, pv = stats.ttest_1samp(x, 0.0)
    sd = x.std(ddof=1)
    lo, hi = bca_ci(x, "mean", rng=rng)
    return {"n": len(x), "mean_diff": float(x.mean()), "sd_diff": float(sd),
            "dz": float(x.mean() / sd) if sd > 0 else np.nan,
            "t": float(t), "p": float(pv), "ci_lo": lo, "ci_hi": hi}


def omnibus(w: pd.DataFrame, cells: list[str]) -> dict:
    """Friedman across cells, concepts as blocks. No sphericity assumption."""
    have = [c for c in cells if c in w.columns]
    sub = w[have].dropna()
    if len(have) < 3 or len(sub) < 3:
        return {"n": len(sub), "note": f"need 3+ cells and concepts, "
                                      f"have {len(have)}/{len(sub)}"}
    chi2, p = stats.friedmanchisquare(*[sub[c].to_numpy(float) for c in have])
    inf_n, inf_dz = informative(w, have)
    return {"n": len(sub), "k_cells": len(have), "chi2": float(chi2),
            "p": float(p), "cells": have,
            "informative_n": inf_n, "detectable_dz": inf_dz}


def holm(pvals: dict[str, float], alpha: float = ALPHA) -> dict[str, dict]:
    """Holm-Bonferroni step-down. Returns per-key adjusted p and reject flag."""
    items = [(k, v) for k, v in pvals.items() if v is not None and np.isfinite(v)]
    items.sort(key=lambda kv: kv[1])
    m = len(items)
    out, running = {}, 0.0
    for i, (k, p) in enumerate(items):
        adj = min(1.0, max(running, (m - i) * p))
        running = adj
        out[k] = {"p": p, "p_holm": adj, "reject": adj < alpha}
    for k, v in pvals.items():
        if k not in out:
            out[k] = {"p": v, "p_holm": None, "reject": None}
    return out


MIN_INFORMATIVE = 15   # see PREREGISTRATION.md amendment 2026-09-05


def verdict(r: dict, alpha: float = ALPHA) -> str:
    """What a result is licensed to claim, given how much signal it had.

    Registered 2026-09-05, before unblinding: a non-significant contrast may be
    reported as a null only if it had enough informative concepts to detect a
    large effect. Below that it is UNDERPOWERED -- absence of evidence.
    """
    p, n = r.get("p"), r.get("informative_n")
    if p is None or n is None:
        return ""
    if p < alpha:
        return "significant"
    return "null" if n >= MIN_INFORMATIVE else "UNDERPOWERED (not a null)"


def fmt(r: dict) -> str:
    if r.get("note"):
        return f"n={r.get('n', 0)}  {r['note']}"
    inf = ""
    if r.get("informative_n") is not None:
        inf = (f"  informative={r['informative_n']}"
               f" (detectable dz {r['detectable_dz']})  [{verdict(r)}]")
    if "chi2" in r:
        return (f"n={r['n']}  k={r['k_cells']}  chi2={r['chi2']:.2f}  "
                f"p={r['p']:.4g}{inf}")
    ci = (f"[{r['ci_lo']:.4g}, {r['ci_hi']:.4g}]"
          if np.isfinite(r.get("ci_lo", np.nan)) else "[--]")
    return (f"n={r['n']}  diff={r['mean_diff']:.4g}  dz={r['dz']:.3f}  "
            f"95%CI {ci}  p={r['p']:.4g}{inf}")


# --------------------------------------------------------------- contrasts ---

def run_contrasts(w: pd.DataFrame, wp: pd.DataFrame, rng) -> dict:
    """All contrasts. Keys are family members; `_sub` holds sub-comparisons."""
    C = CELL
    out: dict[str, dict] = {}

    # ---- Q0: the GATE. Not a family member. ----
    #
    # Evaluation rule fixed by amendment 2026-08-31, before held-out was tested:
    # the two adjacent steps must be ORDERED (sign only) and the overall A-vs-T7
    # span must be significant at alpha, two-sided, uncorrected (the gate sits
    # outside the Holm family).
    #
    # Why sign-only on the steps: the pre-registration says "ordering holds" for
    # Q0 while saying "significantly" where it means significance (Q9: "L1
    # significantly below T6"; Q2: "no significant difference"). T1 is a graded
    # middle term that was never powered as a separate test -- the power
    # calculation is for the primary contrast -- so requiring significance on
    # each step would gate the whole analysis on a comparison the design does
    # not claim to resolve.
    a_t1 = paired(w, C["A"], C["T1"], rng)
    t1_t7 = paired(w, C["T1"], C["T7"], rng)
    a_t7 = paired(w, C["A"], C["T7"], rng)
    ordered = (a_t1.get("mean_diff", -1) > 0 and t1_t7.get("mean_diff", -1) > 0)
    span_ok = (a_t7.get("mean_diff", -1) > 0 and a_t7.get("p", 1) < ALPHA)
    out["Q0"] = {"kind": "gate", "A_vs_T1": a_t1, "T1_vs_T7": t1_t7,
                 "A_vs_T7": a_t7, "ordered_signs": bool(ordered),
                 "span_significant": bool(span_ok),
                 "holds": bool(ordered and span_ok)}

    # ---- family members ----
    out["Q1"] = {"kind": "pair", "test": paired(w, C["T1"], C["L"], rng),
                 "note": "T1 - L; prediction: L below T1 (positive difference)",
                 "_sub": {"L_vs_T7": paired(w, C["L"], C["T7"], rng)}}
    out["Q2"] = {"kind": "pair", "test": paired(w, C["G"], C["T1"], rng),
                 "note": "prediction: no significant difference"}
    out["Q3"] = {"kind": "pair", "test": paired(w, C["I"], C["G"], rng),
                 "note": "PRIMARY; direction not predicted"}
    out["Q3b"] = {"kind": "pair", "test": paired(w, C["C"], C["A"], rng)}
    out["Q3c"] = {"kind": "mean",
                  "test": paired_mean_of(w, [(C["A"], C["I"]),
                                             (C["B"], C["K"])], rng),
                  "_sub": {"A_vs_I": paired(w, C["A"], C["I"], rng),
                           "B_vs_K": paired(w, C["B"], C["K"], rng)}}
    out["Q4"] = {"kind": "pair", "test": paired(w, C["K"], C["M"], rng)}
    out["Q5"] = {"kind": "omnibus",
                 "test": omnibus(w, [C["G"], C["I"], C["K"], C["M"]]),
                 "note": "frame x negation, away side",
                 "_sub": {
                     "interaction_(I-G)-(K-M)": paired_interaction(
                         w, C["I"], C["G"], C["K"], C["M"], rng),
                     "I_vs_K_imperative_vs_decl_none": paired(
                         w, C["I"], C["K"], rng),
                     "G_vs_M_imperative_vs_decl_syn": paired(
                         w, C["G"], C["M"], rng)}}
    out["Q5b"] = {"kind": "mean",
                  "test": paired_mean_of(w, [(C["I"], C["J"]),
                                             (C["K"], C["L"])], rng),
                  "_sub": {"I_vs_J": paired(w, C["I"], C["J"], rng),
                           "K_vs_L": paired(w, C["K"], C["L"], rng)}}
    out["Q5c"] = {"kind": "mean",
                  "test": paired_mean_of(w, [(C["C"], C["D"]),
                                             (C["E"], C["F"])], rng),
                  "_sub": {"C_vs_D": paired(w, C["C"], C["D"], rng),
                           "E_vs_F": paired(w, C["E"], C["F"], rng)}}
    out["Q5d"] = {"kind": "omnibus",
                  "test": omnibus(w, [C["A"], C["B"], C["C"], C["E"]]),
                  "note": "frame x negation, focus side",
                  "_sub": {
                      "interaction_(A-C)-(B-E)": paired_interaction(
                          w, C["A"], C["C"], C["B"], C["E"], rng),
                      "A_vs_B_imperative_vs_decl_none": paired(
                          w, C["A"], C["B"], rng),
                      "C_vs_E_imperative_vs_decl_syn": paired(
                          w, C["C"], C["E"], rng)}}
    out["Q5e"] = {"kind": "pair", "test": paired(w, C["L"], C["M"], rng)}
    out["Q5f"] = {"kind": "mean",
                  "test": paired_mean_of(w, [(C["A"], C["D"]),
                                             (C["B"], C["F"])], rng),
                  "_sub": {"A_vs_D": paired(w, C["A"], C["D"], rng),
                           "B_vs_F": paired(w, C["B"], C["F"], rng)}}
    out["Q6"] = {"kind": "pair", "test": paired(w, C["H"], C["G"], rng)}
    out["Q7"] = {"kind": "omnibus",
                 "test": omnibus(w, [C["I"], C["N"], C["P"], C["Q"], C["R"],
                                     C["S"], C["T1"]]),
                 "note": "coherence gradient, T1 as floor",
                 "_sub": {"Q_vs_T1": paired(w, C["Q"], C["T1"], rng),
                          "N_vs_T1": paired(w, C["N"], C["T1"], rng)}}
    out["Q8"] = {"kind": "omnibus",
                 "test": omnibus(w, [C["T1"], C["T2"], C["T3"], C["T4"],
                                     C["T5"]]),
                 "_sub": {"T1_vs_T2": paired(w, C["T1"], C["T2"], rng),
                          "T1_vs_T3": paired(w, C["T1"], C["T3"], rng),
                          "T3_vs_T4": paired(w, C["T3"], C["T4"], rng),
                          "T4_vs_T5": paired(w, C["T4"], C["T5"], rng)}}

    # ---- Q9: sanity check, outside the family. L1 is a PHRASING. ----
    q9 = {"n": 0, "note": "phrasing L1 or cell T6 absent"}
    if "L1" in set(wp.phrasing_id):
        l1 = wp[wp.phrasing_id == "L1"].set_index("concept").value
        if C["T6"] in w.columns:
            d = (l1 - w[C["T6"]]).dropna()
            if len(d) >= 3:
                x = d.to_numpy(float)
                t, p = stats.ttest_1samp(x, 0.0)
                sd = x.std(ddof=1)
                lo, hi = bca_ci(x, "mean", rng=rng)
                q9 = {"n": len(x), "mean_diff": float(x.mean()),
                      "sd_diff": float(sd),
                      "dz": float(x.mean() / sd) if sd > 0 else np.nan,
                      "t": float(t), "p": float(p), "ci_lo": lo, "ci_hi": hi}
    out["Q9"] = {"kind": "sanity", "test": q9,
                 "note": "L1 - T6; prediction: L1 significantly BELOW T6"}

    # ---- Q10: descriptive, outside the family ----
    var = (w.std(ddof=1) / w.mean().abs().replace(0, np.nan)).rename("cv")
    out["Q10"] = {"kind": "descriptive",
                  "between_concept_sd": w.std(ddof=1).to_dict(),
                  "cv": var.to_dict()}
    return out


FAMILY = ["Q1", "Q2", "Q3", "Q3b", "Q3c", "Q4", "Q5", "Q5b", "Q5c", "Q5d",
          "Q5e", "Q5f", "Q6", "Q7", "Q8"]


def main() -> None:
    global N_BOOT
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-id", default="heldout1")
    ap.add_argument("--pass", dest="pass_", default="generated")
    ap.add_argument("--split", default="held_out")
    ap.add_argument("--readout", default="latent_sum",
                    choices=["latent_sum", "latent_max", "concept_vector"])
    ap.add_argument("--variant", default="word_tokens")
    ap.add_argument("--latents-version", default="v2")
    ap.add_argument("--compliant-only", action="store_true",
                    help="registered robustness check; primary is ALL trials")
    ap.add_argument("--layer", type=int, default=None,
                    help="default: from stage2_values.json")
    ap.add_argument("--pooling", default=None,
                    help="default: from stage2_values.json")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ignore-gate", action="store_true",
                    help="run the family even if Q0 fails. EXPLORATORY ONLY -- "
                         "for developing the code on pilot data, where n=9 "
                         "cannot power the gate. Never for confirmatory use.")
    ap.add_argument("--n-boot", type=int, default=N_BOOT,
                    help="bootstrap resamples; lower only for smoke tests")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    # --- Stage 2 values come from the committed record, not from flags ---
    s2p = REPO_ROOT / "stage2_values.json"
    if not s2p.exists():
        raise SystemExit("stage2_values.json missing -- run "
                         "rk_scripts/11_stage2_choose.py --write-json first. "
                         "The analysis layer must be a committed value.")
    s2 = json.loads(s2p.read_text())
    layer = args.layer if args.layer is not None else int(s2["analysis_layer"])
    pooling = args.pooling or s2["pooling_rule"]
    print(f"Stage 2: layer={layer} pooling={pooling} "
          f"(from stage2_values.json, commit {s2.get('git_commit', '?')[:8]})")
    if args.layer is not None or args.pooling is not None:
        print("  WARNING: layer/pooling overridden on the command line. "
              "Results are EXPLORATORY, not confirmatory.")

    res = RUNS / args.run_id / args.pass_ / "results"
    path = (res / f"readout_concept_vector_{args.variant}.parquet"
            if args.readout == "concept_vector"
            else res / f"readout_sae_{args.latents_version}.parquet")
    if not path.exists():
        raise SystemExit(f"missing {path} -- run rk_scripts/09_measure.py")
    df = pd.read_parquet(path)

    readout_name = ("concept_vector" if args.readout == "concept_vector"
                    else args.readout)
    df = df[(df.split == args.split) & (df.layer == layer)
            & (df.pooling == pooling) & (df.readout == readout_name)]
    if args.compliant_only:
        df = df[df.exact_match]
    print(f"{path.name}: {len(df):,} rows, {df.concept.nunique()} concepts, "
          f"{df.cell_id.nunique()} cells"
          + ("  [COMPLIANT-ONLY robustness check]" if args.compliant_only
             else "  [all trials, as registered]"))
    if df.empty:
        raise SystemExit("no rows after filtering -- check layer/pooling/readout")

    if args.n_boot != N_BOOT:
        print(f"  NOTE: n_boot={args.n_boot} (registered value is {N_BOOT}); "
              f"CIs are approximate -- smoke test only.")
        N_BOOT = args.n_boot
    rng = np.random.default_rng(args.seed)
    pc = per_concept(df)
    w = wide(pc)
    wp = per_concept_phrasing(df)
    print(f"paired table: {w.shape[0]} concepts x {w.shape[1]} cells")

    results = run_contrasts(w, wp, rng)

    # ---- the gate, first and hard ----
    q0 = results["Q0"]
    print("\n" + "=" * 74)
    print("Q0 GATE  (A > T1 > T7)")
    print("=" * 74)
    print("  rule: adjacent steps ordered by SIGN; A-vs-T7 span significant")
    print(f"  A  - T1 : {fmt(q0['A_vs_T1'])}")
    print(f"  T1 - T7 : {fmt(q0['T1_vs_T7'])}")
    print(f"  A  - T7 : {fmt(q0['A_vs_T7'])}   <- the significance test")
    print(f"  ordered signs      : {'YES' if q0['ordered_signs'] else 'NO'}")
    print(f"  span significant   : {'YES' if q0['span_significant'] else 'NO'}")
    print(f"  GATE HOLDS         : {'YES' if q0['holds'] else 'NO'}")
    if not q0["holds"]:
        print("\n" + "!" * 74)
        print("GATE FAILED. Per PREREGISTRATION.md the confirmatory contrasts")
        print("Q1-Q10 are NOT run, and the result is reported as a failed")
        print("replication of the base effect on this model with this readout.")
        print("No further analysis is attempted on this data.")
        print("!" * 74)
        if args.out:
            args.out.write_text(json.dumps(
                {"gate": q0, "gate_failed": True, "layer": layer,
                 "pooling": pooling, "readout": readout_name}, indent=1,
                default=str))
        if not args.ignore_gate:
            raise SystemExit(1)
        print("\n--ignore-gate: continuing anyway. THESE NUMBERS ARE "
              "EXPLORATORY,\nnot confirmatory, and must not be reported as "
              "confirming anything.")

    # ---- family, Holm-corrected ----
    pv = {k: results[k]["test"].get("p") for k in FAMILY}
    adj = holm(pv)
    print("\n" + "=" * 74)
    print(f"CONFIRMATORY FAMILY ({len(FAMILY)} tests, Holm, alpha={ALPHA})")
    print("=" * 74)
    for k in FAMILY:
        r = results[k]
        a = adj[k]
        star = "*" if a.get("reject") else " "
        ph = f"{a['p_holm']:.4g}" if a.get("p_holm") is not None else "--"
        print(f"\n{star} {k:<5} [{r['kind']}] p_holm={ph}"
              + (f"   {r['note']}" if r.get("note") else ""))
        print(f"      {fmt(r['test'])}")
        for name, sub in (r.get("_sub") or {}).items():
            print(f"        - {name:<24} {fmt(sub)}")

    # ---- outside the family ----
    print("\n" + "=" * 74)
    print("OUTSIDE THE CORRECTION FAMILY")
    print("=" * 74)
    print(f"  Q9 (sanity, must pass): {fmt(results['Q9']['test'])}")
    print(f"      {results['Q9']['note']}")
    q9 = results["Q9"]["test"]
    if "p" in q9:
        ok = q9["mean_diff"] < 0 and q9["p"] < ALPHA
        print(f"      L1 significantly below T6: "
              f"{'YES' if ok else 'NO -- sanity check FAILED'}")
    print("\n  Q10 (descriptive) between-concept sd by cell:")
    for cell, sd in sorted(results["Q10"]["between_concept_sd"].items(),
                           key=lambda kv: -(kv[1] or 0))[:8]:
        print(f"      {cell:<22} {sd:.4g}")

    if args.out:
        args.out.write_text(json.dumps(
            {"layer": layer, "pooling": pooling, "readout": readout_name,
             "split": args.split, "compliant_only": args.compliant_only,
             "n_concepts": int(w.shape[0]), "family": FAMILY,
             "holm": adj, "results": results,
             "family_combination_rule": FAMILY_RULE}, indent=1, default=str))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
