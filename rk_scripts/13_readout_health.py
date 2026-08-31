"""Readout health tables: where the SAE readout works, and where it is silent.

Combines the positive control (rk_scripts/12_latent_positive_control.py) with
the pilot readout, and prints the tables needed to judge whether an exact-0.0
cell is a non-measurement or a finding.

Verdicts, per (concept, layer):

  NO_LATENTS      k=0. No latent survived selection, so no instrument exists.
                  09_measure.py emits no row (missing, not zero).
  SELECTION_DEAD  latents exist but are silent even on the concept's own tokens
                  in the prompts they were selected from. The 0.0 in the task is
                  an artifact and must not be read as "concept not activated".
  CONTEXT_NULL    latents fire in the selection context but not in the task.
                  The 0.0 is a real measurement: the concept is not activated
                  while the model copies an unrelated carrier sentence.
  OK              fires in both.

No GPU, no model.

    python3 rk_scripts/13_readout_health.py
"""

from irc import env  # noqa: F401

import argparse
from pathlib import Path

import pandas as pd

from irc.constants import SAE_LAYERS
from irc.paths import RUNS

REPO_ROOT = Path(__file__).resolve().parents[1]

CELL_A = "focus_imp_none"
CELL_T1 = "base_bare_none"
CELL_T7 = "base_absent"

MARK = {"OK": ".", "CONTEXT_NULL": "n", "SELECTION_DEAD": "D",
        "NO_LATENTS": "-", "CONTROL_ONLY": "?"}


def recompute_verdict(ctl: pd.DataFrame) -> pd.DataFrame:
    """Relabel verdicts, distinguishing "no task data" from "context null".

    The pilot run contains only the 10 pilot concepts, so `task_max` is missing
    for the 40 held-out ones. Script 12 scores a missing task_max as not firing,
    which reads as CONTEXT_NULL when the truth is simply that the task side has
    not been measured. That matters because CONTEXT_NULL is a *finding* and
    CONTROL_ONLY is an absence of data.

    The usable-n counts are unaffected: a layer is usable for a concept when the
    selection is non-empty and fires in the control context, which needs no task
    data at all.
    """
    ctl = ctl.copy()

    def v(r):
        if r.k == 0:
            return "NO_LATENTS"
        if not (r.control_max > 0):
            return "SELECTION_DEAD"
        if pd.isna(r.task_max):
            return "CONTROL_ONLY"
        return "OK" if r.task_max > 0 else "CONTEXT_NULL"

    ctl["verdict"] = ctl.apply(v, axis=1)
    return ctl


def grid(df: pd.DataFrame, value: str, fmt: str = "{:.0f}") -> str:
    """concept x layer grid of one column."""
    w = df.pivot_table(index="concept", columns="layer", values=value,
                       aggfunc="first")
    w = w.reindex(columns=[l for l in SAE_LAYERS if l in w.columns])
    head = "  " + f"{'concept':<16}" + "".join(f"{'L'+str(l):>12}" for l in w.columns)
    lines = [head]
    for c, r in w.iterrows():
        cells = "".join(
            f"{'--':>12}" if pd.isna(v) else f"{fmt.format(v):>12}" for v in r
        )
        lines.append(f"  {c:<16}{cells}")
    return "\n".join(lines)


def verdict_grid(df: pd.DataFrame) -> str:
    w = df.pivot_table(index="concept", columns="layer", values="verdict",
                       aggfunc="first")
    w = w.reindex(columns=[l for l in SAE_LAYERS if l in w.columns])
    head = "  " + f"{'concept':<16}" + "".join(f"{'L'+str(l):>6}" for l in w.columns)
    lines = [head]
    for c, r in w.iterrows():
        cells = "".join(f"{MARK.get(v, '?'):>6}" for v in r)
        bad = sum(1 for v in r if v in ("SELECTION_DEAD", "NO_LATENTS"))
        lines.append(f"  {c:<16}{cells}" + ("   <-- problem" if bad else ""))
    lines.append(f"\n  key: {'  '.join(f'{v}={k}' for k, v in MARK.items())}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-id", default="pilot2")
    ap.add_argument("--control", type=Path,
                    default=REPO_ROOT / "latent_positive_control.parquet")
    ap.add_argument("--latents-version", default="v2")
    args = ap.parse_args()

    if not args.control.exists():
        raise SystemExit(f"no {args.control} -- run "
                         f"rk_scripts/12_latent_positive_control.py first")
    ctl = recompute_verdict(pd.read_parquet(args.control))

    print("=" * 78)
    print("1. VERDICT COUNTS BY LAYER")
    print("=" * 78)
    for label, d in (("all 50 concepts", ctl),
                     ("pilot (10)", ctl[ctl.split == "pilot"]),
                     ("held out (40)", ctl[ctl.split == "held_out"])):
        if not len(d):
            continue
        t = pd.crosstab(d.layer, d.verdict)
        cols = ["OK", "CONTEXT_NULL", "CONTROL_ONLY", "SELECTION_DEAD",
                "NO_LATENTS"]
        for col in cols:
            if col not in t.columns:
                t[col] = 0
        t = t[cols]
        # usable = an instrument exists and demonstrably fires somewhere.
        t["usable"] = t.OK + t.CONTEXT_NULL + t.CONTROL_ONLY
        print(f"\n{label}:")
        print(t.to_string())

    print("\n" + "=" * 78)
    print("2. PILOT: VERDICT PER CONCEPT x LAYER")
    print("=" * 78)
    print(verdict_grid(ctl[ctl.split == "pilot"].sort_values("concept")))

    print("\n" + "=" * 78)
    print("3. PILOT: CONTROL vs TASK ACTIVATION")
    print("   control_max = max latent_sum on the concept's own tokens in the")
    print("   four selection prompts. task_max = max in the pilot A cell.")
    print("=" * 78)
    p = ctl[ctl.split == "pilot"].sort_values("concept")
    print("\ncontrol_max (selection context):")
    print(grid(p, "control_max", "{:.1f}"))
    print("\ntask_max (A cell, the strongest instruction):")
    print(grid(p, "task_max", "{:.1f}"))
    print("\nk (latents selected):")
    print(grid(p, "k", "{:.0f}"))

    print("\n" + "=" * 78)
    print("4. Q0 ORDERING PER LAYER, AND WHAT IT RESTS ON")
    print("=" * 78)
    pq = (RUNS / args.run_id / "generated" / "results"
          / f"readout_sae_{args.latents_version}.parquet")
    t = pd.read_parquet(pq)
    t = t[(t.split == "pilot") & (t.pooling == "token_mean")
          & (t.readout == "latent_sum")]
    keys = ["layer", "concept", "cell_id"]
    per = t.groupby(keys, observed=True).value.mean().reset_index()
    print(f"\n  {'layer':>5} {'mean A':>10} {'mean T1':>10} {'mean T7':>10} "
          f"{'A>T1>T7':>9} {'nonzero concepts':>17}")
    for layer, d in per.groupby("layer", observed=True):
        w = d.pivot_table(index="concept", columns="cell_id", values="value")
        for c in (CELL_A, CELL_T1, CELL_T7):
            if c not in w:
                w[c] = float("nan")
        nz = int((w[[CELL_A, CELL_T1, CELL_T7]].fillna(0).abs().sum(axis=1) > 0).sum())
        a, t1, t7 = w[CELL_A].mean(), w[CELL_T1].mean(), w[CELL_T7].mean()
        ok = "YES" if a > t1 > t7 else "no"
        print(f"  {layer:>5} {a:>10.1f} {t1:>10.1f} {t7:>10.1f} {ok:>9} "
              f"{nz:>10} of {len(w)}")

    print("\n" + "=" * 78)
    print("5. HELD-OUT USABLE n PER LAYER (instrument metadata only)")
    print("   Recomputed with SELECTION_DEAD excluded, since a silent latent")
    print("   measures nothing even though k>0.")
    print("=" * 78)
    h = ctl[ctl.split == "held_out"]
    if len(h):
        rows = []
        for layer, d in h.groupby("layer", observed=True):
            k_pos = int((d.k > 0).sum())
            usable = int(d.verdict.isin(
                ["OK", "CONTEXT_NULL", "CONTROL_ONLY"]).sum())
            rows.append({"layer": layer, "k>0 (old rule)": k_pos,
                         "fires (new rule)": usable,
                         "detectable_dz_old": round(0.60 * (40 / max(k_pos, 1)) ** .5, 3),
                         "detectable_dz_new": round(0.60 * (40 / max(usable, 1)) ** .5, 3)})
        print("\n" + pd.DataFrame(rows).set_index("layer").to_string())


if __name__ == "__main__":
    main()
