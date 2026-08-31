"""Positive control for the SAE readout: do the selected latents fire at all?

The pilot readout is exactly 0.0 for several (concept, layer) cells in EVERY
condition, including A -- the strongest instruction to think about the concept.
That zero is ambiguous between two very different things:

  instrument insensitivity  the model represents the concept, but the five
                            latents we selected do not detect it in this task,
                            so 0.0 is a non-measurement and must not be read as
                            "concept not activated"
  a genuine null            the latents are valid detectors and the concept
                            really is not activated while the model copies an
                            unrelated carrier sentence

Nothing in the experiment distinguishes them, so this script measures the same
latents in the context they were SELECTED in: the four WORD_TEMPLATES_V1
prompts, at the concept's own token positions. That is the context where
`select_latents` scored them, so it is the strongest positive control available
-- if a latent does not fire there, it does not fire anywhere.

Reads the selection, runs the model, and writes a per-(concept, layer) table:

  k              latents selected (0..5; k=0 means no readout exists at all)
  control_max    max latent_sum over the concept's own tokens, 4 templates
  control_fires  control_max > 0
  task_max       max latent_sum seen in the pilot A cell (from the parquet)
  verdict        SELECTION_DEAD  control silent -> the zero is an artifact
                 CONTEXT_NULL    control fires, task silent -> the zero is data
                 OK              both fire
                 NO_LATENTS      k=0, no instrument

    python3 rk_scripts/12_latent_positive_control.py --run-id pilot2
"""

from irc import env  # noqa: F401

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from irc.constants import SAE_LAYERS
from irc.paths import ARTIFACTS, RUNS
from irc.words import WORD_TEMPLATES_V1

REPO_ROOT = Path(__file__).resolve().parents[1]


@torch.no_grad()
def control_activations(model, tokenizer, saes, sel: dict) -> list[dict]:
    """max latent_sum on the concept's own tokens, across the four templates."""
    from irc.pipeline import _sae_feats

    rows = []
    for i, (concept, per_layer) in enumerate(sorted(sel.items()), 1):
        best = {l: 0.0 for l in SAE_LAYERS}
        for t in WORD_TEMPLATES_V1:
            feats = _sae_feats(model, tokenizer, saes, list(SAE_LAYERS),
                               t.format(word=concept), word=concept)
            for l in SAE_LAYERS:
                idx = per_layer.get(l) or []
                if not idx:
                    continue
                # latent_sum per token, then the max over the word's tokens --
                # the same collapse the experiment uses, minus the pooling.
                v = float(feats[l][:, idx].float().sum(-1).max())
                best[l] = max(best[l], v)
        for l in SAE_LAYERS:
            rows.append({"concept": concept, "layer": l,
                         "k": len(per_layer.get(l) or []),
                         "control_max": best[l]})
        if i % 10 == 0:
            print(f"  [control] {i}/{len(sel)} concepts", flush=True)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-id", default="pilot2")
    ap.add_argument("--latents-version", default="v2")
    ap.add_argument("--cell", default="focus_imp_none",
                    help="task cell to compare against (default A)")
    ap.add_argument("--out", type=Path,
                    default=REPO_ROOT / "latent_positive_control.csv")
    args = ap.parse_args()

    # --- the selection, keyed lowercase to match stimuli.csv ---
    sel = {}
    for f in sorted((ARTIFACTS / f"latents_{args.latents_version}").glob("*.json")):
        j = json.loads(f.read_text())
        sel[f.stem.lower()] = {int(l): [e["latent"] for e in ents]
                               for l, ents in j["layers"].items()}
    print(f"{len(sel)} concepts in latents_{args.latents_version}")

    # --- the task side, from the already-written parquet (no GPU needed) ---
    pq = (RUNS / args.run_id / "generated" / "results"
          / f"readout_sae_{args.latents_version}.parquet")
    task = pd.read_parquet(pq)
    task = task[(task.readout == "latent_sum") & (task.pooling == "token_mean")
                & (task.cell_id == args.cell)]
    tmax = (task.groupby(["concept", "layer"], observed=True).value.max()
                .rename("task_max").reset_index())
    splits = task.drop_duplicates("concept").set_index("concept").split.to_dict()
    print(f"task side: {pq.name}, cell={args.cell}, "
          f"{task.concept.nunique()} concepts")

    # --- run the control ---
    print("loading model (~1-9 min)...", flush=True)
    from irc.model import load_model, load_tokenizer
    from irc.pipeline import load_saes

    tokenizer = load_tokenizer()
    model = load_model()
    saes = load_saes(list(SAE_LAYERS), "cuda")
    rows = control_activations(model, tokenizer, saes, sel)

    df = pd.DataFrame(rows).merge(tmax, on=["concept", "layer"], how="left")
    df["split"] = df.concept.map(splits).fillna("held_out")
    df["control_fires"] = df.control_max > 0
    df["task_fires"] = df.task_max.fillna(0) > 0

    def verdict(r):
        if r.k == 0:
            return "NO_LATENTS"
        if not r.control_fires:
            return "SELECTION_DEAD"
        if not r.task_fires:
            return "CONTEXT_NULL"
        return "OK"

    df["verdict"] = df.apply(verdict, axis=1)
    df.to_parquet(args.out.with_suffix(".parquet"), index=False)
    df.to_csv(args.out, index=False)
    print(f"\nwrote {args.out}")

    print("\n=== verdict counts by layer (all 50 concepts) ===")
    print(pd.crosstab(df.layer, df.verdict).to_string())
    print("\n=== verdict counts by layer, PILOT concepts only ===")
    p = df[df.split == "pilot"]
    if len(p):
        print(pd.crosstab(p.layer, p.verdict).to_string())


if __name__ == "__main__":
    main()
