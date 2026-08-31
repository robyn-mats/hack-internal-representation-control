"""Measure stage: turn stored activations into one readout number per trial.

Reads a fork run (`artifacts/runs/{run_id}/{pass}/`) plus `stimuli.csv`, and
emits a tidy table with one row per trial x layer x pooling rule. Model-free for
the concept-vector readout; the SAE readout needs the SAEs on the GPU but not the
language model.

Three things the runner made this responsible for:

  last record wins    Re-runs APPEND to generations.jsonl rather than replacing,
                      so a prompt re-run after a template change leaves both
                      versions in the file. The last one is the live record.
  leak recomputed     `leaked_concepts` in a stored record reflects whatever
                      form list existed when it was written. pilot1 predates
                      emoji support. Leak is recomputed here from the stored
                      completion, which is the durable input.
  factor coding       Contrasts group on cell_id / direction / frame / negation,
                      which live in stimuli.csv, not in the run records.

Two implementation traps worth knowing:

  The concept-vector bank spans all 62 layers; stored acts hold only SAE_LAYERS.
  The bank must be indexed to match or `concept_cosines` einsums mismatched
  shapes into nonsense rather than erroring.

  Pooling rule 3 from PREREGISTRATION.md -- "activation at positions where the
  concept is a plausible next token" -- needs next-token logits, which are not
  stored. It is NOT implemented here. See `--pooling` and the module note below.

Usage:
    python3 rk_scripts/09_measure.py --run-id pilot2 --pass generated
    python3 rk_scripts/09_measure.py --run-id pilot2 --pass generated --limit 200
"""

from irc import env  # noqa: F401  -- must be the first import

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import torch

from irc.constants import SAE_LAYERS
from irc.paths import REPO_ROOT, RUNS
from irc.pipeline import concept_cosines, load_vector_bank

# Pooling rules.
#
# `topk_mean` was pre-registered and is DROPPED, decided before any readout was
# computed: it selects positions BY the quantity being measured, so a condition
# with higher variance scores higher at equal true mean. Since conditions are
# exactly what the contrasts compare, that is a confound rather than a nuisance.
#
# `max` is kept as a descriptive companion, not a candidate -- it has the same
# selection problem in sharper form. Only `token_mean` and `plausible` (rule 3,
# computed by 08_plausible_positions.py) are eligible to win.
POOLINGS = ("token_mean", "max")
ELIGIBLE = ("token_mean", "plausible")


def load_stimuli(path: Path) -> dict[str, dict]:
    """-> {prompt_group: row}. Carries the factor coding the contrasts need."""
    out = {}
    for r in csv.DictReader(path.open()):
        out.setdefault(r["prompt_group"], r)   # first row per group is enough
    return out


def load_records(gen_path: Path) -> dict[str, dict]:
    """-> {prompt_group: record}, LAST occurrence winning."""
    out = {}
    with gen_path.open() as fh:
        for line in fh:
            r = json.loads(line)
            out[r["prompt_group"]] = r
    return out


def leak_patterns(path: Path) -> dict[str, re.Pattern]:
    """Word-boundary for words, literal for emoji (\\b never matches an emoji)."""
    pats = {}
    for row in csv.DictReader(path.open()):
        words = (row.get("forms") or row["concept"]).split("|")
        emoji = [e for e in (row.get("forms_emoji") or "").split("|") if e]
        parts = [r"\b(?:%s)\b" % "|".join(re.escape(w) for w in
                                          sorted(words, key=len, reverse=True))]
        if emoji:
            parts.append("|".join(re.escape(e) for e in emoji))
        pats[row["concept"]] = re.compile("|".join(parts), re.IGNORECASE)
    return pats


def pool(x: torch.Tensor, weights: torch.Tensor | None = None) -> dict[str, float]:
    """Collapse the token axis of a 1-D per-token readout.

    `weights`, when given, are the rule-3 plausibility weights for the same
    positions: P(concept is the next token | prefix), from
    08_plausible_positions.py. The pooled value is their weighted mean, so the
    readout is taken where the concept could actually have surfaced.
    """
    out = {"token_mean": float(x.mean()), "max": float(x.max())}
    if weights is not None and float(weights.sum()) > 0:
        out["plausible"] = float((x * weights).sum() / weights.sum())
    return out


def load_plausibility(run_dir: Path) -> dict[str, list[float]]:
    """Rule-3 weights, if 08_plausible_positions.py has been run."""
    path = run_dir / "results" / "plausible_positions.json"
    if not path.exists():
        print("  note: no plausible_positions.json -- pooling rule 3 is absent. "
              "Run 08_plausible_positions.py first.")
        return {}
    return json.loads(path.read_text())


def measure_vectors(run_dir: Path, stimuli: dict, records: dict,
                    plaus: dict, variant: str = "word_tokens",
                    device: str = "cuda", limit: int = 0) -> list[dict]:
    """Concept-vector cosine per trial, for the target concept and a control null.

    The null is the 100 control words: `PLAN.md` notes raw cosines are dominated
    by a shared generic direction, so an absolute cosine means little and the
    control spread is what makes it interpretable.
    """
    bank = load_vector_bank(variant, device)
    w_idx, ctrl_idx = bank["w_idx"], bank["ctrl_idx"]
    # The bank spans all 62 layers; stored acts hold only SAE_LAYERS.
    Vn = bank["Vn"][:, list(SAE_LAYERS), :]

    rows, missing, n = [], 0, 0
    for pg, rec in records.items():
        st = stimuli.get(pg)
        if st is None or not rec.get("acts_file"):
            missing += 1
            continue
        concept = st["concept"]
        if concept not in w_idx:          # T7 names no concept
            continue
        A = torch.load(run_dir / rec["acts_file"]).float().to(device)
        cos = concept_cosines(A, Vn)                      # (L, W, T)
        tgt = w_idx[concept]
        w = plaus.get(pg)
        w = torch.tensor(w, device=device) if w is not None else None
        for li, layer in enumerate(SAE_LAYERS):
            p = pool(cos[li, tgt], w)
            null = cos[li, ctrl_idx].mean(-1)             # per control word
            for rule, val in p.items():
                rows.append({
                    "prompt_group": pg, "readout": "concept_vector",
                    "layer": layer, "pooling": rule, "value": val,
                    "null_mean": float(null.mean()), "null_std": float(null.std()),
                    "concept": concept, "cell_id": st["cell_id"],
                    "phrasing_id": st["phrasing_id"], "split": st["split"],
                    "direction": st["direction"], "frame_type": st["frame_type"],
                    "negation": st["negation"], "carrier_order": st["carrier_order"],
                    "exact_match": rec["exact_match"],
                    "n_resp_tokens": rec["n_resp_tokens"],
                })
        n += 1
        if limit and n >= limit:
            break
        if n % 500 == 0:
            print(f"  [vectors] {n} trials")
    if missing:
        print(f"  note: {missing} records had no stimulus row or no acts file")
    return rows


def load_selected_latents(version: str = "v2") -> dict[str, dict[int, list[int]]]:
    """-> {concept_lower: {layer: [latent indices]}} from artifacts/latents_{version}.

    Files are keyed by the capitalized word as stored in irc/words_paper.py;
    stimuli carry the lowercased form, so the map is lowercased here.
    """
    from irc.paths import ARTIFACTS
    from irc.words_paper import CONCEPT_WORDS_PAPER

    out, missing = {}, []
    for word in CONCEPT_WORDS_PAPER:
        path = ARTIFACTS / f"latents_{version}" / f"{word}.json"
        if not path.exists():
            missing.append(word)
            continue
        d = json.loads(path.read_text())
        out[word.lower()] = {int(l): [e["latent"] for e in ents]
                             for l, ents in d["layers"].items()}
    if missing:
        raise SystemExit(f"no latent selection for {len(missing)} concepts: "
                         f"{missing[:5]}... run scripts/run_pipeline.py "
                         f"--stages latents")
    return out


def measure_sae(run_dir: Path, stimuli: dict, records: dict, plaus: dict,
                latents_version: str = "v2", device: str = "cuda",
                limit: int = 0) -> list[dict]:
    """SAE latent activation per trial -- the PRIMARY readout (PLAN.md section 4).

    Two ways of collapsing the 5 selected latents at a layer, both recorded:

      latent_sum   per-token total across the concept's latents. The natural
                   "how much of this concept is active" quantity, and what
                   upstream reports as act_sum_mean.
      latent_max   per-token strongest single latent. Less diluted if only one
                   of the five is the real concept latent -- which the selection
                   cannot guarantee, since it ranks contrastively rather than
                   verifying meaning.

    Both are then pooled over tokens by the same rules as the vector readout.
    """
    from irc.pipeline import load_saes

    saes = load_saes(list(SAE_LAYERS), device)
    sel = load_selected_latents(latents_version)

    rows, n, skipped = [], 0, 0
    for pg, rec in records.items():
        st = stimuli.get(pg)
        if st is None or not rec.get("acts_file"):
            continue
        concept = st["concept"]
        if concept not in sel:          # T7 names no concept
            skipped += 1
            continue
        A = torch.load(run_dir / rec["acts_file"]).to(device)
        w = plaus.get(pg)
        w = torch.tensor(w, device=device) if w is not None else None
        for li, layer in enumerate(SAE_LAYERS):
            sae = saes[layer]
            feats = sae.encode(A[li].to(sae.dtype))            # (T, d_sae)
            idx = sel[concept][layer]
            chosen = feats[:, idx].float()                     # (T, k)
            for name, per_token in (("latent_sum", chosen.sum(-1)),
                                    ("latent_max", chosen.max(-1).values)):
                for rule, val in pool(per_token, w).items():
                    rows.append({
                        "prompt_group": pg, "readout": name,
                        "layer": layer, "pooling": rule, "value": val,
                        "n_latents": len(idx),
                        "concept": concept, "cell_id": st["cell_id"],
                        "phrasing_id": st["phrasing_id"], "split": st["split"],
                        "direction": st["direction"], "frame_type": st["frame_type"],
                        "negation": st["negation"], "carrier_order": st["carrier_order"],
                        "exact_match": rec["exact_match"],
                        "n_resp_tokens": rec["n_resp_tokens"],
                    })
        n += 1
        if limit and n >= limit:
            break
        if n % 500 == 0:
            print(f"  [sae] {n} trials")
    print(f"  [sae] {n} trials measured, {skipped} concept-free skipped")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--pass", dest="pass_", default="generated",
                    choices=["generated", "teacher_forced"])
    ap.add_argument("--variant", default="word_tokens")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--stimuli", type=Path, default=REPO_ROOT / "stimuli.csv")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--readout", default="both",
                    choices=["concept_vector", "sae", "both"])
    ap.add_argument("--latents-version", default="v2")
    args = ap.parse_args()

    run_dir = RUNS / args.run_id / args.pass_
    gen_path = run_dir / "generations.jsonl"
    if not gen_path.exists():
        raise SystemExit(f"no run at {gen_path}")

    stimuli = load_stimuli(args.stimuli)
    records = load_records(gen_path)
    raw = sum(1 for _ in gen_path.open())
    print(f"{args.run_id}/{args.pass_}: {raw} records -> {len(records)} unique "
          f"prompt_groups (last wins)")

    pats = leak_patterns(REPO_ROOT / "irc" / "concepts.csv")
    leaks = {pg: [c for c, p in pats.items() if p.search(r["completion"])]
             for pg, r in records.items()}
    stale = sum(1 for pg, r in records.items()
                if sorted(r.get("leaked_concepts") or []) != sorted(leaks[pg]))
    print(f"  leak recomputed from completions; {stale} records disagreed with "
          f"their stored leaked_concepts")

    plaus = load_plausibility(run_dir)
    out_dir = run_dir / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    import pandas as pd

    if args.readout in ("concept_vector", "both"):
        rows = measure_vectors(run_dir, stimuli, records, plaus, args.variant,
                               args.device, args.limit)
        path = out_dir / f"readout_concept_vector_{args.variant}.parquet"
        pd.DataFrame(rows).to_parquet(path, index=False)
        print(f"wrote {path} ({len(rows):,} rows)")

    if args.readout in ("sae", "both"):
        rows_sae = measure_sae(run_dir, stimuli, records, plaus,
                               args.latents_version, args.device, args.limit)
        path = out_dir / f"readout_sae_{args.latents_version}.parquet"
        pd.DataFrame(rows_sae).to_parquet(path, index=False)
        print(f"wrote {path} ({len(rows_sae):,} rows)")
        rows = rows if args.readout == "both" else rows_sae

    df = pd.DataFrame(rows)
    path = out_dir / f"readout_concept_vector_{args.variant}.parquet"
    print(f"\nwrote {path}  ({len(df):,} rows, "
          f"{df.prompt_group.nunique():,} trials x {len(SAE_LAYERS)} layers x "
          f"{df.pooling.nunique()} poolings)")
    print(f"  poolings present: {sorted(df.pooling.unique())}")
    print(f"  eligible to win the Stage 2 comparison: {list(ELIGIBLE)}")


if __name__ == "__main__":
    main()
