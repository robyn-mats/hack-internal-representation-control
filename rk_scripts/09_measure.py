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
import time
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
# `plausible` (rule 3) was found degenerate on the pilot and dropped by the
# 2026-08-31 amendment (commit 61ab4f2), leaving token_mean as the fixed rule.
# It is still computed where weights exist, so the finding stays checkable.
ELIGIBLE = ("token_mean",)


def load_stimuli(path: Path) -> dict[str, list[dict]]:
    """-> {prompt_group: [rows]}, ALL rows for the group.

    Most prompt_groups have exactly one row. The concept-free baseline does not:
    T7 (`base_absent`) has no third line, so its prompt never names a concept and
    one prompt per carrier is shared by all 50 concepts -- 350 stimulus rows over
    7 prompt_groups. An earlier `setdefault` kept only the first, which measured
    those 7 activations against ONE arbitrary concept. But T7 is what the
    pre-registered analysis-layer and pooling choices are separation *against*,
    and that needs it read out per concept. So every row is kept and the readout
    is expanded over them: the stored activation is the same tensor, only the
    concept vector / latent set applied to it differs.

    Rows of both splits are returned. Each output row carries its own `split`,
    so a pilot run yields T7 baselines labeled for held-out concepts too;
    downstream must filter on `split`, which is authoritative.
    """
    out: dict[str, list[dict]] = {}
    for r in csv.DictReader(path.open()):
        out.setdefault(r["prompt_group"], []).append(r)
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


def pool_batch(x: torch.Tensor, weights: torch.Tensor | None = None,
               wsum: float = 0.0) -> dict[str, torch.Tensor]:
    """Collapse the LAST axis (tokens) of a readout, vectorized over the rest.

    Returns device tensors, not floats. Every `float()` on a CUDA tensor is a
    host sync; the scalar version cost ~20 syncs per trial (three poolings x
    four layers, plus the two null moments), which dominated runtime and left
    the process spinning at 500% CPU with the GPU idle. The caller now stacks
    all of a trial's results and transfers once.

    `wsum` is the plausibility weights' sum, computed in Python from the stored
    list so that testing it costs no sync either.
    """
    out = {"token_mean": x.mean(-1), "max": x.max(-1).values}
    if weights is not None and wsum > 0:
        out["plausible"] = (x * weights).sum(-1) / wsum
    return out


def pool(x: torch.Tensor, weights: torch.Tensor | None = None) -> dict[str, float]:
    """Scalar form of pool_batch, for a single 1-D per-token readout.

    Defined in terms of pool_batch so the two cannot drift apart on what a
    rule means.
    """
    wsum = float(weights.sum()) if weights is not None else 0.0
    return {k: float(v) for k, v in pool_batch(x, weights, wsum).items()}


def load_plausibility(run_dir: Path) -> dict[str, list[float]]:
    """Rule-3 weights, if 08_plausible_positions.py has been run."""
    path = run_dir / "results" / "plausible_positions.json"
    if not path.exists():
        print("  note: no plausible_positions.json -- pooling rule 3 is absent. "
              "Run 08_plausible_positions.py first.")
        return {}
    return json.loads(path.read_text())


def prefetch_acts(run_dir: Path, records: dict, workers: int = 32):
    """Yield (prompt_group, record, activation tensor), reading files ahead.

    The activations live on a network filesystem (MooseFS on the pod) where a
    single open+read costs most of a second in round trips but concurrent reads
    parallelize almost perfectly: measured 1.1 files/s sequential against
    42-149 files/s at 48-way concurrency on the same directory. A sequential
    torch.load per trial therefore spends ~99% of the time waiting, and the
    stage looks compute-bound while the GPU sits idle.

    Order is preserved so output rows do not depend on thread scheduling.
    Tensors stay on the CPU here; the caller moves them, because CUDA calls
    from worker threads would serialize on the context anyway.
    """
    from concurrent.futures import ThreadPoolExecutor

    items = [(pg, rec) for pg, rec in records.items() if rec.get("acts_file")]

    def load(item):
        pg, rec = item
        try:
            return pg, rec, torch.load(run_dir / rec["acts_file"],
                                       map_location="cpu")
        except Exception as exc:                      # noqa: BLE001
            print(f"  warn: could not load {rec['acts_file']}: {exc}",
                  flush=True)
            return pg, rec, None

    with ThreadPoolExecutor(max_workers=workers) as pool:
        # map() with a bounded chunk keeps at most `workers * 4` tensors in
        # flight; the full set would be ~1.8 GB for a pilot pass and ~7 GB for
        # held-out, which is real memory inside a 233 GiB container limit.
        window = workers * 4
        for start in range(0, len(items), window):
            for pg, rec, A in pool.map(load, items[start:start + window]):
                if A is not None:
                    yield pg, rec, A


@torch.no_grad()
def measure_vectors(run_dir: Path, stimuli: dict, records: dict,
                    plaus: dict, variant: str = "word_tokens",
                    device: str = "cuda", limit: int = 0) -> list[dict]:
    """Concept-vector cosine per trial, for the target concept and a control null.

    The null is the 100 control words: `PLAN.md` notes raw cosines are dominated
    by a shared generic direction, so an absolute cosine means little and the
    control spread is what makes it interpretable.

    One trial may expand to several output rows -- see load_stimuli on the T7
    baseline, whose prompt names no concept and is therefore read out once per
    concept sharing it. The cosines against the whole bank are computed anyway,
    so expansion costs only extra row-building.
    """
    bank = load_vector_bank(variant, device)
    # The bank keys words as stored in irc/words_paper.py -- CAPITALIZED --
    # while stimuli.csv carries the lowercased form used in prompts. Without
    # this the lookup misses every trial and the readout silently writes 0 rows.
    w_idx = {k.lower(): v for k, v in bank["w_idx"].items()}
    # Keep the control index on the same device as what it indexes: a CPU index
    # tensor against a CUDA tensor forces a transfer on every use.
    ctrl_idx = bank["ctrl_idx"].to(device)
    # The bank spans all 62 layers; stored acts hold only SAE_LAYERS.
    Vn = bank["Vn"][:, list(SAE_LAYERS), :]

    rows, missing, n, unknown = [], 0, 0, set()
    t0 = time.time()
    for pg, rec, A_cpu in prefetch_acts(run_dir, records):
        sts = stimuli.get(pg)
        if not sts:
            missing += 1
            continue
        keep = [st for st in sts if st["concept"] in w_idx]
        unknown.update(st["concept"] for st in sts if st["concept"] not in w_idx)
        if not keep:
            continue

        A = A_cpu.float().to(device)
        cos = concept_cosines(A, Vn)                       # (L, W, T)

        wl = plaus.get(pg)
        wsum = float(sum(wl)) if wl else 0.0
        w = (torch.tensor(wl, device=device, dtype=cos.dtype)
             if wl and wsum > 0 else None)

        tgt = torch.tensor([w_idx[st["concept"]] for st in keep], device=device)
        pooled = pool_batch(cos[:, tgt, :], w, wsum)       # rule -> (L, C)
        null = cos[:, ctrl_idx, :].mean(-1)                # (L, n_ctrl)

        # Two host transfers for the whole trial, instead of one per scalar.
        rules = [r for r in ("token_mean", "max", "plausible") if r in pooled]
        packed = (torch.stack([pooled[r] for r in rules])      # (R, L, C)
                       .reshape(len(rules), -1).tolist())
        null_stats = torch.stack([null.mean(-1), null.std(-1)]).tolist()

        C = len(keep)
        for ri, rule in enumerate(rules):
            flat = packed[ri]
            for li, layer in enumerate(SAE_LAYERS):
                nm, ns = null_stats[0][li], null_stats[1][li]
                for ci, st in enumerate(keep):
                    rows.append({
                        "prompt_group": pg, "readout": "concept_vector",
                        "layer": layer, "pooling": rule,
                        "value": flat[li * C + ci],
                        "null_mean": nm, "null_std": ns,
                        "concept": st["concept"], "cell_id": st["cell_id"],
                        "phrasing_id": st["phrasing_id"], "split": st["split"],
                        "direction": st["direction"],
                        "frame_type": st["frame_type"],
                        "negation": st["negation"],
                        "carrier_order": st["carrier_order"],
                        "exact_match": rec["exact_match"],
                        "n_resp_tokens": rec["n_resp_tokens"],
                    })
        n += 1
        if limit and n >= limit:
            break
        if n % 250 == 0:
            el = time.time() - t0
            print(f"  [vectors] {n} trials, {len(rows):,} rows, "
                  f"{n / el:.1f} trial/s, eta "
                  f"{(len(records) - n) / max(n / el, 1e-9) / 60:.1f} min",
                  flush=True)
    if missing:
        print(f"  note: {missing} records had no stimulus row or no acts file")
    if unknown:
        print(f"  note: {len(unknown)} concepts absent from the vector bank: "
              f"{sorted(unknown)[:5]}")
    print(f"  [vectors] {n} trials measured -> {len(rows):,} rows in "
          f"{(time.time() - t0) / 60:.1f} min", flush=True)
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


@torch.no_grad()
def measure_sae(run_dir: Path, stimuli: dict, records: dict, plaus: dict,
                latents_version: str = "v2", device: str = "cuda",
                limit: int = 0) -> list[dict]:
    """SAE latent activation per trial -- the PRIMARY readout (PLAN.md section 4).

    Two ways of collapsing the selected latents at a layer, both recorded:

      latent_sum   per-token total across the concept's latents. The natural
                   "how much of this concept is active" quantity, and what
                   upstream reports as act_sum_mean.
      latent_max   per-token strongest single latent. Less diluted if only one
                   of the five is the real concept latent -- which the selection
                   cannot guarantee, since it ranks contrastively rather than
                   verifying meaning.

    Both are then pooled over tokens by the same rules as the vector readout.

    The SAE encode is hoisted above the concept loop, so a T7 trial expanded
    over 50 concepts costs 50 column gathers rather than 50 encodes.
    """
    from irc.pipeline import load_saes

    saes = load_saes(list(SAE_LAYERS), device)
    sel = load_selected_latents(latents_version)

    rows, n, unknown = [], 0, set()
    empty: dict[tuple[str, int], bool] = {}
    t0 = time.time()
    for pg, rec, A_cpu in prefetch_acts(run_dir, records):
        sts = stimuli.get(pg)
        if not sts:
            continue
        keep = [st for st in sts if st["concept"] in sel]
        unknown.update(st["concept"] for st in sts if st["concept"] not in sel)
        if not keep:
            continue

        A = A_cpu.to(device)
        wl = plaus.get(pg)
        wsum = float(sum(wl)) if wl else 0.0

        # keyed by (readout, layer, rule, concept) -> 0-dim tensor.
        # The selection is NOT rectangular: k ranges 0..5 across (concept,
        # layer), so concepts are gathered one at a time. The SAE encode -- the
        # only expensive step -- still happens once per (trial, layer), and the
        # scalars are transferred in a single batch at the end of the trial.
        vals: dict[tuple[str, int, str, str], torch.Tensor] = {}
        n_lat: dict[str, int] = {}
        for li, layer in enumerate(SAE_LAYERS):
            sae = saes[layer]
            feats = sae.encode(A[li].to(sae.dtype))         # (T, d_sae)
            w = (torch.tensor(wl, device=device, dtype=torch.float32)
                 if wl and wsum > 0 else None)
            for st in keep:
                concept = st["concept"]
                idx = sel[concept][layer]
                if not idx:
                    # No latent survived selection for this concept at this
                    # layer. The readout does not exist; emit no row rather
                    # than a zero, which would enter the analysis as a real
                    # measurement of "no activation".
                    empty[(concept, layer)] = True
                    continue
                n_lat[concept] = len(idx)
                chosen = feats[:, idx].float()              # (T, k)
                for name, per_tok in (("latent_sum", chosen.sum(-1)),
                                      ("latent_max", chosen.max(-1).values)):
                    for rule, v in pool_batch(per_tok, w, wsum).items():
                        vals[(name, layer, rule, concept)] = v

        if not vals:
            n += 1
            continue
        # One host transfer for the whole trial.
        keys = list(vals)
        flat = torch.stack([vals[k_] for k_ in keys]).tolist()
        by_concept = {st["concept"]: st for st in keep}
        for (name, layer, rule, concept), val in zip(keys, flat):
            st = by_concept[concept]
            if True:
                rows.append({
                    "prompt_group": pg, "readout": name,
                    "layer": layer, "pooling": rule,
                    "value": val,
                    "n_latents": n_lat[concept],
                    "concept": concept, "cell_id": st["cell_id"],
                    "phrasing_id": st["phrasing_id"], "split": st["split"],
                    "direction": st["direction"],
                    "frame_type": st["frame_type"],
                    "negation": st["negation"],
                    "carrier_order": st["carrier_order"],
                    "exact_match": rec["exact_match"],
                    "n_resp_tokens": rec["n_resp_tokens"],
                })
        n += 1
        if limit and n >= limit:
            break
        if n % 250 == 0:
            el = time.time() - t0
            print(f"  [sae] {n} trials, {len(rows):,} rows, "
                  f"{n / el:.1f} trial/s, eta "
                  f"{(len(records) - n) / max(n / el, 1e-9) / 60:.1f} min",
                  flush=True)
    if unknown:
        print(f"  note: {len(unknown)} concepts had no latent selection: "
              f"{sorted(unknown)[:5]}")
    if empty:
        by_layer: dict[int, list[str]] = {}
        for (concept, layer) in sorted(empty):
            by_layer.setdefault(layer, []).append(concept)
        print(f"  note: {len(empty)} (concept, layer) cells had k=0 selected "
              f"latents and produced NO row (not a zero):")
        for layer, cs in sorted(by_layer.items()):
            print(f"        layer {layer}: {len(cs)} concepts -- "
                  f"{', '.join(sorted(cs)[:6])}")
    print(f"  [sae] {n} trials measured -> {len(rows):,} rows in "
          f"{(time.time() - t0) / 60:.1f} min", flush=True)
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

    def write(rows: list[dict], path: Path, label: str) -> None:
        """Write one readout and summarize it. Each readout summarizes itself --
        the previous version rebuilt a single `df` from whichever rows variable
        was last bound, so a --readout sae run reported the concept-vector path
        and an empty vector half crashed the summary for the SAE half too."""
        if not rows:
            raise SystemExit(
                f"{label} readout produced 0 rows -- every trial was skipped. "
                f"Check that stimuli concepts match the readout's keys "
                f"(the vector bank stores them capitalized, stimuli.csv "
                f"lowercased) and that acts_file is set on the records.")
        df = pd.DataFrame(rows)
        df.to_parquet(path, index=False)
        print(f"\nwrote {path}\n  {len(df):,} rows = "
              f"{df.prompt_group.nunique():,} trials x {df.layer.nunique()} layers "
              f"x {df.readout.nunique()} readouts x {df.pooling.nunique()} poolings",
              flush=True)
        print(f"  poolings present: {sorted(df.pooling.unique())}")
        print(f"  pooling rule fixed for analysis: {list(ELIGIBLE)}")

    if args.readout in ("concept_vector", "both"):
        write(measure_vectors(run_dir, stimuli, records, plaus, args.variant,
                              args.device, args.limit),
              out_dir / f"readout_concept_vector_{args.variant}.parquet",
              "concept-vector")

    if args.readout in ("sae", "both"):
        write(measure_sae(run_dir, stimuli, records, plaus,
                          args.latents_version, args.device, args.limit),
              out_dir / f"readout_sae_{args.latents_version}.parquet", "SAE")


if __name__ == "__main__":
    main()
