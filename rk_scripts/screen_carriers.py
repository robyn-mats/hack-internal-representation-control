"""Screen candidate carrier sentences for concept contamination.

A carrier is the sentence the model is told to repeat while thinking / not
thinking about a concept word. A carrier that already evokes one of the 50
concepts confounds that word's measurement (the paper's own list contains
"Snowflakes drifted lazily from the gray sky." alongside the concept "Snow",
"Fresh bread was baking in the oven." alongside "Bread", and so on).

This screens with Gemma itself -- no separate sentence embedder. All 50
CONCEPT_WORDS_PAPER and every candidate carrier are embedded, and the full
carrier x concept x layer cosine table is written, so screening thresholds and
layers can be chosen afterwards without touching a GPU.

Encoding (identical for both pools, so the cosines are apples-to-apples):
bare text, no chat template, mean-pooled resid_post over the item's own tokens
with BOS excluded, captured at every layer in one forward pass. Concept words
are lowercased, per the prompt convention.

Two cosine variants are written:
  raw       -- cosine of the embeddings as-is. Dominated by a shared generic
               direction (see CLAUDE.md), so nearly everything scores high and
               only the *ranking* is informative.
  centered  -- each pool centered by its own per-layer mean first. This is the
               sensitive test and the default for the printed summary.

The pieces are importable: rk_scripts/gemma_session.ipynb calls embed_all /
build_tables / screen_view / save_outputs against an already-loaded model.

Examples:
    uv run python rk_scripts/screen_carriers.py
    uv run python rk_scripts/screen_carriers.py --carriers-file my_candidates.txt
    uv run python rk_scripts/screen_carriers.py --model-id google/gemma-3-4b-it --tag dev
"""

from irc import env  # noqa: F401  -- must be the first import

import dataclasses
import json
import subprocess
import time
from pathlib import Path

import torch
import tyro

from irc.model import (
    MODEL_ID,
    ResidualCapture,
    get_decoder_layers,
    load_model,
    load_tokenizer,
)
from irc.paths import ARTIFACTS
from irc.words_paper import CONCEPT_WORDS_PAPER, SENTENCES_PAPER

OUT_ROOT = ARTIFACTS / "carrier_screen"


@dataclasses.dataclass
class Config:
    tag: str = "paper"
    """Output subdirectory under artifacts/carrier_screen/."""
    model_id: str = MODEL_ID
    """Model to embed with (e.g. google/gemma-3-4b-it for a dev pass)."""
    carriers_file: Path | None = None
    """Extra candidate carriers, one sentence per line (blank lines and #
    comments ignored). Appended to the paper's 50 unless they are disabled."""
    include_paper_carriers: bool = True
    """Include SENTENCES_PAPER as candidates."""
    concept_template: str = "{word}"
    """Context the concept word is embedded in. Default is the bare word; e.g.
    "I've been thinking about {word}." embeds it in a sentence instead."""
    layers: tuple[int, ...] = ()
    """Layers to embed at. Empty = every layer of the loaded model (one forward
    pass either way)."""
    screen_layer: int | None = None
    """Layer for the printed summary and summary_L{n}_centered.csv.
    None = the middle layer of the model."""
    top_n: int = 3
    """Nearest concepts recorded per carrier in the summary."""
    show: int = 15
    """Rows printed (worst carriers first). 0 = print nothing."""


def load_carriers(
    carriers_file: Path | None = None, include_paper: bool = True
) -> list[str]:
    """Candidate carriers: the paper's 50, plus one-per-line extras from a file."""
    carriers = list(SENTENCES_PAPER) if include_paper else []
    if carriers_file is not None:
        extra = [
            line.strip()
            for line in Path(carriers_file).read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        print(f"[carriers] +{len(extra)} from {carriers_file}")
        carriers += extra
    carriers = list(dict.fromkeys(carriers))
    if not carriers:
        raise SystemExit("no candidate carriers (paper list disabled and no file given)")
    return carriers


def all_layers(model) -> list[int]:
    """Every decoder layer index of the loaded model (34 for 4b, 62 for 27b)."""
    return list(range(len(get_decoder_layers(model))))


@torch.no_grad()
def embed(model, tokenizer, text: str, layers: list[int]) -> torch.Tensor:
    """(n_layers, d_model) fp32 cpu: mean resid_post over the text's own tokens.

    BOS is excluded -- its residual norm is ~20x other tokens and would
    otherwise dominate the mean.
    """
    ids = tokenizer(text, return_tensors="pt")["input_ids"]
    start = 1 if ids[0, 0].item() == tokenizer.bos_token_id else 0
    assert ids.shape[1] > start, f"no content tokens in {text!r}"
    with ResidualCapture(model, layers) as cap:
        model(ids.to(model.device))
    return torch.stack([cap.acts[l][0, start:].mean(dim=0) for l in layers])


def embed_all(
    model, tokenizer, texts: list[str], layers: list[int], label: str = "texts"
) -> torch.Tensor:
    """(n_texts, n_layers, d_model)."""
    out = []
    for i, t in enumerate(texts):
        out.append(embed(model, tokenizer, t, layers))
        if (i + 1) % 25 == 0 or i + 1 == len(texts):
            print(f"  [embed] {label} {i + 1}/{len(texts)}")
    return torch.stack(out)


def embed_concepts(
    model, tokenizer, layers: list[int], template: str = "{word}",
    concepts: list[str] | None = None,
) -> torch.Tensor:
    """Concept-pool embeddings, words lowercased into `template`."""
    concepts = concepts or list(CONCEPT_WORDS_PAPER)
    texts = [template.format(word=w.lower()) for w in concepts]
    return embed_all(model, tokenizer, texts, layers, "concepts")


def cosines(E_carrier: torch.Tensor, E_concept: torch.Tensor, center: bool) -> torch.Tensor:
    """(n_layers, n_carriers, n_concepts) cosine, optionally pool-centered.

    Centering subtracts each pool's own per-layer mean: carriers and concepts
    are different text distributions, so each gets its own center.
    """
    A, B = E_carrier.float(), E_concept.float()
    if center:
        A = A - A.mean(dim=0, keepdim=True)
        B = B - B.mean(dim=0, keepdim=True)
    A = A / A.norm(dim=-1, keepdim=True)
    B = B / B.norm(dim=-1, keepdim=True)
    return torch.einsum("kld,cld->lkc", A, B)


def build_tables(
    carriers: list[str],
    concepts: list[str],
    layers: list[int],
    E_carrier: torch.Tensor,
    E_concept: torch.Tensor,
    top_n: int = 3,
):
    """-> (long_df, summary_df).

    long_df is the full carrier x concept x layer x variant cosine table;
    everything else is derivable from it, so thresholds and layers can be
    revisited without a GPU. summary_df is one row per carrier x layer x
    variant with the nearest concepts.
    """
    import pandas as pd

    long_rows, summary_rows = [], []
    for variant, center in (("raw", False), ("centered", True)):
        C = cosines(E_carrier, E_concept, center)  # (L, K, N)
        for li, layer in enumerate(layers):
            M = C[li]
            long_rows.append(pd.DataFrame({
                "carrier_idx": torch.arange(len(carriers))
                                    .repeat_interleave(len(concepts)).numpy(),
                "concept": concepts * len(carriers),
                "layer": layer,
                "variant": variant,
                "cos": M.flatten().numpy(),
            }))
            top = M.topk(min(top_n, len(concepts)), dim=-1)
            for ki in range(len(carriers)):
                summary_rows.append({
                    "carrier_idx": ki,
                    "carrier": carriers[ki],
                    "layer": layer,
                    "variant": variant,
                    "max_cos": top.values[ki, 0].item(),
                    "top_concept": concepts[top.indices[ki, 0]],
                    "top_concepts": ", ".join(
                        f"{concepts[c]} {top.values[ki, j]:+.3f}"
                        for j, c in enumerate(top.indices[ki].tolist())
                    ),
                    "mean_cos": M[ki].mean().item(),
                    "std_cos": M[ki].std().item(),
                    # How far the nearest concept stands out from this carrier's
                    # own concept distribution -- threshold on this, not max_cos.
                    "z_max": ((top.values[ki, 0] - M[ki].mean()) / M[ki].std()).item(),
                })

    long_df = pd.concat(long_rows, ignore_index=True)
    long_df["carrier"] = long_df["carrier_idx"].map(dict(enumerate(carriers)))
    return long_df, pd.DataFrame(summary_rows)


def screen_view(summary, layer: int, variant: str = "centered", by: str = "max_cos"):
    """One layer/variant of the summary, worst (most contaminated) first."""
    view = summary[(summary.layer == layer) & (summary.variant == variant)]
    return view.sort_values(by, ascending=False).drop(columns=["layer", "variant"])


def print_screen(summary, layer: int, n: int = 15, variant: str = "centered") -> None:
    view = screen_view(summary, layer, variant)
    print(f"\nworst {min(n, len(view))} carriers "
          f"(L{layer}, {variant}, highest max cosine first):\n")
    for _, r in view.head(n).iterrows():
        print(f"  {r.max_cos:+.3f}  z={r.z_max:5.2f}  {r.carrier}")
        print(f"           -> {r.top_concepts}")


def save_outputs(
    out_dir: Path,
    carriers: list[str],
    concepts: list[str],
    layers: list[int],
    E_carrier: torch.Tensor,
    E_concept: torch.Tensor,
    long_df,
    summary,
    screen_layer: int,
    meta_extra: dict | None = None,
    extra_tensors: dict | None = None,
) -> Path:
    """Write the datafiles; returns out_dir."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # extra_tensors carries the concept pools NOT used for this table (see the
    # notebook's section 3): re-deciding the encoding otherwise needs the GPU again.
    torch.save(
        {"concepts": concepts, "carriers": carriers, "layers": layers,
         "E_concept": E_concept, "E_carrier": E_carrier,
         **(extra_tensors or {})},
        out_dir / "embeddings.pt",
    )
    long_df.to_parquet(out_dir / "cosines.parquet", index=False)
    summary.to_parquet(out_dir / "summary.parquet", index=False)
    csv_path = out_dir / f"summary_L{screen_layer}_centered.csv"
    screen_view(summary, screen_layer).to_csv(csv_path, index=False)

    commit = subprocess.run(["git", "rev-parse", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    (out_dir / "meta.json").write_text(json.dumps({
        "n_carriers": len(carriers),
        "n_concepts": len(concepts),
        "layers": layers,
        "screen_layer": screen_layer,
        "encoding": ("bare text, no chat template, mean resid_post over own "
                     "tokens, BOS excluded"),
        "concept_case": "lowercased",
        "word_lists": "irc/words_paper.py",
        "git_commit": commit,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        **(meta_extra or {}),
    }, indent=1, default=str))

    print(f"\nwrote {out_dir}/")
    print("  cosines.parquet  full carrier x concept x layer x variant table "
          f"({len(long_df):,} rows)")
    print("  summary.parquet  per carrier x layer x variant: max_cos, top_concept, z_max")
    print(f"  {csv_path.name}  the same at L{screen_layer}, centered, worst first")
    print("  embeddings.pt    both embedding pools (rescore without a GPU)")
    return out_dir


# ---------------------------------------------------------------------------
# Screening rule v1 -- frozen 2026-08-30. See PREREGISTRATION.md and NOTES.md.
# ---------------------------------------------------------------------------
# Layer 43, not the experiment's readout layer 40: the known-pairs diagnostic
# (notebook section 5) recovers contaminated pairs at ceiling over layers 42-54
# and only 2 of 3 semantic pairs at 40, where one known contaminant ranked 45th
# of 50. The screen and the readout answer different questions, and the screen
# is not bound to the Gemma Scope layers because it uses no SAE.
SCREEN_LAYER = 43
SCREEN_VARIANT = "centered"
Z_CUT = 2.0

# Concepts a sentence must be checked against. Screening runs against all 50,
# not just the held-out subset, so the claim does not depend on the split.
SCREEN_CONCEPTS = CONCEPT_WORDS_PAPER


def pair_table(long_df, layer: int = SCREEN_LAYER, variant: str = SCREEN_VARIANT):
    """Per (carrier, concept) contamination scores at one layer.

    Three columns, because no single normalization is sufficient:

      z_carrier  cosine standardized within each carrier, across the 50
                 concepts -- "does this concept stand out for this sentence?"
                 Blind to CLUSTER contamination: several related concepts
                 elevated together raise the sentence's own mean and sd, which
                 suppresses each individual z.
      z_concept  standardized within each concept, across the carriers --
                 "is this concept unusually close to this sentence, versus
                 other sentences?" Kills generic attractors (a handful of
                 low-frequency nouns sit near every bland sentence and would
                 otherwise dominate), but buries broadly-evocative concepts
                 like Oceans, which are close to many sentences.
      z2         min of the two. Ranks the known-contaminated pairs 1-7 of
                 2500, but inherits the z_concept blind spot, so it is
                 reported rather than used as the gate.

    The gate is the UNION of the two marginals (see screen_carriers): Lightning
    is caught only by z_concept (5.94), Oceans only by z_carrier (2.98).
    """
    import pandas as pd  # noqa: F401  -- local, matching build_tables

    c = long_df[(long_df.variant == variant) & (long_df.layer == layer)].copy()
    c["z_carrier"] = c.groupby("carrier_idx").cos.transform(
        lambda s: (s - s.mean()) / s.std())
    c["z_concept"] = c.groupby("concept").cos.transform(
        lambda s: (s - s.mean()) / s.std())
    c["z2"] = c[["z_carrier", "z_concept"]].min(axis=1)
    return c


# Sentences removed by hand after reading the survivors. The embedding gate does
# not detect implicit part-whole or entailment overlap (an orchestra contains
# trumpets; a forest is trees), and it buries broadly-evocative concepts. Manual
# exclusion only ever REMOVES candidates, is enumerated in full, and was fixed
# before any experimental data existed.
MANUAL_EXCLUSIONS: dict[str, str] = {
    "The orchestra tuned their instruments before the concert.":
        "Trumpets -- an orchestra contains trumpets; z_carrier only 1.79",
    "He assembled the furniture without reading the instructions.":
        "Contraptions -- assembling furniture is contraption-adjacent",
    "They planted tulip bulbs in the garden last fall.":
        "Frosts -- bulbs are planted ahead of frost",
    "She collected seashells every summer at the beach.":
        "Oceans -- buried by z_concept, Oceans is close to many sentences",
    "Children built sandcastles at the water's edge.":
        "Oceans -- same blind spot",
    "The comedian's joke made everyone laugh.":
        "Amphitheaters -- an amphitheatre is a comedy venue",
    "Fog shrouded the valley below the mountain.":
        "Volcanoes (z_concept 1.27) -- volcanoes are mountains",
    "The hiker followed the trail markers through the forest.":
        "Trees -- a forest entails trees",
    "The antique vase was carefully wrapped in bubble wrap.":
        "Treasures (z_concept 1.40, highest in the survivor pool)",
    "He couldn't remember where he had parked his car.":
        "Memories -- the sentence is about memory",
    "She solved the crossword puzzle in record time.":
        "no concept hit; the most generic sentence in the pool, top-5 entirely "
        "generic attractors",
    "She planted herbs in pots on the kitchen windowsill.":
        "no concept hit; vocabulary overlap ('herbs', 'windowsill') with two "
        "sentences already selected",
}

# The frozen stimulus set. Order is fixed: truncating to fewer carriers is a
# pre-registered choice, selecting which ones after seeing results would not be.
SELECTED_CARRIERS_V1: tuple[str, ...] = (
    "The train arrived precisely on schedule.",
    "The basketball bounced off the rim.",
    "The chef garnished the plate with fresh herbs.",
    "The cat jumped onto the windowsill to watch birds.",
    "The air conditioner hummed quietly in the background.",
    "The book fell open to page 217.",
    "Fragrant lilacs bloomed along the garden fence.",
)


def screen_carriers(long_df, carriers: list[str], z_cut: float = Z_CUT,
                    layer: int = SCREEN_LAYER):
    """-> (pairs, flagged_df, survivors). The automated gate, before hand review.

    A sentence is excluded if ANY concept exceeds z_cut on EITHER marginal.
    """
    import pandas as pd

    c = pair_table(long_df, layer)
    bad = c[(c.z_carrier > z_cut) | (c.z_concept > z_cut)]
    gated = set(bad.carrier)
    survivors = [s for s in carriers
                 if s not in gated and s not in MANUAL_EXCLUSIONS]
    rows = [{"carrier": s,
             "gated": s in gated,
             "manual": MANUAL_EXCLUSIONS.get(s, ""),
             "selected": s in SELECTED_CARRIERS_V1,
             "max_z_carrier": round(float(c[c.carrier == s].z_carrier.max()), 3),
             "max_z_concept": round(float(c[c.carrier == s].z_concept.max()), 3),
             "nearest": c.loc[c[c.carrier == s].z_carrier.idxmax(), "concept"]}
            for s in carriers]
    return c, bad, survivors, pd.DataFrame(rows)


def write_carrier_similarity(long_df, out_path: Path, layer: int = SCREEN_LAYER):
    """Write the committed 50x50 evidence table (CLAUDE.md's file list).

    Goes to the repo root, NOT artifacts/ -- artifacts/ is gitignored and this
    file is the published justification for the screening rule.
    """
    c = pair_table(long_df, layer)
    wide = c.pivot(index="carrier", columns="concept", values="z_carrier").round(3)
    out_path = Path(out_path)
    wide.to_csv(out_path)
    print(f"wrote {out_path}  ({wide.shape[0]}x{wide.shape[1]} z_carrier at L{layer})")
    return wide


def main(cfg: Config) -> None:
    carriers = load_carriers(cfg.carriers_file, cfg.include_paper_carriers)
    concepts = list(CONCEPT_WORDS_PAPER)

    tokenizer = load_tokenizer(cfg.model_id)
    model = load_model(cfg.model_id)
    layers = list(cfg.layers) or all_layers(model)
    screen_layer = cfg.screen_layer if cfg.screen_layer is not None else layers[len(layers) // 2]
    if screen_layer not in layers:
        raise SystemExit(f"--screen-layer {screen_layer} not in embedded layers")

    out_dir = OUT_ROOT / cfg.tag
    print(f"screening {len(carriers)} carriers against {len(concepts)} concepts "
          f"over {len(layers)} layers of {cfg.model_id} -> {out_dir}")

    t0 = time.time()
    E_concept = embed_concepts(model, tokenizer, layers, cfg.concept_template, concepts)
    E_carrier = embed_all(model, tokenizer, carriers, layers, "carriers")
    print(f"[embed] done in {time.time() - t0:.0f}s")

    long_df, summary = build_tables(
        carriers, concepts, layers, E_carrier, E_concept, cfg.top_n
    )
    save_outputs(
        out_dir, carriers, concepts, layers, E_carrier, E_concept,
        long_df, summary, screen_layer,
        # cfg's own layers/screen_layer are the *requests* (possibly empty/None);
        # save_outputs already recorded the resolved values, so don't shadow them.
        meta_extra={k: v for k, v in dataclasses.asdict(cfg).items()
                    if k not in ("layers", "screen_layer")}
        | {"carriers_file": str(cfg.carriers_file) if cfg.carriers_file else None},
    )
    if cfg.show:
        print_screen(summary, screen_layer, cfg.show)


if __name__ == "__main__":
    main(tyro.cli(Config))
