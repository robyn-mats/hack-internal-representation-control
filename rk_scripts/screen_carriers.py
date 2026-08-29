"""Screen candidate carrier sentences for concept contamination.

A carrier is the sentence the model is told to repeat while thinking / not
thinking about a concept word. A carrier that already evokes one of the 50
concepts confounds that word's measurement (the paper's own list contains
"Snowflakes drifted lazily from the gray sky." alongside the concept "Snow",
"Fresh bread was baking in the oven." alongside "Bread", and so on).

This script embeds all 50 CONCEPT_WORDS_PAPER and every candidate carrier with
Gemma itself -- no separate sentence embedder -- and writes the full
carrier x concept x layer cosine table, so screening thresholds and layers can
be chosen afterwards without touching a GPU.

Encoding (identical for both pools, so the cosines are apples-to-apples):
bare text, no chat template, mean-pooled resid_post over the item's own tokens
with position 0 (BOS) excluded, captured at every layer in one forward pass.
Concept words are lowercased, per the prompt convention.

Two cosine variants are written:
  raw       -- cosine of the embeddings as-is. Dominated by a shared generic
               direction (see CLAUDE.md), so nearly everything scores high and
               only the *ranking* is informative.
  centered  -- each pool centered by its own per-layer mean first. This is the
               sensitive test and the default for the printed summary.

Examples:
    uv run python rk_scripts/screen_carriers.py
    uv run python rk_scripts/screen_carriers.py --carriers-file my_candidates.txt
    uv run python rk_scripts/screen_carriers.py --tag v2 --screen-layer 40
"""

from irc import env  # noqa: F401  -- must be the first import

import dataclasses
import json
import subprocess
import time
from pathlib import Path

import torch
import tyro

from irc import constants
from irc.model import MODEL_ID, ResidualCapture, load_model, load_tokenizer
from irc.paths import ARTIFACTS
from irc.words_paper import CONCEPT_WORDS_PAPER, SENTENCES_PAPER

OUT_ROOT = ARTIFACTS / "carrier_screen"


@dataclasses.dataclass
class Config:
    tag: str = "paper"
    """Output subdirectory under artifacts/carrier_screen/."""
    carriers_file: Path | None = None
    """Extra candidate carriers, one sentence per line (blank lines and #
    comments ignored). Appended to the paper's 50 unless they are disabled."""
    include_paper_carriers: bool = True
    """Include SENTENCES_PAPER as candidates."""
    concept_template: str = "{word}"
    """Context the concept word is embedded in. Default is the bare word; e.g.
    "I've been thinking about {word}." embeds it in a sentence instead."""
    layers: tuple[int, ...] = ()
    """Layers to embed at. Empty = all N_LAYERS (one forward pass either way)."""
    screen_layer: int = 31
    """Layer used for the printed summary and summary_L{n}_centered.csv."""
    top_n: int = 3
    """Nearest concepts recorded per carrier in the summary."""
    show: int = 15
    """Rows printed (worst carriers first). 0 = print nothing."""


def load_carriers(cfg: Config) -> list[str]:
    carriers = list(SENTENCES_PAPER) if cfg.include_paper_carriers else []
    if cfg.carriers_file is not None:
        extra = [
            line.strip()
            for line in cfg.carriers_file.read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        print(f"[carriers] +{len(extra)} from {cfg.carriers_file}")
        carriers += extra
    carriers = list(dict.fromkeys(carriers))
    if not carriers:
        raise SystemExit("no candidate carriers (paper list disabled and no file given)")
    return carriers


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


def embed_all(model, tokenizer, texts: list[str], layers: list[int], label: str) -> torch.Tensor:
    """(n_texts, n_layers, d_model)."""
    out = []
    for i, t in enumerate(texts):
        out.append(embed(model, tokenizer, t, layers))
        if (i + 1) % 25 == 0 or i + 1 == len(texts):
            print(f"  [embed] {label} {i + 1}/{len(texts)}")
    return torch.stack(out)


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


def main(cfg: Config) -> None:
    import pandas as pd

    carriers = load_carriers(cfg)
    concepts = list(CONCEPT_WORDS_PAPER)
    layers = list(cfg.layers) or list(range(constants.N_LAYERS))
    if cfg.screen_layer not in layers:
        raise SystemExit(f"--screen-layer {cfg.screen_layer} not in embedded layers")

    out_dir = OUT_ROOT / cfg.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"screening {len(carriers)} carriers against {len(concepts)} concepts "
          f"over {len(layers)} layers -> {out_dir}")

    tokenizer = load_tokenizer()
    model = load_model()
    t0 = time.time()
    E_concept = embed_all(
        model, tokenizer,
        [cfg.concept_template.format(word=w.lower()) for w in concepts],
        layers, "concepts",
    )
    E_carrier = embed_all(model, tokenizer, carriers, layers, "carriers")
    print(f"[embed] done in {time.time() - t0:.0f}s")

    torch.save(
        {"concepts": concepts, "carriers": carriers, "layers": layers,
         "E_concept": E_concept, "E_carrier": E_carrier},
        out_dir / "embeddings.pt",
    )

    # Full carrier x concept x layer x variant table -- everything else is
    # derivable from this, so thresholds/layers can be revisited without a GPU.
    long_rows, summary_rows = [], []
    for variant, center in (("raw", False), ("centered", True)):
        C = cosines(E_carrier, E_concept, center)  # (L, K, N)
        for li, layer in enumerate(layers):
            M = C[li]
            long_rows.append(pd.DataFrame({
                "carrier_idx": torch.arange(len(carriers)).repeat_interleave(len(concepts)).numpy(),
                "concept": concepts * len(carriers),
                "layer": layer,
                "variant": variant,
                "cos": M.flatten().numpy(),
            }))
            top = M.topk(min(cfg.top_n, len(concepts)), dim=-1)
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
    long_df.to_parquet(out_dir / "cosines.parquet", index=False)

    summary = pd.DataFrame(summary_rows)
    summary.to_parquet(out_dir / "summary.parquet", index=False)

    screen = (summary[(summary.layer == cfg.screen_layer) & (summary.variant == "centered")]
              .sort_values("max_cos", ascending=False))
    csv_path = out_dir / f"summary_L{cfg.screen_layer}_centered.csv"
    screen.drop(columns=["layer", "variant"]).to_csv(csv_path, index=False)

    commit = subprocess.run(["git", "rev-parse", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    (out_dir / "meta.json").write_text(json.dumps({
        **dataclasses.asdict(cfg),
        "carriers_file": str(cfg.carriers_file) if cfg.carriers_file else None,
        "n_carriers": len(carriers),
        "n_concepts": len(concepts),
        "layers": layers,
        "model_id": MODEL_ID,
        "encoding": ("bare text, no chat template, mean resid_post over own "
                     "tokens, BOS (position 0) excluded"),
        "concept_case": "lowercased",
        "word_lists": "irc/words_paper.py",
        "git_commit": commit,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }, indent=1, default=str))

    print(f"\nwrote {out_dir}/")
    print("  cosines.parquet  full carrier x concept x layer x variant table "
          f"({len(long_df):,} rows)")
    print("  summary.parquet  per carrier x layer x variant: max_cos, top_concept, z_max")
    print(f"  {csv_path.name}  the same at L{cfg.screen_layer}, centered, worst first")
    print("  embeddings.pt    both embedding pools (rescore without a GPU)")

    if cfg.show:
        print(f"\nworst {min(cfg.show, len(screen))} carriers "
              f"(L{cfg.screen_layer}, centered, highest max cosine first):\n")
        for _, r in screen.head(cfg.show).iterrows():
            print(f"  {r.max_cos:+.3f}  z={r.z_max:5.2f}  {r.carrier}")
            print(f"           -> {r.top_concepts}")


if __name__ == "__main__":
    main(tyro.cli(Config))
