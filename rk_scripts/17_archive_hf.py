"""Archive the expensive artifacts to a private HuggingFace dataset repo.

What is worth archiving is exactly what cannot be cheaply recreated:

  packed activations   ~8.3 GB. 15 GPU-hours to regenerate. Residual stream at
                       layers 16/31/40/53, which is **SAE-width independent** --
                       so this is precisely what the pre-registered 262k arm
                       needs, and that arm does not require re-generating
                       anything.
  concept vector bank  383 MB. Re-derivable, but costs a model load and 150
                       words x 4 templates of forward passes.
  latents_v2           40 MB. The 16k selection the pilot used. Keeping it is
                       what makes the pilot reproducible after the switch to
                       262k, per CLAUDE.md.
  neuronpedia cache    28 KB. Auto-interp labels; the API may not return the
                       same strings later.

NOT archived, because a re-download is cheaper than storage: the 125 GB of
model and SAE weights under $HF_HOME.

Everything else the analysis needs (parquets, generations.jsonl, stimuli,
conditions, concepts) is small enough to live in git and is already committed.

Writes a manifest with a sha256 per file so a later restore can be verified
rather than assumed.

Requires a **write** token in .env as HF_TOKEN; the read token that suffices for
downloading models will fail here. Create one at
https://huggingface.co/settings/tokens -- a fine-grained token scoped to write
on this one dataset repo is the tighter choice.

    python3 rk_scripts/17_archive_hf.py --repo RobynMATS/whitebear-acts --dry-run
    python3 rk_scripts/17_archive_hf.py --repo RobynMATS/whitebear-acts
"""

from irc import env  # noqa: F401

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

from irc.paths import ARTIFACTS

REPO_ROOT = Path(__file__).resolve().parents[1]


def collect(runs: list[str]) -> list[tuple[Path, str]]:
    """[(local path, path in repo)] for everything worth archiving."""
    items: list[tuple[Path, str]] = []
    for run in runs:
        for p in ("generated", "teacher_forced"):
            d = ARTIFACTS / "runs" / run / p
            for name in ("acts_packed.pt", "acts_index.json",
                         "generations.jsonl", "invocations.jsonl"):
                f = d / name
                if f.exists():
                    items.append((f, f"runs/{run}/{p}/{name}"))
    bank = ARTIFACTS / "concept_vectors" / "bank_word_tokens_v1.pt"
    if bank.exists():
        items.append((bank, "concept_vectors/bank_word_tokens_v1.pt"))
    lat = ARTIFACTS / "latents_v2"
    if lat.is_dir():
        for f in sorted(lat.glob("*.json")):
            items.append((f, f"latents_v2/{f.name}"))
    npc = ARTIFACTS / "neuronpedia_cache.json"
    if npc.exists():
        items.append((npc, "neuronpedia_cache.json"))
    return items


def sha256(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def readme(commit: str, items: list[tuple[Path, str]], total: int) -> str:
    n_acts = sum(1 for _, r in items if r.endswith("acts_packed.pt"))
    return f"""---
license: other
tags:
  - interpretability
  - sparse-autoencoders
  - activations
---

# whitebear — stored activations

Residual-stream activations and derived artifacts for an instruction-frame
ablation on `google/gemma-3-27b-it`, forking the "intentional control"
experiment from *Emergent Introspective Awareness in Large Language Models*.

Code and all small data: see the project repository. This dataset holds only
what is expensive to recreate.

Generated at repo commit `{commit}`.
{len(items)} files, {total / 1e9:.2f} GB, {n_acts} packed activation tensors.

## Layout

    runs/{{run_id}}/{{pass}}/acts_packed.pt     (n_layers, total_tokens, d_model) bf16
    runs/{{run_id}}/{{pass}}/acts_index.json    per-trial offset and length
    runs/{{run_id}}/{{pass}}/generations.jsonl  completions, exactness, provenance
    runs/{{run_id}}/{{pass}}/invocations.jsonl  run provenance
    concept_vectors/bank_word_tokens_v1.pt    mean-difference concept vectors
    latents_v2/{{Concept}}.json                16k SAE latent selections
    neuronpedia_cache.json                    auto-interp labels
    MANIFEST.json                             sha256 per file

## Reading an activation

Trials are **concatenated** along the token axis, not padded:

```python
import json, torch
packed = torch.load("runs/pilot2/generated/acts_packed.pt", map_location="cpu")
index = json.loads(open("runs/pilot2/generated/acts_index.json").read())
byname = {{t["prompt_group"]: t for t in index["trials"]}}
t = byname["A1|dust|3"]
A = packed[:, t["start"]:t["start"] + t["length"], :]   # (n_layers, tokens, d_model)
```

Layers are `(16, 31, 40, 53)` in that order — the widths with published
Gemma Scope 2 SAEs. **Token position 0 (BOS) was never captured**, so no
BOS exclusion is needed on read.

## Why the activations are the reusable part

They are residual-stream reads, so they are independent of SAE width. The
pre-registered confirmatory arm switches from the 16k SAEs to 262k; that
requires downloading the 262k weights and re-running latent selection, but
**not** re-generating any activation. `latents_v2/` is kept so the 16k pilot
stays reproducible after the switch.

## Not included

The ~125 GB of model and SAE weights, which re-download from the Hub.
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", required=True,
                    help="e.g. RobynMATS/whitebear-acts")
    ap.add_argument("--runs", nargs="+", default=["pilot2", "heldout1"])
    ap.add_argument("--public", action="store_true",
                    help="default is PRIVATE; pass this only deliberately")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip-hashes", action="store_true",
                    help="skip sha256 (faster, but the manifest cannot verify)")
    args = ap.parse_args()

    items = collect(args.runs)
    if not items:
        raise SystemExit("nothing to archive -- run "
                         "rk_scripts/16_consolidate_acts.py first")
    total = sum(f.stat().st_size for f, _ in items)
    big = sorted(items, key=lambda it: -it[0].stat().st_size)[:8]

    print(f"{len(items)} files, {total / 1e9:.2f} GB total")
    print("largest:")
    for f, r in big:
        print(f"  {f.stat().st_size / 1e9:8.3f} GB  {r}")

    if any(r.endswith("acts_packed.pt") for _, r in items):
        print("\npacked activations present")
    else:
        print("\nWARNING: no acts_packed.pt found. Archiving unpacked "
              "per-trial files is a bad idea -- run 16_consolidate_acts.py.")

    if args.dry_run:
        print("\n--dry-run: nothing uploaded.")
        return

    tok = os.environ.get("HF_TOKEN")
    if not tok:
        raise SystemExit("HF_TOKEN not set (irc.env loads it from .env)")

    from huggingface_hub import HfApi
    api = HfApi()
    who = api.whoami(token=tok)
    role = who.get("auth", {}).get("accessToken", {}).get("role")
    print(f"\nuser {who.get('name')}, token role {role}")
    if role == "read":
        raise SystemExit(
            "This is a READ-only token; upload will fail. Create a token with "
            "write access at https://huggingface.co/settings/tokens and set "
            "HF_TOKEN in .env. A fine-grained token scoped to write on just "
            "this dataset repo is the tighter option.")

    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                            text=True, cwd=REPO_ROOT).stdout.strip()

    api.create_repo(args.repo, repo_type="dataset", private=not args.public,
                    exist_ok=True, token=tok)
    print(f"repo {args.repo} ready (private={not args.public})")

    manifest = {"git_commit": commit, "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "runs": args.runs, "files": []}
    for i, (f, rpath) in enumerate(items, 1):
        size = f.stat().st_size
        entry = {"path": rpath, "bytes": size}
        if not args.skip_hashes:
            entry["sha256"] = sha256(f)
        manifest["files"].append(entry)
        if i % 25 == 0 or size > 1e8:
            print(f"  hashed {i}/{len(items)}  {rpath}", flush=True)

    (REPO_ROOT / "archive_manifest.json").write_text(
        json.dumps(manifest, indent=1))

    # README and manifest first, so a partial upload is still self-describing.
    api.upload_file(path_or_fileobj=readme(commit, items, total).encode(),
                    path_in_repo="README.md", repo_id=args.repo,
                    repo_type="dataset", token=tok)
    api.upload_file(path_or_fileobj=json.dumps(manifest, indent=1).encode(),
                    path_in_repo="MANIFEST.json", repo_id=args.repo,
                    repo_type="dataset", token=tok)
    print("uploaded README.md and MANIFEST.json")

    for i, (f, rpath) in enumerate(items, 1):
        t0 = time.time()
        api.upload_file(path_or_fileobj=str(f), path_in_repo=rpath,
                        repo_id=args.repo, repo_type="dataset", token=tok)
        mb = f.stat().st_size / 1e6
        el = time.time() - t0
        print(f"  [{i}/{len(items)}] {rpath}  {mb:.1f} MB in {el:.1f}s "
              f"({mb / max(el, 1e-9):.1f} MB/s)", flush=True)

    print(f"\ndone: https://huggingface.co/datasets/{args.repo}")
    print("Verify a restore against MANIFEST.json before deleting anything "
          "from the volume.")


if __name__ == "__main__":
    main()
