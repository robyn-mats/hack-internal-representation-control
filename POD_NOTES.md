# Running on the RunPod pod

Environment notes for a Claude Code instance working on the GPU pod, as opposed
to a laptop checkout. Read this alongside CLAUDE.md — **where the two disagree
about how to run things, this file wins on the pod.**

## The big one: do not use `uv` here

CLAUDE.md says everything runs through `uv run`. That is true on a laptop and
false on this pod. The pod's stack does not satisfy `pyproject.toml`:

| | pyproject wants | pod has |
|---|---|---|
| torch | >=2.12.1 | 2.8.0+cu128 (works, CUDA live) |
| numpy | >=2.5.1 | 2.1.2 |
| orjson | >=3.11.9 | not installed |

`uv sync` / `uv run` would build a `.venv` with a different multi-GB torch and
risk breaking CUDA. Use `/usr/bin/python3` directly. When installing anything
from the repo, `pip install -e . --no-deps` — without `--no-deps` pip reads
pyproject and starts upgrading torch.

pip will print an `ERROR:` block about unsatisfied dependencies after any such
install. That is pip's post-install consistency *check*, not a failure, and it
is expected. `orjson` is only imported by `irc/vendor/nla_inference.py`, so it
matters only if you run `scripts/nla_explain.py`.

## Layout

- Repo: `/workspace/hack-internal-representation-control` (the git root).
- `/workspace/setup.sh` — pod bootstrap, **outside the repo**, not version
  controlled. Exports `HF_HOME=/workspace/hf`, pip-installs packages, installs
  claude code, sets git config. Re-run after every pod start.
- `/workspace` is the persistent volume. **Everything else is container disk and
  resets when the pod restarts**, including all of system site-packages. That is
  why setup.sh reinstalls packages every time.
- Weights live in `/workspace/hf` (~55 GB for gemma-3-27b-it), so they survive.

## Known issues in /workspace/setup.sh

Not yet fixed at the time of writing:

- `huggingface_hub[cli]` — hf-hub 1.x dropped that extra; warns. Drop `[cli]`.
- `PYTHONPATH=$REPO` is exported but **omitted from the `~/.bashrc` block**, so
  new shells lose it. Superseded anyway by the editable install (below); the
  cleanest fix is to delete the PYTHONPATH export entirely.
- `git config --global --add safe.directory $VOL/whitebear` — stale path, that
  directory does not exist. Should be `$REPO`.
- The requirements.txt branch tests `$REPO/requirements.txt` but installs from
  `$VOL/whitebear/requirements.txt`. Dead code; there is no requirements.txt.
- Missing packages that the notebook and scripts need. The pip line should add:
  `python-dotenv matplotlib tyro pyarrow ipywidgets`, plus
  `pip install -e "$REPO" --no-deps`.

## Auth and .env

`setup.sh` reads the HF token from `/workspace/.hf_token`, **which does not
exist** — so that line has always been a silent no-op and `HF_TOKEN` was never
exported. The real token is at `/workspace/hf/token`, written by
`hf auth login` (i.e. `$HF_HOME/token`). `huggingface_hub` finds it on its own,
which is why downloads worked, but `irc/env.py:require_hf_token()` checks
`os.environ["HF_TOKEN"]` specifically and raises before reaching the hub.

The repo `.env` (gitignored) now carries it, which works regardless of how a
process was launched, because `irc/env.py` loads it at import:

    HF_HOME=/workspace/hf
    HF_TOKEN=<from /workspace/hf/token>
    WB_MODE=prod

A `PYTHONPATH=.` line used to be in there. It was inert — Python reads
PYTHONPATH at interpreter startup, so setting it via `load_dotenv` afterwards
does nothing — and has been removed.

## GPU: check before you allocate

A100 80 GB. gemma-3-27b-it in bf16 occupies ~53 GB and takes ~9 minutes to
load. **A Jupyter kernel may be holding it.** Always run `nvidia-smi` before
anything that touches the GPU: the `Processes` table names the PID and its
usage, and `Memory-Usage` in the top table gives the total. Killing or OOMing
someone's resident model costs them a 9-minute reload.

## Jupyter / VS Code

Work happens in `rk_scripts/gemma_session.ipynb`, driven from VS Code over
Remote-SSH, loading the model once and reusing it across cells.

- RunPod runs its own Jupyter server on **8888**, rooted at `/`. Get its URL and
  token with `jupyter server list`; VS Code attaches via "Existing Jupyter
  Server". Port 8888 is taken, so a second server needs another port.
- The kernel is `/usr/bin/python3`. It does **not** source `~/.bashrc`, so it
  inherits none of setup.sh's exports — hence `.env` being the reliable channel.
- `irc` is importable from any cwd only because of the editable install; before
  that, cell 1 failed since VS Code starts kernels with cwd = the notebook's
  directory.
- Editing the notebook file on disk (e.g. `git pull`) while it is open reloads
  the editor and drops displayed outputs. Kernel *state* is unaffected — objects
  live in the kernel process, not the file.

## Current work

Screening the paper's 50 carrier sentences for concept contamination before
running the intentional-control experiment: a carrier that already evokes a
concept word (the paper's list pairs "Snowflakes drifted lazily from the gray
sky." with the concept Snow, "Fresh bread was baking in the oven." with Bread)
confounds that word's measurement.

- `rk_scripts/screen_carriers.py` — CLI plus importable pieces (`embed_all`,
  `build_tables`, `screen_view`, `save_outputs`). Embeds concepts and carriers
  with Gemma itself, all layers in one pass, and writes the full
  carrier x concept x layer cosine table so thresholds and layers can be chosen
  afterwards without a GPU. Two cosine variants: `raw` and `centered` (each pool
  centered by its own per-layer mean — raw cosines are dominated by a shared
  generic direction).
- `rk_scripts/gemma_session.ipynb` — the interactive version.

Open questions, both meant to be settled by the notebook's §5 diagnostic (which
ranks layers by how well they recover known-contaminated pairs), not by taste:

1. **Which concept encoding.** §3 builds two pools: `E_concept_wt` (word placed
   in the four `WORD_TEMPLATES_V1`, pooled over only its own token positions —
   the method `irc/concept_vectors.py` uses) and `E_concept_bare` (the bare
   word). Note `word_token_activations` wraps text in the chat template while
   carriers are bare text, so the pools differ in framing as well as pooling.
2. **Which layer.** Screening at `config.LAYER` = 40 aligns the screen with the
   experiment's SAE readout layer, which is the defensible default. If §5 shows
   layer 40 ranks the known pairs badly, that is evidence the embedding-cosine
   proxy is weak there — not a reason to screen at a layer you do not measure
   at. The sharper alternative is to screen with the actual instrument: the
   `word_tokens` concept vectors cosined against carrier response-token
   activations at layer 40, which is literally the quantity the experiment
   reports.

**Never run end to end on real hardware:** the embedding path in
`screen_carriers.py`. Its table-building and I/O were verified against stubbed
embeddings only.
