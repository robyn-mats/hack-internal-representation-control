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

## The setup script

`/workspace/setup.sh` is now a **thin wrapper** that execs `pod_setup.sh` in the
repo, so the bootstrap is version controlled and survives a volume loss. Edit the
repo copy. The previous standalone script is kept at `/workspace/setup.sh.bak`.

`/workspace/setup.sh` itself is not on the volume's git history, so a copy of it
lives here as `pod_setup_wrapper.sh`. To recreate it:

    cp /workspace/hack-internal-representation-control/pod_setup_wrapper.sh \
       /workspace/setup.sh && chmod +x /workspace/setup.sh

Either entry point works and they behave identically -- the wrapper sources
`pod_setup.sh`. Prefer `/workspace/setup.sh`: it is the stable path if the repo
moves or is re-cloned, and it fails with a clear message if the repo is absent.

**Source it, don't `bash` it:** `source /workspace/setup.sh`. Sourcing is how
`HF_HOME`, `HF_TOKEN` and `CLAUDE_CONFIG_DIR` reach your current shell; running
it with `bash` sets them only in a subprocess that then exits, so they reach
future shells via `~/.bashrc` but not the one you are sitting in. Both work; only
sourcing helps the shell you are about to run things from. Repeated sourcing is
safe and does not stack duplicate PATH entries.

**Do NOT source the `nla_server/` scripts.** `setup.sh`, `launch.sh` and
`patches/apply_sglang_patches.sh` all use `set -euo pipefail` and bare `exit`,
so sourcing them leaves `set -e` in your interactive shell — where the next
command returning non-zero closes it — or exits it outright. Run those with
`bash`.

Re-run it after every pod start. Fixed 2026-08-30:

- **`config` was imported before `irc.env`** in the verify step, so setup always
  printed `google/gemma-3-4b-it` — the *dev* profile — on a prod pod, and called
  it "config ok". `config.py` reads `WB_MODE` from the environment at import
  time and `.env` is what sets it, so import order is load-bearing exactly as
  CLAUDE.md's first-import rule says. It now prints WB_MODE, model and layer.
- **`git config --global --add safe.directory`** ran unconditionally on a stale
  path (`$VOL/whitebear`, which does not exist), accumulating a duplicate entry
  every pod start — three had built up — while the real repo was never listed.
  Now set idempotently to `$REPO`.
- **The dead `requirements.txt` block** tested `$REPO/requirements.txt` and
  installed from `$VOL/whitebear/requirements.txt`. Removed; there is no
  requirements.txt.
- **Missing packages added**: `orjson` (was genuinely absent; needed by
  `irc/vendor/nla_inference.py`), plus `httpx` and `ipywidgets` made explicit
  rather than relied on as transitive.
- **`echo "PYTHONPATH=$PYTHONPATH"`** removed — vestigial, always printed empty,
  and superseded by the editable install.
- The verify step now reports which expected packages are missing instead of
  failing at first use mid-session.
- **The wrapper used `exec`, which closed the ssh connection.** `exec` replaces
  the calling process, so sourcing the wrapper replaced the login shell itself
  and the session ended when the script did. Also removed `set -u` (it persisted
  into the sourcing shell, making any unset variable an error afterwards) and
  the bare `exit` (it would have closed the shell outright). PATH is now
  idempotent under repeated sourcing.

`huggingface_hub[cli]` and the `$VOL/.hf_token` no-op were fixed earlier.

### Claude Code sessions do not survive a pod stop by default

`~/.claude` — session transcripts, credentials, memory — is **container disk**.
It is wiped on every pod stop, so `claude --continue` finds nothing the next
morning even though the repo is untouched.

`pod_setup.sh` now symlinks `~/.claude` to `$VOL/.claude`, which is on the
volume. A symlink rather than `CLAUDE_CONFIG_DIR` because `bash pod_setup.sh`
runs in a **subshell**: its exports never reach the shell you launch `claude`
from, so the env var only ever worked in a fresh login shell.

Two related fixes in the same commit:

- The `~/.bashrc` block was written under `if ! grep -q 'whitebear-setup'`, so
  once present it was **never updated**. The block on this pod had been carrying
  a stale `$VOL/.hf_token` line and no `CLAUDE_CONFIG_DIR` for days. It is now
  delimited by BEGIN/END markers and replaced on every run.
- Transcripts already on container disk were copied to
  `$VOL/.claude/projects/` by hand on 2026-08-30. Sessions from before that
  date are only there because of that copy.

Note the repo itself is never at risk — it is on the volume and pushed to
GitHub. What was at risk was only the conversation history.

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

## Disk: the HF cache, and why each model is there

The network volume ceiling is **200 GB**; `df` reports the underlying cluster
filesystem (hundreds of TB) and is meaningless here. `du -sh /workspace/*` is the
real measure. `/workspace/hf` is ~123 GB of it.

**Do not delete cached models on the assumption that anything absent from
PLAN.md is unused.** Two are held deliberately:

| model | size | why |
|---|---|---|
| `gemma-3-27b-it` | 52 GB | the model under test |
| `gemma-scope-2-27b-it` | 22 GB | the SAEs — primary readout |
| `gemma-3-12b-it` | 23 GB | **capacity insurance.** Every >=80 GB card in us-ks-2 was out of capacity when this was downloaded. If the pod comes back on a 48 GB card, switch `MODEL_ID` to the 12B and continue rather than lose the day — Gemma Scope 2 has 12B SAEs, so the pipeline works there. Costs direct comparability with Kamp's numbers, gains a project that happens. Still live because the pod is stopped nightly and capacity fluctuates. Also a scale rung between the 4B dev profile and the 27B if the effect is size-dependent. |
| `gemma-scope-2-4b-it` | 11 GB | SAEs for the 4B dev profile |
| `gemma-3-4b-it` | 8.1 GB | dev profile at matched relative depth (layer 22/34 ~= 40/62 ~= 65%); named in `PREREGISTRATION.md` |
| `Qwen3.5-4B` | 8.8 GB | **for the J-lens arm.** `anthropics/jacobian-lens` ships its examples on Qwen, and the plan is to get the walkthrough running there before pointing it at Gemma — Gemma 3 loads as a conditional-generation class rather than a plain causal LM, so the layer-access code likely needs adjusting. Debugging an unfamiliar method and an unfamiliar architecture at once is how a Sunday disappears. Conditional: the J-lens is third priority behind SAE latents and concept vectors, so this may never be used and can be reclaimed if the arm is dropped. |

## Storage budget for the generation runs

Activations dominate everything else. At ~6.7 MB per trial across all 62 layers:

| capture | pilot 4,690 | full 23,107 |
|---|---|---|
| all 62 layers | 31 GB | 154 GB — **does not fit** |
| 4 SAE layers | 2.0 GB | **10 GB** |

Capture `constants.SAE_LAYERS` (16/31/40/53) by default. `PREREGISTRATION.md`
restricts the analysis layer to those four anyway, so nothing the pilot decides
needs the other 58; all 62 layers are only for the layer-curve secondary figure,
which is descriptive and can come from a few hundred trials rather than all of
them.

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
