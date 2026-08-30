#!/usr/bin/env bash
# Pod bootstrap. Re-run after every pod start: /workspace is the persistent
# volume, everything else is container disk and resets, including all of system
# site-packages.
#
# Version controlled here so a volume loss does not take it with it.
# /workspace/setup.sh is a thin wrapper that execs this file.
#
# Do NOT use uv on the pod -- see POD_NOTES.md. The pod's torch (2.8.0+cu128)
# does not satisfy pyproject.toml, and `uv sync` would build a .venv with a
# different multi-GB torch and risk breaking CUDA. Hence pip and --no-deps.

# First executable lines, so they print before anything that can hang or fail.
# `exec` in the wrapper replaces the process, so $0 becomes this script either
# way and cannot reveal the wrapper -- the wrapper exports POD_SETUP_VIA instead.
echo "==> pod_setup.sh starting $(date '+%Y-%m-%d %H:%M:%S')"
echo "    script: ${BASH_SOURCE[0]}"
[ -n "${POD_SETUP_VIA:-}" ] && echo "    via:    $POD_SETUP_VIA (wrapper)"

# No `set -u`, and no bare `exit`: this script is routinely SOURCED so that its
# exports reach the calling shell. `set -u` would persist into that shell and
# make it error on any unset variable, and `exit` would close it outright --
# which on an ssh login drops the connection.
VOL=/workspace
REPO=$VOL/hack-internal-representation-control

if [ ! -d "$REPO" ]; then
  echo "repo not found at $REPO"
  return 1 2>/dev/null || exit 1
fi

# --- environment ---
export HF_HOME=$VOL/hf
export HF_HUB_ENABLE_HF_TRANSFER=1
export PATH="$HOME/.local/bin:$PATH"
export CLAUDE_CONFIG_DIR=$VOL/.claude
mkdir -p "$CLAUDE_CONFIG_DIR"
# $VOL/.hf_token never existed; `hf auth login` writes to $HF_HOME/token.
# irc/env.py reads HF_TOKEN from the repo .env, which works regardless of how a
# process was launched -- a Jupyter kernel does not source ~/.bashrc.
if   [ -f "$VOL/.hf_token" ];  then export HF_TOKEN=$(cat "$VOL/.hf_token")
elif [ -f "$HF_HOME/token" ]; then export HF_TOKEN=$(cat "$HF_HOME/token"); fi

# Claude Code state (session transcripts, credentials, memory) lives under
# ~/.claude, which is CONTAINER DISK and is wiped on every pod stop -- so
# `claude --continue` would find nothing the next morning. Symlink it onto the
# volume rather than relying on CLAUDE_CONFIG_DIR: `bash pod_setup.sh` runs in a
# subshell, so its exports never reach the shell you launch claude from.
mkdir -p "$VOL/.claude"
if [ ! -L "$HOME/.claude" ]; then
  if [ -d "$HOME/.claude" ]; then
    cp -a "$HOME/.claude/." "$VOL/.claude/" 2>/dev/null || true
    rm -rf "$HOME/.claude"
  fi
  ln -s "$VOL/.claude" "$HOME/.claude"
fi

# Persist for future shells. Replace the block rather than skipping when the
# marker exists: the old guard was `if ! grep -q 'whitebear-setup'`, so a block
# written by an earlier version was never updated -- one had been carrying a
# stale $VOL/.hf_token line and no CLAUDE_CONFIG_DIR for days.
if [ -f ~/.bashrc ]; then
  sed -i '/# whitebear-setup BEGIN/,/# whitebear-setup END/d' ~/.bashrc
  sed -i '/^# whitebear-setup$/,+5d' ~/.bashrc          # pre-marker legacy block
fi
cat >> ~/.bashrc <<'BASHRC'
# whitebear-setup BEGIN
export HF_HOME=/workspace/hf
export HF_HUB_ENABLE_HF_TRANSFER=1
export PATH="$HOME/.local/bin:$PATH"
export CLAUDE_CONFIG_DIR=/workspace/.claude
if [ -f /workspace/hf/token ]; then export HF_TOKEN=$(cat /workspace/hf/token); fi
# whitebear-setup END
BASHRC

cat > ~/.tmux.conf <<'TMUX'
set -g set-clipboard on
set -ga terminal-features ',*:clipboard'
TMUX

# --- packages (container disk, so re-run every pod) ---
# orjson, httpx: the NLA path (irc/vendor/nla_inference.py, scripts/nla_judge.py).
# ipywidgets: progress bars in the notebook.
# pip prints an ERROR block about unsatisfied deps after --no-deps. That is its
# post-install consistency *check*, not a failure, and is expected.
pip install -q -U huggingface_hub hf_transfer sae_lens transformers pandas scipy \
    python-dotenv matplotlib tyro pyarrow ipywidgets httpx orjson
pip install -q -e "$REPO" --no-deps

# --- claude code ---
if ! command -v claude >/dev/null; then
  curl -fsSL https://claude.ai/install.sh | bash
fi

# --- git ---
git config --global user.name "Robyn"
git config --global user.email "robyn@matsprogram.org"
git config --global credential.helper "store --file=$VOL/.git-credentials"
# --add appends unconditionally, so this accumulated duplicates of a stale path
# ($VOL/whitebear, which does not exist) on every run. Set it idempotently.
git config --global --get-all safe.directory 2>/dev/null | grep -qx "$REPO" \
  || git config --global --add safe.directory "$REPO"

# --- verify ---
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv
python -c "import torch; print('torch', torch.__version__, '| cuda:', torch.cuda.is_available())"
python - <<'PYCHECK'
import importlib.util
missing = [m for m in ("dotenv", "transformers", "sae_lens", "pandas", "pyarrow",
                       "matplotlib", "tyro", "scipy", "orjson", "httpx")
           if not importlib.util.find_spec(m)]
print("missing packages:", missing or "none")
# `from irc import env` MUST come first: config.py reads WB_MODE from the
# environment at import time, and .env is what sets it. Importing config first
# silently yields the dev profile (gemma-3-4b-it) on a prod pod -- which the
# previous version of this check did, and reported as "config ok".
from irc import env  # noqa: F401
import config
print(f"profile: WB_MODE={config.MODE} model={config.MODEL_ID} layer={config.LAYER}")
PYCHECK
echo "HF_HOME=$HF_HOME  HF_TOKEN=${HF_TOKEN:+set}"
du -sh "$VOL/hf" 2>/dev/null
