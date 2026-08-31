#!/usr/bin/env bash
# Queue the held-out generation behind the pilot2 teacher-forced pass and the
# readouts build (concept vectors + SAE latents).
#
# A script file rather than pasted lines: bash executes a pasted compound
# command before reading further input, so the rest of a multi-line paste sits
# in the terminal's input buffer for however long the first loop runs -- hours,
# here. Anything that disturbs the tty loses it.

set -u
cd /workspace/hack-internal-representation-control

echo "==> queue_heldout starting $(date '+%F %T %Z')"

# 1. Wait for the readouts build to produce its 50 latent files (one .json per
#    concept). Bounded at 4h so a hang there does not cost the night -- the
#    held-out generation does not need the latents, only the later measure does.
echo "    waiting for artifacts/latents_v2 (50 files, 4h bound)..."
for _ in $(seq 240); do
  [ "$(ls artifacts/latents_v2/*.json 2>/dev/null | wc -l)" -ge 50 ] && break
  sleep 60
done
echo "    latents present: $(ls artifacts/latents_v2/*.json 2>/dev/null | wc -l)/50"

# 2. Then wait for the GPU to be genuinely free. If the bound above fired while
#    something was still running, this waits rather than letting the run hit the
#    VRAM guard and refuse -- which would mean no held-out run at all.
echo "    waiting for a free GPU..."
while nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q . \
   || pgrep -f "run_pipeline.py|03_run_generation.py" > /dev/null; do
  sleep 60
done

echo "==> starting held-out generation $(date '+%F %T %Z')"
python3 rk_scripts/03_run_generation.py --run-id heldout1 --split held_out
echo "==> queue_heldout done $(date '+%F %T %Z')"
