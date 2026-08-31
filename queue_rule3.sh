#!/usr/bin/env bash
# Queue pooling rule 3 (08_plausible_positions.py) on the pilot, behind the
# held-out generation.
#
# Why this is worth the GPU time: it decides whether rule 3 is ever needed on
# HELD-OUT, which is ~8 hours at 18,487 trials. If token_mean beats plausible on
# the pilot, that 8 hours never happens. Two hours here to resolve eight there.
#
# A script rather than pasted lines -- bash executes a pasted compound command
# before reading further input, so the rest of a multi-line paste sits in the
# terminal buffer for however long the first loop runs.

set -u
cd /workspace/hack-internal-representation-control

echo "==> queue_rule3 starting $(date '+%F %T %Z')"

# Wait for the held-out generation to record its completion. Waiting on the
# record rather than on the process: a completion record only appears when the
# run actually finished, so there is no gap between one process exiting and the
# next appearing for a poll to fall into.
echo "    waiting for heldout1 to complete..."
until grep -q '"event": "completed"' \
      artifacts/runs/heldout1/generated/invocations.jsonl 2>/dev/null; do
  sleep 60
done
echo "    heldout1 complete $(date '+%F %T %Z')"

# Cheap and informative: the held-out deviation rate, in seconds, no GPU.
# First test of whether the tagged scaffold's 0.04% holds on 40 unseen concepts.
python3 rk_scripts/10_deviant_stimuli.py --run-id heldout1 || true

# Belt and braces: confirm the GPU is actually free before claiming it.
while nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q .; do
  sleep 30
done

echo "==> starting rule 3 on the pilot $(date '+%F %T %Z')"
# exec so tmux names the pane after python rather than after this script.
exec python3 rk_scripts/08_plausible_positions.py --run-id pilot2 --pass generated
