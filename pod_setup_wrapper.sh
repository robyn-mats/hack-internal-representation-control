#!/usr/bin/env bash
# Thin wrapper. The real script is version controlled in the repo; edit it there.
# Previous standalone copy kept at /workspace/setup.sh.bak
#
# NEVER exec here. This file is normally SOURCED (`. /workspace/setup.sh`) so the
# exports land in the calling shell -- and exec replaces that shell, so on an ssh
# login the connection closes as soon as the script finishes.
export POD_SETUP_VIA="${BASH_SOURCE[0]}"
_pod_setup_real=/workspace/hack-internal-representation-control/pod_setup.sh
if [ "${BASH_SOURCE[0]}" != "$0" ]; then
  . "$_pod_setup_real"                # sourced: exports reach the caller
else
  bash "$_pod_setup_real" "$@"        # executed: ordinary subprocess
fi
# Don't leave the wrapper's own bookkeeping in an interactive shell.
unset _pod_setup_real POD_SETUP_VIA
