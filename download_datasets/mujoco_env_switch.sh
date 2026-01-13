#!/usr/bin/env bash
set -euo pipefail

# Save current env so we can restore later.
backup_file="${MUJOCO_ENV_BACKUP:-$HOME/.mujoco/mujoco_env_backup}"
mkdir -p "$(dirname "$backup_file")"

{
  echo "export MUJOCO_PY_MUJOCO_PATH=\"${MUJOCO_PY_MUJOCO_PATH-}\""
  echo "export LD_LIBRARY_PATH=\"${LD_LIBRARY_PATH-}\""
} > "$backup_file"

# Switch to MuJoCo 2.1.0 for mujoco_py.
export MUJOCO_PY_MUJOCO_PATH="$HOME/.mujoco/mujoco210"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH-}:$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia"

echo "Switched to MuJoCo 2.1.0."
echo "Backup saved to: $backup_file"
echo "Use: source minimal_wm/mujoco_env_restore.sh"
