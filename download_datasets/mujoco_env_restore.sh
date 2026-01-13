#!/usr/bin/env bash
set -euo pipefail

backup_file="${MUJOCO_ENV_BACKUP:-$HOME/.mujoco/mujoco_env_backup}"

if [[ ! -f "$backup_file" ]]; then
  echo "No backup file found at $backup_file"
  echo "Nothing to restore."
  exit 1
fi

# shellcheck disable=SC1090
source "$backup_file"

echo "Restored environment from: $backup_file"
