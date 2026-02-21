#!/bin/bash

# Step 3: Full setup (env + datasets/links)
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "🧩 Running full setup: env + datasets/links"

bash "$script_dir/01_setup_env.sh"
bash "$script_dir/02_datasets_and_links.sh" "$@"

echo "🎉 Full setup complete."
