#!/bin/bash
# Local macOS counterpart to vae3d/view_train_t1.bat for G0 source inspection.
set -euo pipefail
cd "$(dirname "$0")"
exec python3 -m experiments.vae_sdf.view_meshes "$@"
