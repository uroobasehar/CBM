#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source offline/activate.sh
python offline/validate.py
# --sys-prefix avoids embedding the build-machine interpreter in the kernel spec.
python -m ipykernel install --sys-prefix --name rg-cbm --display-name 'RG-CBM offline (WSL GPU)'
exec python -m jupyterlab --no-browser --ip=127.0.0.1 --notebook-dir="$CBM_HOME"
