#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ "$(uname -s)" != Linux || "$(uname -m)" != x86_64 ]]; then
    echo 'Run this in Ubuntu WSL2 x86-64, not Windows Anaconda Prompt.' >&2; exit 1
fi
if [[ "$PWD" == /mnt/* ]]; then
    echo 'Copy the extracted bundle into the WSL Linux filesystem (e.g. ~/CBM) first.' >&2; exit 1
fi
if [[ ! -f offline/environment.tar.gz ]]; then
    echo 'This is the source/setup ZIP, not the full offline payload. See README.md.' >&2; exit 1
fi
python3 offline/package.py --verify
if [[ -e offline/.env ]]; then
    echo 'offline/.env already exists. Activate it; do not extract over an existing environment.' >&2; exit 1
fi
mkdir offline/.env
tar -xzf offline/environment.tar.gz -C offline/.env
source offline/.env/bin/activate
conda-unpack
python -m pip check
echo 'Environment extracted. Do not move this folder after conda-unpack.'
echo 'Next: edit offline/settings.local.json, then bash offline/start.sh'
