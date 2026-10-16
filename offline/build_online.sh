#!/usr/bin/env bash
# Run ONLY on a connected Ubuntu 22.04+ x86-64 machine or connected WSL2.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$PWD"
BUILD="$ROOT/.offline-build"
mkdir -p "$BUILD" "$ROOT/offline/models"
if [[ "$(uname -s)" != Linux || "$(uname -m)" != x86_64 ]]; then
    echo 'Build inside Linux x86-64 / WSL2, not Windows Anaconda Prompt.' >&2; exit 1
fi
if [[ -n "${CONDA_EXE:-}" && -x "$CONDA_EXE" ]]; then
    CONDA_BIN="$CONDA_EXE"
elif command -v conda >/dev/null; then
    CONDA_BIN="$(command -v conda)"
else
    INSTALLER="Miniforge3-26.7.2-0-Linux-x86_64.sh"
    BASE_URL="https://github.com/conda-forge/miniforge/releases/download/26.7.2-0"
    curl --fail --location --retry 3 "$BASE_URL/$INSTALLER" -o "$BUILD/$INSTALLER"
    curl --fail --location --retry 3 "$BASE_URL/$INSTALLER.sha256" -o "$BUILD/$INSTALLER.sha256"
    (cd "$BUILD" && sha256sum --check "$INSTALLER.sha256")
    if [[ ! -x "$BUILD/miniforge/bin/conda" ]]; then
        bash "$BUILD/$INSTALLER" -b -p "$BUILD/miniforge"
    fi
    CONDA_BIN="$BUILD/miniforge/bin/conda"
fi
ENV_PREFIX="$BUILD/rg-cbm"
if [[ ! -x "$ENV_PREFIX/bin/python" ]]; then
    "$CONDA_BIN" create --yes --override-channels -c conda-forge -p "$ENV_PREFIX" python=3.12 pip
fi
PY="$ENV_PREFIX/bin/python"
unset PIP_NO_INDEX HF_HUB_OFFLINE TRANSFORMERS_OFFLINE HF_DATASETS_OFFLINE
"$PY" -m pip install --no-cache-dir torch==2.7.1+cpu torchvision==0.22.1+cpu --index-url https://download.pytorch.org/whl/cpu
"$PY" -m pip install --no-cache-dir -r offline/requirements.txt
"$PY" -m pip check
"$PY" -m pip freeze > offline/resolved-requirements.txt
"$CONDA_BIN" list --explicit -p "$ENV_PREFIX" > offline/conda-explicit.txt
"$PY" offline/prepare_models.py
# These checks deliberately do NOT claim to validate the target GPU.
"$PY" offline/validate.py --models-only
"$PY" -m unittest discover -s tests -v
"$PY" offline/test_adapters.py
"$PY" offline/test_notebook.py
"$ENV_PREFIX/bin/conda-pack" -p "$ENV_PREFIX" -o offline/environment.tar.gz --force
# Exercise relocation before publishing; the target does not have the build prefix.
VERIFY_PREFIX="$BUILD/relocation-check"
if [[ -e "$VERIFY_PREFIX" ]]; then
    echo 'Move/remove the prior .offline-build/relocation-check before a repeat build.' >&2; exit 1
fi
mkdir "$VERIFY_PREFIX"
tar -xzf offline/environment.tar.gz -C "$VERIFY_PREFIX"
(
    source "$VERIFY_PREFIX/bin/activate"
    conda-unpack
    python -m pip check
    python offline/validate.py --models-only
)
"$PY" offline/package.py --full
echo 'Full payload built. Transfer every CBM-offline-full.zip.part* file, SHA256SUMS, and assemble.ps1.'
