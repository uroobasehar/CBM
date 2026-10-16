#!/usr/bin/env bash
# Source this file. Never install a Linux NVIDIA display driver in WSL.
CBM_HOME="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export CBM_HOME
if [[ ! -x "$CBM_HOME/offline/.env/bin/python" ]]; then
    echo 'Run bash offline/install_offline.sh first.' >&2; return 1
fi
source "$CBM_HOME/offline/activate_prefix.sh" "$CBM_HOME/offline/.env"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1 TFDS_DISABLE_GCS=1 PIP_NO_INDEX=1
export KERAS_HOME="$CBM_HOME/offline/models/keras"
export HF_HOME="$CBM_HOME/offline/models/huggingface-cache"
# TensorFlow wheels need the pip-provided CUDA libraries and Blackwell-capable ptxas.
CBM_SITE="$CBM_HOME/offline/.env/lib/python3.12/site-packages"
CBM_LIBS="$(find "$CBM_SITE/nvidia" -type d -name lib -print | paste -sd: -)"
export LD_LIBRARY_PATH="$CBM_LIBS:/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PATH="$CBM_SITE/nvidia/cuda_nvcc/bin:$PATH"
export XLA_FLAGS="--xla_gpu_cuda_data_dir=$CBM_SITE/nvidia/cuda_nvcc${XLA_FLAGS:+ $XLA_FLAGS}"
unset CBM_SITE CBM_LIBS
