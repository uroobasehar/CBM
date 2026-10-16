# RG-CBM offline on Windows 11 + RTX PRO 4500 Blackwell

This repository adapts the uploaded notebook to existing local datasets and provides a complete offline **Linux conda environment inside Windows WSL2**. The notebook's classifier remains TensorFlow/Keras EfficientNet-B1 with concept bottlenecks; Qwen generates text on CPU. No cloud account or API key is needed at runtime.

**Modern TensorFlow GPU training does not work in native Windows Anaconda.** TensorFlow 2.10 was the last native Windows GPU release and is unsuitable for this notebook and Blackwell. Install/use Ubuntu 22.04 or newer **WSL2** on Windows 11. The supplied Linux environment cannot be activated in Windows Anaconda Prompt. Anaconda/Miniforge inside WSL can use it as a prefix environment, but no additional conda installation is needed to extract and run the packed environment.

The target GPU is not present on the build host. CUDA 12.8/cuDNN 9.8 are selected for Blackwell, but the official TensorFlow wheel relies on PTX for newer GPUs. **The target GPU convolution, matrix multiplication and pretrained EfficientNet gradient checks must pass before calling the GPU environment ready.** XLA is an optional diagnostic because the notebook explicitly uses `jit_compile=False`. The scripts fail visibly if the target driver/runtime cannot execute them; they do not silently fall back to CPU.

## Which download to transfer

| Download | Contents | Sufficient for a disconnected PC? |
|---|---|---|
| `downloads/CBM-offline-setup.zip` or GitHub “Download ZIP” | Adapted notebook, settings, scripts, tests and instructions | **No**: no conda runtime or pretrained weights |
| Release assets `CBM-offline-full.zip.part0001`, `part0002`, … plus `SHA256SUMS` and `assemble.ps1` | Packed conda environment, all application dependencies, ImageNet EfficientNet-B1 weights, Qwen safetensors/tokenizer and source | **Yes for the application**, after Windows driver + WSL2 + Ubuntu prerequisites are installed and your local datasets are configured |

The [GitHub build workflow](https://github.com/uroobasehar/CBM/actions/workflows/build-offline.yml) runs after relevant pushes to `main` or via **Actions → Build Windows WSL offline bundle → Run workflow**. A successful build publishes the full payload under [Releases](https://github.com/uroobasehar/CBM/releases). A queued or failed build is **not** an available offline payload: check its log and release assets before disconnecting. Building uses internet; running the resulting payload does not. Datasets are deliberately not downloaded or included.

The full payload is several GB and split into 1,500 MiB parts to fit GitHub's per-release-asset limit. Binary environments/models do not belong in ordinary Git commits (GitHub limits individual Git files to 100 MB). The small setup ZIP is committed for convenience; full binaries are release assets.

## 1. Prepare Windows/WSL prerequisites before disconnecting

1. Install the current NVIDIA **Windows** production driver explicitly supporting the RTX PRO 4500 Blackwell and CUDA 12.8 or newer. Keep its offline installer on your USB drive. Do **not** install a Linux NVIDIA display driver inside WSL. The driver shown by `nvidia-smi` must actually support this GPU; its displayed CUDA version is driver capability, not proof that TensorFlow works.
2. Enable firmware CPU virtualization, Windows Subsystem for Linux and Virtual Machine Platform. Install/update WSL2 and Ubuntu 22.04 (x86-64) or newer. The simplest connected preparation is Administrator PowerShell: `wsl --install -d Ubuntu-22.04`, restart when requested, and complete Ubuntu's first launch. Verify `wsl --list --verbose` reports version **2**.
3. If the target **never** has internet, obtain the official WSL installer/kernel and NVIDIA driver on a connected PC. A connected Windows donor can prepare Ubuntu and export it with `wsl --export Ubuntu-22.04 X:\Transfer\ubuntu-22.04.tar`. On the target, install the official WSL package, enable the Windows features, restart, then import that transferred image: `wsl --import CBM-Ubuntu C:\WSL\CBM-Ubuntu X:\Transfer\ubuntu-22.04.tar --version 2`. Use `wsl -d CBM-Ubuntu` to enter it. Windows feature activation may need matching offline Windows installation media if local components are unavailable. The donor image is a separate OS prerequisite; **the application ZIP does not contain Ubuntu, WSL or a display driver**.
4. Have at least 32 GB RAM recommended (Qwen CPU weights and training/data use RAM) and plan at least **40 GB free WSL disk space**, in addition to datasets, checkpoints, the downloaded parts and assembled ZIP. Actual size is reported by the build. Use NTFS/exFAT transfer media; FAT32 cannot hold the assembled multi-GB ZIP.

Official references: [Microsoft offline WSL installation](https://learn.microsoft.com/windows/wsl/install), [WSL import/export](https://learn.microsoft.com/windows/wsl/basic-commands), [NVIDIA CUDA on WSL](https://docs.nvidia.com/cuda/wsl-user-guide/index.html), [TensorFlow pip / Windows support](https://www.tensorflow.org/install/pip), [CUDA 12.8 release notes](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-toolkit-release-notes/index.html).

## 2. Assemble and install the FULL payload offline

Download **all** release parts, `SHA256SUMS` and `assemble.ps1` on a connected machine. Copy them together to the offline PC. In Windows PowerShell, run:

```powershell
cd X:\Transfer\CBM
powershell -ExecutionPolicy Bypass -File .\assemble.ps1
```

The script verifies SHA256 for every part and streams them into `CBM-offline-full.zip` without loading the archive into RAM. It refuses to overwrite an existing ZIP. Extract it with a ZIP64-capable tool such as 7-Zip. Copy the extracted `CBM` directory into the WSL Linux filesystem, for example from Ubuntu:

```bash
cp -a /mnt/x/Transfer/CBM/extracted/CBM "$HOME/CBM"
cd "$HOME/CBM"
bash offline/install_offline.sh
```

Adjust the source path to your extraction directory. Keep the installed environment in the Linux filesystem, not `/mnt/c` or another Windows mount. Installation verifies the included payload checksums, extracts the Python/conda environment and runs `conda-unpack` and `pip check`. It performs no internet access or dependency resolution. Do not move the directory after `conda-unpack`; extract a fresh copy if relocating.

If you already have Anaconda/Miniforge **inside WSL**, you can activate the extracted prefix with `conda activate "$HOME/CBM/offline/.env"`; **also source `offline/activate.sh`** for CUDA library, compiler and offline flags. Use the supplied start script by default.

## 3. Point the notebook at your existing datasets

Copy the settings file and edit the local copy:

```bash
cp offline/settings.json offline/settings.local.json
nano offline/settings.local.json
```

Example (Windows drive paths are translated automatically inside WSL):

```json
{
  "datasets": ["brain_tumor", "pathmnist", "caltech101", "cub2002011"],
  "brain_tumor_root": "D:/Datasets/brain_tumor",
  "medmnist_root": "D:/Datasets/medmnist",
  "caltech101_root": "D:/Datasets/101_ObjectCategories",
  "caltech101_format": "folders",
  "tfds_root": "D:/Datasets/tensorflow_datasets",
  "cub_root": "D:/Datasets/CUB_200_2011",
  "results_root": "results",
  "text_backend": "local_qwen",
  "require_gpu": true
}
```

Use forward slashes in JSON Windows paths (or escape each backslash). Relative paths are relative to the extracted `CBM` root, not to an arbitrary shell directory. You can select only datasets you have, for example `"datasets": ["brain_tumor"]`. The datasets may live in a local repository or any accessible local folder; no GitHub access occurs. Extract archives first.

Required layouts:

```text
brain_tumor/
  Training/<class_name>/*.jpg
  Testing/<same_class_names>/*.jpg
medmnist/
  pathmnist.npz
101_ObjectCategories/
  accordion/*.jpg
  airplanes/*.jpg
  ...
CUB_200_2011/
  images/<class_name>/*.jpg
  images.txt
  image_class_labels.txt
  train_test_split.txt
  classes.txt
```

PathMNIST must be the official **28-pixel NPZ** with `train_images`, `train_labels`, `val_images`, `val_labels`, `test_images`, `test_labels`; do not rename arbitrary arrays or image folders to `pathmnist.npz`. The notebook uses the provided boundaries. DermAMNIST is also supported if selected and supplied as `dermamnist.npz`.

For Caltech, raw folders use sorted folders/files and `RandomState(1234)` to choose a 30-image/class train+validation pool; the remainder forms test. Validation is stratified with split seed 42. **This is a local split and is not claimed identical to TFDS's original partition.** The background category is excluded by default. To retain an existing TFDS partition, use `"caltech101_format": "tfds"` and point `tfds_root` at the parent of the prepared `caltech101/<version>/dataset_info.json` and TFRecord files. The loader reads the highest locally prepared numeric version directly, so it does not demand that the cache match the installed TFDS builder version. An unprepared raw archive is not a TFDS cache. No `download_and_prepare` is called.

MRI preserves supplied test images, checks conflicting byte-identical labels, and excludes duplicate train images; patient-level separation remains unknown. CUB requires the official 200 classes and 5,994/5,794 train/test counts. Missing/corrupt data fails visibly. Use a new campaign after changing datasets, splits or runtime versions; do not reuse a Colab optimizer checkpoint under a different TensorFlow version.

## 4. Verify and run

Inside Ubuntu WSL:

```bash
cd "$HOME/CBM"
nvidia-smi
bash offline/start.sh
```

The start script activates local CUDA libraries, sets Hugging Face/TFDS/pip offline flags, validates local paths, executes **GPU convolution/gradients and matrix multiplication**, reports a separate optional XLA diagnostic, loads and backpropagates through local EfficientNet weights, and generates Qwen tokens from local weights. It then starts JupyterLab, listening only on loopback, with its normal token authentication. Open the token-bearing address printed by Jupyter in your Windows browser and select **RG-CBM offline (WSL GPU)**. Open `RG_CBM_Cross_Domain_Caltech101_Ablation_(6).ipynb` and run cells top to bottom.

For validation alone:

```bash
source offline/activate.sh
python offline/validate.py
```

If a required GPU smoke check fails, keep its real error and fix the Windows driver/WSL/runtime compatibility before training. Setting `require_gpu` to false is an explicit CPU experiment; it does not validate the requested GPU workflow. First PTX compilation on Blackwell can take longer. TensorFlow stays float32 as in the notebook; CPU Qwen avoids GPU memory competition. Lower the notebook's batch size only if actual GPU memory exhaustion occurs and use a new campaign for changed scientific settings.

Outputs and resumable checkpoints go under the configured `results_root`. Choose a new `CAMPAIGN_NAME` for a new experiment; retain it and other controls to resume. Qwen failures retain labeled template fallbacks in each case's metadata. For intentionally deterministic non-LLM explanations, choose `"text_backend": "template"` in settings.

## Connected builder (alternative to GitHub Releases)

On a **connected** Linux x86-64 / Ubuntu WSL2 machine with sufficient disk space:

```bash
git clone https://github.com/uroobasehar/CBM.git
cd CBM
bash offline/build_online.sh
```

The builder uses an existing conda installation or checksum-verifies and installs a pinned Miniforge. It creates a Python 3.12 environment, installs pinned TensorFlow 2.21.0/Keras 3.15.1 and CUDA 12.8/cuDNN 9.8 packages, CPU-only PyTorch 2.7.1/torchvision, Transformers 4.57.6, notebook tools and dataset libraries. It records transitive dependency versions and conda URLs. Those packages are packed into the payload, so the offline PC does not need a wheelhouse, conda channel or package cache. Nothing attempts to restore online versions when the notebook runs.

Keras downloads ImageNet EfficientNet-B1 weights and verifies its published hash. At runtime the loader verifies the same local cache/hash before selecting Keras ImageNet mode; this preserves its extra ImageNet-specific rescaling layer. Passing only a local weight filename would incorrectly omit that preprocessing layer. Hugging Face downloads the Qwen safetensors/tokenizer at a resolved commit; `offline/build-info.json` records that commit. The builder performs local model-loading/generation checks in offline mode, synthetic two-stage training/testing/report/restart checks and real EfficientNet random-weight CPU backpropagation, then packs the environment and verifies extraction, `conda-unpack` and local model loading in a second prefix before publishing. All payload files get SHA256 records; transfers get per-part SHA256 records. These checks detect corruption, not authenticity of a bundle from an untrusted source; obtain releases from this repository and retain model/dependency licenses shipped with the assets.

Network is needed **only on the builder**: GitHub/its release CDN, conda-forge, PyPI/Python file hosts, PyTorch CPU wheel hosts, `storage.googleapis.com` (Keras weights), `huggingface.co` and the Hugging Face/Xet model-file CDN. No Kaggle/Caltech/MedMNIST data downloads are performed. If a corporate proxy blocks a model host, fix that policy; the builder stops rather than creating a supposedly complete package without weights.

## Validation scope

The development checks exercise local path conversion, disjoint/reproducible Caltech splits, missing-data failures, split ZIP reassembly and absence of notebook download/install calls. The synthetic integration runs a two-class local MRI fixture through head/fine-tune training, saved-model restart, all test predictions, audits and a portable HTML report. The actual EfficientNet random-weight gradient path is checked separately.

Real dataset accuracy, actual Windows/WSL installation and **RTX PRO 4500 Blackwell execution are not established by these CPU/build-host checks**. The packaged target smoke test supplies that final evidence. Review the successful release/build logs before relying on a full payload; source-only delivery is not a completed offline installation.
