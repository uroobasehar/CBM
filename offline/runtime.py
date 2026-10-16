"""Offline paths and raw-folder Caltech adapter. No network operations."""
import json
import os
from pathlib import Path
import re


def local_path(value, home):
    value = os.path.expandvars(str(value))
    match = re.match(r"^([A-Za-z]):[\\/](.*)$", value)
    if match and os.name != "nt":
        value = f"/mnt/{match[1].lower()}/" + match[2].replace("\\", "/")
    path = Path(value).expanduser()
    return path if path.is_absolute() else home / path


def configure(home):
    home = Path(home).resolve()
    config_path = home / "offline/settings.local.json"
    if not config_path.exists():
        config_path = home / "offline/settings.json"
    settings = json.loads(config_path.read_text(encoding="utf-8-sig"))
    for key in ("brain_tumor_root", "medmnist_root", "caltech101_root", "tfds_root",
                "cub_root", "results_root"):
        settings[key] = str(local_path(settings[key], home))
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE",
                 "TFDS_DISABLE_GCS"):
        os.environ[name] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["HF_HOME"] = str(home / "offline/models/huggingface-cache")
    os.environ["KERAS_HOME"] = str(home / "offline/models/keras")
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    # Also covers accidentally invoked pip commands; do not restore online versions.
    os.environ["PIP_NO_INDEX"] = "1"
    settings["qwen_root"] = str(home / "offline/models/Qwen2.5-1.5B-Instruct")
    settings["efficientnet_weights"] = str(home / "offline/models/keras/models/efficientnetb1_notop.h5")
    return settings


def preflight(settings, require_models=True):
    errors = []
    required = []
    selected = settings["datasets"]
    unknown = set(selected) - {"brain_tumor", "pathmnist", "dermamnist", "caltech101", "cub2002011"}
    if unknown or not selected:
        errors.append(f"Unsupported/empty dataset selection: {selected}")
    if "brain_tumor" in selected:
        root = Path(settings["brain_tumor_root"])
        for split in ("Training", "Testing"):
            if not any(p.is_dir() and p.name.lower() == split.lower() for p in [root, *root.glob("**/*")]):
                errors.append(f"MRI: missing {split}/class_name/images under {root}")
    for name in ("pathmnist", "dermamnist"):
        if name in selected:
            required.append(Path(settings["medmnist_root"]) / f"{name}.npz")
    if "cub2002011" in selected:
        root = Path(settings["cub_root"])
        required += [root / p for p in ("images", "images.txt", "classes.txt", "image_class_labels.txt", "train_test_split.txt")]
    if "caltech101" in selected:
        if settings["caltech101_format"] == "folders":
            root = Path(settings["caltech101_root"])
            if not root.is_dir() or not any(p.is_dir() for p in root.iterdir()):
                errors.append(f"Caltech: expected class folders directly under {root}")
        elif settings["caltech101_format"] == "tfds":
            root = Path(settings["tfds_root"]) / "caltech101"
            if not any(root.glob("*/dataset_info.json")):
                errors.append(f"Missing prepared TFDS Caltech dataset under {root}")
        else:
            errors.append("caltech101_format must be folders or tfds")
    if require_models:
        required.append(Path(settings["efficientnet_weights"]))
        if settings["text_backend"] == "local_qwen":
            root = Path(settings["qwen_root"])
            required += [root / "config.json", root / "tokenizer_config.json"]
            if not list(root.glob("*.safetensors")):
                errors.append(f"Missing Qwen safetensors weights under {root}")
    errors += [f"Missing local file/directory: {p}" for p in required if not p.exists()]
    if errors:
        raise FileNotFoundError("Offline prerequisites missing (no download attempted):\n" + "\n".join(errors))


def caltech_folder_frames(root, validation_fraction, split_seed, include_background=False):
    """Deterministic local 30/class training pool; not TFDS's filesystem-order split."""
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split
    root = Path(root)
    folders = sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")
                     and (include_background or "background" not in p.name.lower()))
    if not folders:
        raise ValueError(f"No class folders in {root}")
    rng = np.random.RandomState(1234)
    pool, heldout = [], []
    for label, folder in enumerate(folders):
        files = sorted(p for p in folder.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        if len(files) <= 30:
            raise ValueError(f"Caltech class {folder.name} needs >30 images, found {len(files)}")
        chosen = set(rng.choice(len(files), 30, replace=False).tolist())
        for index, path in enumerate(files):
            row = dict(id="caltech101:" + str(path.relative_to(root)), path=str(path), label=label)
            (pool if index in chosen else heldout).append(row)
    frame = pd.DataFrame(pool)
    a, b = train_test_split(np.arange(len(frame)), test_size=validation_fraction,
                            random_state=split_seed, stratify=frame.label)
    parts = [frame.iloc[np.sort(a)].copy(), frame.iloc[np.sort(b)].copy(), pd.DataFrame(heldout)]
    for part, split in zip(parts, ("train", "val", "test")):
        part["split"] = split
    return parts, [p.name for p in folders]
