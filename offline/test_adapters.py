"""Synthetic local data fixtures for every adapter; never download real datasets."""
import json
import os
from pathlib import Path
import sys
import tempfile
import types

os.environ.update(CUDA_VISIBLE_DEVICES="-1", TF_CPP_MIN_LOG_LEVEL="3", MPLBACKEND="Agg",
                  TF_NUM_INTRAOP_THREADS="2", TF_NUM_INTEROP_THREADS="2")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from offline.runtime import configure, caltech_folder_frames


def main():
    settings = configure(ROOT)
    settings["require_gpu"] = False
    settings["caltech101_format"] = "folders"
    import numpy as np
    import tensorflow_datasets as tfds
    from PIL import Image
    tfds.disable_progress_bar()
    notebook = json.loads(next(ROOT.glob("*.ipynb")).read_text())
    ns = dict(RUN_MODE="auto", OFFLINE_SETTINGS=settings, caltech_folder_frames=caltech_folder_frames)
    for index in (11, 13):
        exec(compile("".join(notebook["cells"][index]["source"]), f"cell-{index}", "exec"), ns)
    rng = np.random.default_rng(22)
    with tempfile.TemporaryDirectory(prefix="cbm-adapters-") as directory:
        root = Path(directory)
        med = root / "med"
        med.mkdir()
        settings["medmnist_root"] = str(med)
        arrays = {}
        for split in ("train", "val", "test"):
            arrays[split + "_images"] = rng.integers(0, 256, (18, 28, 28, 3), dtype=np.uint8)
            arrays[split + "_labels"] = np.tile(np.arange(9, dtype=np.uint8), 2)[:, None]
        np.savez(med / "pathmnist.npz", **arrays)
        out = root / "pathout"
        out.mkdir()
        cfg = ns["config_for"]("pathmnist", overrides=dict(image_size=32, batch_size=2))
        bundle = ns["load_data"](cfg, out)
        assert len(bundle.manifest) == 54 and len(bundle.names) == 9
        print("PASS: real MedMNIST local NPZ loader, nine classes and three supplied splits")
        cal = root / "cal"
        cal.mkdir()
        settings["caltech101_root"] = str(cal)
        for name in ("birds", "faces"):
            folder = cal / name
            folder.mkdir()
            for index in range(40):
                Image.fromarray(rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)).save(folder / f"{index:04d}.jpg")
        out = root / "calout"
        out.mkdir()
        cfg = ns["config_for"]("caltech101", overrides=dict(image_size=32, batch_size=2))
        bundle = ns["load_data"](cfg, out)
        assert len(bundle.manifest) == 80 and len(bundle.names) == 2
        print("PASS: raw Caltech folders through image decoding and split validation")
        cub = root / "cub"
        (cub / "images").mkdir(parents=True)
        tables = {"images.txt": [], "image_class_labels.txt": [], "train_test_split.txt": []}
        for index in range(1, 9):
            name = f"{index}.jpg"
            Image.fromarray(rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)).save(cub / "images" / name)
            tables["images.txt"].append(f"{index} {name}")
            tables["image_class_labels.txt"].append(f"{index} {1 + index % 2}")
            tables["train_test_split.txt"].append(f"{index} {int(index <= 4)}")
        tables["classes.txt"] = ["1 alpha", "2 beta"]
        for name, lines in tables.items():
            (cub / name).write_text("\n".join(lines) + "\n")
        frame, names = ns["read_cub_metadata"](cub, expected_counts=(4, 4), expected_classes=2)
        assert len(frame) == 8 and len(names) == 2
        print("PASS: CUB metadata joins, path/hash checks and fixed train/test boundaries")
        # Create TFRecords from generated arrays without calling a downloader.
        settings["tfds_root"] = str(root / "tfds")
        settings["caltech101_format"] = "tfds"
        builder = tfds.builder("caltech101", data_dir=settings["tfds_root"], try_gcs=False)
        def examples(count, split):
            for label in range(builder.info.features["label"].num_classes):
                for index in range(count):
                    yield f"{split}/{label}/{index}", dict(image=np.full((8, 8, 3), 128, np.uint8),
                        label=label, **{"image/file_name": f"{split}_{index}.jpg"})
        def splits(self, manager):
            return dict(train=examples(10, "train"), test=examples(2, "test"))
        builder._split_generators = types.MethodType(splits, builder)
        builder.download_and_prepare(download_config=tfds.download.DownloadConfig(try_download_gcs=False))
        out = root / "tfdsout"
        out.mkdir()
        bundle = ns["load_data"](cfg, out)
        assert len(bundle.names) == 101 and len(bundle.manifest) == 1212
        print("PASS: prepared TFDS Caltech tree; 101 non-background classes and 1212 fixture records")


if __name__ == "__main__":
    main()
