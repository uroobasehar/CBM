"""CPU synthetic integration: local adapter -> train -> checkpoint -> test/audit/report.

No user datasets, pretrained downloads or target GPU are used by this test.
"""
import json
import os
from pathlib import Path
import sys
import tempfile

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["MPLBACKEND"] = "Agg"
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["TF_NUM_INTRAOP_THREADS"] = "2"
os.environ["TF_NUM_INTEROP_THREADS"] = "2"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from offline.runtime import configure, caltech_folder_frames


def main():
    import numpy as np
    import tensorflow as tf
    import nbformat
    from PIL import Image
    notebook = nbformat.read(next(ROOT.glob("*.ipynb")), as_version=4)
    nbformat.validate(notebook)
    settings = configure(ROOT)
    settings["require_gpu"] = False
    ns = dict(RUN_MODE="auto", OFFLINE_SETTINGS=settings,
              caltech_folder_frames=caltech_folder_frames)
    for index in (11, 13, 15, 17, 19, 21, 23, 25, 27):
        exec(compile(notebook.cells[index].source, f"notebook-cell-{index}", "exec"), ns)
    rng = np.random.default_rng(42)
    with tempfile.TemporaryDirectory(prefix="cbm-integration-") as directory:
        root = Path(directory)
        mri = root / "mri"
        for split, count in (("Training", 12), ("Testing", 4)):
            for label in ("class_a", "class_b"):
                target = mri / split / label
                target.mkdir(parents=True)
                for index in range(count):
                    Image.fromarray(rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)).save(target / f"{index}.png")
        cfg = ns["config_for"]("brain_tumor", 42, dict(brain_root=str(mri), weights=None,
                    image_size=32, batch_size=4, concepts=16, prototype_dim=8,
                    epochs_head=1, epochs_finetune=1, fine_tune_last_n=2,
                    warmup_epochs=0, ramp_epochs=0, audit_samples=2, random_deletions=1))
        def tiny_backbone(config):
            inputs = tf.keras.Input((config.image_size, config.image_size, 3))
            x = tf.keras.layers.Rescaling(1 / 255)(inputs)
            x = tf.keras.layers.Conv2D(8, 3, strides=4, activation="relu", name="tiny_conv")(x)
            return tf.keras.Model(inputs, x, name="synthetic_backbone")
        output = root / "run"
        status = ns["run_one"](cfg, output, backbone_factory=tiny_backbone, verbose=0,
            visuals=dict(show=False, preview_pages=0, cases=1, live_every=0),
            reports=dict(backend="template", fallback="template"))
        if status["status"] != "completed":
            raise AssertionError(status)
        for name in ("final.weights.h5", "test_metrics.csv", "split_manifest.csv",
                     "visual_report.html", "explanation_summary.json", "training_complete.json"):
            if not (output / name).is_file():
                raise AssertionError(f"Missing synthetic output: {name}")
        if status["n_test"] != 8:
            raise AssertionError(f"Expected all 8 test examples, got {status['n_test']}")
        # Reuse the final saved model through the real restart path, no new training.
        bundle = ns["load_data"](cfg, output)
        recovered, _ = ns["fit_model"](cfg, bundle, output, verbose=0)
        metrics, _ = ns["evaluate_model"](recovered, bundle, cfg, output)
        if metrics["n_test"] != 8:
            raise AssertionError("Restart omitted test examples")
        print("PASS: local MRI adapter, two-stage synthetic training, checkpoints, restart, all 8 test predictions, audits and HTML report")
    # Exercise the actual EfficientNet gradient path with random weights; no network.
    tf.keras.backend.clear_session()
    backbone = tf.keras.applications.EfficientNetB1(include_top=False, weights=None,
                                                   input_shape=(64, 64, 3))
    with tf.GradientTape() as tape:
        loss = tf.reduce_sum(backbone(tf.ones((2, 64, 64, 3)) * 128, training=False))
    gradients = tape.gradient(loss, backbone.trainable_variables)
    if not gradients or any(g is None for g in gradients):
        raise AssertionError("EfficientNet gradient path disconnected")
    for gradient in gradients:
        tf.debugging.assert_all_finite(gradient, "Nonfinite EfficientNet gradient")
    print("PASS: real EfficientNet-B1 random-weight CPU forward/backpropagation; pretrained/GPU checks remain separate")


if __name__ == "__main__":
    main()
