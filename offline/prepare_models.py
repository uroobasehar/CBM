"""Connected build step only; downloads and records the exact pretrained assets."""
import hashlib
import json
import os
from pathlib import Path

HOME = Path(__file__).resolve().parents[1]
MODELS = HOME / "offline/models"
os.environ["KERAS_HOME"] = str(MODELS / "keras")
os.environ["HF_HOME"] = str(HOME / ".offline-build/huggingface")
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def main():
    import tensorflow as tf
    from huggingface_hub import HfApi, snapshot_download
    # Keras verifies its published hash rather than accepting arbitrary weights.
    model = tf.keras.applications.EfficientNetB1(include_top=False, weights="imagenet",
                                               input_shape=(224, 224, 3))
    del model
    repo = "Qwen/Qwen2.5-1.5B-Instruct"
    revision = HfApi().model_info(repo).sha
    target = MODELS / "Qwen2.5-1.5B-Instruct"
    snapshot_download(repo, revision=revision, local_dir=target,
                      allow_patterns=["*.json", "*.safetensors", "*.txt", "*.tiktoken",
                                      "*.model", "*.jinja", "README.md", "LICENSE*"])
    info = dict(tensorflow=tf.__version__, qwen_repository=repo, qwen_revision=revision,
                efficientnet="Keras EfficientNetB1 ImageNet, include_top=False",
                gpu_validation="NOT RUN: target Windows/WSL Blackwell hardware required")
    (HOME / "offline/build-info.json").write_text(json.dumps(info, indent=2) + "\n")


if __name__ == "__main__":
    main()
