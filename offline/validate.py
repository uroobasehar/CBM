"""Fail fast on missing models, dataset roots, or actual GPU execution failures."""
import argparse
import os
from pathlib import Path
import sys
import subprocess

HOME = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HOME))
from offline.runtime import configure, preflight, efficientnet_b1


def validate_models(settings, require_gpu=False):
    import tensorflow as tf
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    weights = Path(settings["efficientnet_weights"])
    if not weights.is_file():
        raise FileNotFoundError(weights)
    backbone = efficientnet_b1(tf, "imagenet", 64, settings)
    if sum(isinstance(layer, tf.keras.layers.Rescaling) for layer in backbone.layers) != 2:
        raise RuntimeError("EfficientNet ImageNet preprocessing graph differs from the expected two rescaling layers")
    if require_gpu:
        with tf.device("/GPU:0"), tf.GradientTape() as tape:
            output = backbone(tf.ones((2, 64, 64, 3)) * 128, training=False)
            loss = tf.reduce_mean(output)
        gradients = tape.gradient(loss, backbone.trainable_variables)
        if any(gradient is None for gradient in gradients):
            raise RuntimeError("EfficientNet GPU gradient path is disconnected")
        for gradient in gradients:
            tf.debugging.assert_all_finite(gradient, "Nonfinite EfficientNet GPU gradient")
        print("PASS: pretrained EfficientNet GPU forward/backpropagation")
    else:
        output = backbone(tf.zeros((1, 64, 64, 3)), training=False)
    tf.debugging.assert_all_finite(output, "EfficientNet forward pass failed")
    if settings["text_backend"] == "local_qwen":
        path = settings["qwen_root"]
        tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        llm = AutoModelForCausalLM.from_pretrained(path, local_files_only=True, torch_dtype=torch.float32).eval()
        inputs = tokenizer("Explain this classification:", return_tensors="pt")
        with torch.inference_mode():
            generated = llm.generate(**inputs, max_new_tokens=2, do_sample=False,
                                     pad_token_id=tokenizer.eos_token_id)
        if generated.shape[1] <= inputs["input_ids"].shape[1]:
            raise RuntimeError("Qwen did not generate tokens")
    print("PASS: local EfficientNet weights and selected text backend loaded without downloads")


def validate_gpu():
    import tensorflow as tf
    devices = tf.config.list_physical_devices("GPU")
    if not devices:
        raise RuntimeError("No TensorFlow GPU: check WSL2, Windows NVIDIA driver and offline/activate.sh")
    for device in devices:
        tf.config.experimental.set_memory_growth(device, True)
    tf.config.set_soft_device_placement(False)
    # Required: CUDA/cuDNN gradients and cuBLAS execution on the selected GPU.
    with tf.device("/GPU:0"):
        layer = tf.keras.layers.Conv2D(8, 3)
        x = tf.random.normal((2, 32, 32, 3))
        with tf.GradientTape() as tape:
            loss = tf.reduce_mean(layer(x) ** 2)
        gradients = tape.gradient(loss, layer.trainable_variables)
        for gradient in gradients:
            tf.debugging.assert_all_finite(gradient, "GPU gradient is not finite")
        a = tf.ones((32, 32))
        result = tf.linalg.matmul(a, a) + 1
        tf.debugging.assert_near(result, tf.fill((32, 32), 33.0))
        if "GPU" not in result.device:
            raise RuntimeError("GPU smoke test fell back to CPU")
    print("PASS: GPU convolution/backpropagation and cuBLAS matmul", devices)


def validate_xla():
    import tensorflow as tf
    if not tf.config.list_physical_devices("GPU"):
        raise RuntimeError("No GPU for optional XLA diagnostic")
    for device in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(device, True)
    tf.config.set_soft_device_placement(False)
    @tf.function(jit_compile=True)
    def compiled(a):
        return tf.linalg.matmul(a, a) + 1
    with tf.device("/GPU:0"):
        result = compiled(tf.ones((32, 32)))
        tf.debugging.assert_near(result, tf.fill((32, 32), 33.0))
    print("PASS: optional XLA diagnostic")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models-only", action="store_true", help="Build host: do not require datasets/GPU")
    parser.add_argument("--xla-only", action="store_true", help="Optional GPU diagnostic, isolated from required validation")
    args = parser.parse_args()
    settings = configure(HOME)
    if args.xla_only:
        validate_xla()
        return
    if args.models_only:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    else:
        preflight(settings)
        if settings["require_gpu"]:
            validate_gpu()
    validate_models(settings, require_gpu=not args.models_only and settings["require_gpu"])
    if not args.models_only and settings["require_gpu"]:
        # Compiler failures can abort a process; keep this optional check isolated.
        try:
            result = subprocess.run([sys.executable, __file__, "--xla-only"], capture_output=True,
                                    text=True, timeout=180)
            if result.returncode == 0:
                print("PASS: optional XLA diagnostic")
            else:
                print("OPTIONAL XLA DIAGNOSTIC FAILED (notebook uses jit_compile=False):",
                      result.stderr[-1600:])
        except subprocess.TimeoutExpired:
            print("OPTIONAL XLA DIAGNOSTIC TIMED OUT (notebook uses jit_compile=False)")
    print("Validation completed", "(GPU/datasets NOT tested)" if args.models_only else "")


if __name__ == "__main__":
    main()
