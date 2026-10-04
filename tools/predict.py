"""Predict one image or an image directory using trained models 1 through 4.

python tools/predict.py --model all --input "pork_dataset/unseen data"
Labels refer to the dataset classes; scores are not calibrated food-safety probabilities.
"""

import argparse
import csv
import gc
import json
import os
from pathlib import Path

from test_models import ROOT, EXTENSIONS, image_files, load_images, check_scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["1", "2", "3", "4", "all"], default="4")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/predictions/predictions.csv")
    parser.add_argument("--device", choices=["auto", "cpu", "gpu"], default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.input.is_dir():
        files = image_files(args.input)
    elif args.input.is_file() and args.input.suffix.lower() in EXTENSIONS:
        files = [args.input]
    else:
        parser.error("--input must be an image file or image directory")
    if not files:
        parser.error("No supported images found")
    if args.output.resolve() in {p.resolve() for p in files}:
        parser.error("Output must not overwrite an input image")
    manifest = json.loads((ROOT / "models.json").read_text(encoding="utf-8"))
    ids = list(manifest["models"]) if args.model == "all" else [args.model]
    for model_id in ids:
        path = ROOT / manifest["models"][model_id]["path"]
        if not path.exists() or path.stat().st_size < 1000:
            parser.error(f"Model weights are missing: {path}. Run git lfs pull first.")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    gpus = tf.config.list_physical_devices("GPU")
    if args.device == "gpu" and not gpus:
        parser.error("GPU requested but TensorFlow cannot detect a GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    device = "/GPU:0" if gpus else "/CPU:0"
    rows = []
    for model_id in ids:
        info = manifest["models"][model_id]
        model = tf.keras.models.load_model(ROOT / info["path"], compile=False)
        size = tuple(model.input_shape[1:3])
        if list(size) != info["input_size"]:
            raise ValueError("Model input size differs from models.json")
        for offset in range(0, len(files), args.batch_size):
            batch_files = files[offset:offset+args.batch_size]
            images = load_images(tf, batch_files, size)
            with tf.device(device):
                prediction = model(images, training=False)
            check_scores(prediction.numpy())
            if args.device == "gpu" and "GPU:0" not in prediction.device:
                raise RuntimeError("Prediction did not execute on GPU")
            for path, score in zip(batch_files, prediction.numpy().flatten()):
                score = float(score)
                label = manifest["class_names"][int(score > manifest["threshold"])]
                rows.append({"image": str(path), "model": info["name"],
                             "predicted_class": label, "unsafe_score": score})
                print(f"{info['name']} | {path.name} | {label} | unsafe_score={score:.6f}")
        del model
        tf.keras.backend.clear_session()
        gc.collect()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "model", "predicted_class", "unsafe_score"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} predictions to {args.output} ({device})")


if __name__ == "__main__":
    main()
