"""Run bounded GPU checks without modifying notebooks or existing model files.

WSL: source ~/ml/bin/activate && python tools/test_models.py
Saved models are evaluated on the complete test set. Notebook checks use two
real training images, one update per training phase, and ImageNet weights.
These notebook checks do not measure the accuracy of trained pork classifiers.
"""

import argparse
import ast
import contextlib
import csv
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}


def image_files(directory):
    return sorted(p for p in directory.rglob("*") if p.suffix.lower() in EXTENSIONS)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def source(nb, index):
    return "".join(nb["cells"][index]["source"])


def before_fit(code):
    """Keep the notebook's exact optimizer and layer-freezing configuration."""
    return code.split("history = model.fit", 1)[0]


def assignment_values(code):
    return {
        node.targets[0].id: ast.literal_eval(node.value)
        for node in ast.parse(code).body
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id in {"IMAGE_SIZE", "BATCH_SIZE"}
    }


def load_images(tf, files, size):
    import numpy as np
    # Match image_dataset_from_directory's TensorFlow decoder and bilinear
    # resize exactly. PIL resizing uses different sampling/antialiasing.
    return np.stack([
        tf.image.resize(
            tf.image.decode_image(tf.io.read_file(str(p)), channels=3, expand_animations=False),
            size, method="bilinear",
        ).numpy() for p in files
    ])


def check_scores(scores):
    import numpy as np
    scores = np.asarray(scores)
    assert scores.ndim == 2 and scores.shape[1] == 1, scores.shape
    assert np.isfinite(scores).all(), "Nonfinite prediction"
    assert ((scores >= 0) & (scores <= 1)).all(), "Invalid sigmoid probability"


def predict(tf, model, images):
    with tf.device("/GPU:0"):
        output = model(images, training=False)
    assert "GPU:0" in output.device, output.device
    values = output.numpy()
    check_scores(values)
    return values


def worker(kind, filename, out):
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")
    os.environ.setdefault("TF_CUDNN_USE_AUTOTUNE", "0")
    os.environ.setdefault("MPLBACKEND", "Agg")
    result = {"kind": kind, "file": filename, "status": "failed"}
    start = time.monotonic()
    stage = "imports"
    try:
        import numpy as np
        import tensorflow as tf
        from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

        tf.config.threading.set_intra_op_parallelism_threads(2)
        tf.config.threading.set_inter_op_parallelism_threads(2)
        result["tensorflow"] = tf.__version__
        gpu = tf.config.list_physical_devices("GPU")
        assert gpu, "No GPU available"
        for device in gpu:
            memory_limit = os.environ.get("ML_TEST_GPU_MEMORY_MB")
            if memory_limit:
                tf.config.set_logical_device_configuration(device, [tf.config.LogicalDeviceConfiguration(memory_limit=int(memory_limit))])
            else:
                tf.config.experimental.set_memory_growth(device, True)
        tf.keras.utils.set_random_seed(42)
        result["gpu"] = tf.config.experimental.get_device_details(gpu[0])
        classes = sorted(p.name for p in (ROOT / "pork_dataset/train set").iterdir() if p.is_dir())
        assert classes == ["safe", "unsafe"], classes
        result["class_names"] = classes
        result["image_preprocessing"] = "tf.image.decode_image(channels=3, expand_animations=False), tf.image.resize(method=bilinear, antialias=False); matches training"
        stage = "load_or_build"

        if kind == "notebook":
            nb = json.loads((ROOT / filename).read_text(encoding="utf-8"))
            settings = assignment_values(source(nb, 2))
            size = tuple(settings["IMAGE_SIZE"])
            result["original_batch_size"] = settings["BATCH_SIZE"]
            result["image_size"] = size
            result["overrides"] = {
                "dataset": str(ROOT / "pork_dataset"),
                "training_images": 2,
                "updates_per_phase": 1,
                "checkpoint": str(out / "smoke-checkpoint.keras"),
            }
            env = {"tf": tf, "np": np, "IMAGE_SIZE": size}
            with contextlib.redirect_stdout(io.StringIO()):
                exec(compile(source(nb, 4), filename + ":cell4", "exec"), env)
                exec(compile(source(nb, 5), filename + ":cell5", "exec"), env)
            model, base = env["model"], env["base_model"]
            result["backbone"] = base.name
            result["parameters"] = model.count_params()
            train_files = [image_files(ROOT / "pork_dataset/train set" / c)[0] for c in classes]
            images = load_images(tf, train_files, size)
            labels = np.array([0, 1], dtype=np.float32).reshape(-1, 1)
            stage = "forward"
            result["initial_predictions"] = predict(tf, model, images).flatten().tolist()
            stage = "head_training"
            # Instantiate the notebook's callback with an isolated output path.
            code = before_fit(source(nb, 6)).replace(
                "filepath='model.weights.best.keras'",
                "filepath=" + repr(str(out / "smoke-checkpoint.keras")),
            )
            exec(compile(code, filename + ":cell6", "exec"), env)
            if "DenseNet" in filename:
                model.jit_compile = False
            result["head_optimizer"] = model.optimizer.__class__.__name__
            result["head_trainable_parameters"] = sum(int(np.prod(v.shape)) for v in model.trainable_weights)
            head = model.train_on_batch(images, labels, return_dict=True)
            assert all(np.isfinite(v) for v in head.values()), head
            result["head_training"] = {k: float(v) for k, v in head.items()}
            assert "GPU:0" in base.weights[0].value.handle.device
            stage = "fine_tuning"
            exec(compile(before_fit(source(nb, 9)), filename + ":cell9", "exec"), env)
            if "DenseNet" in filename:
                model.jit_compile = False
            result["fine_tune_optimizer"] = model.optimizer.__class__.__name__
            result["fine_tune_trainable_parameters"] = sum(int(np.prod(v.shape)) for v in model.trainable_weights)
            fine = model.train_on_batch(images, labels, return_dict=True)
            assert all(np.isfinite(v) for v in fine.values()), fine
            result["fine_tuning"] = {k: float(v) for k, v in fine.items()}
            stage = "validation"
            valid_files = [image_files(ROOT / "pork_dataset/valid set" / c)[0] for c in classes]
            result["validation_predictions"] = predict(tf, model, load_images(tf, valid_files, size)).flatten().tolist()
            stage = "full_test_inference"
            test_files = [p for c in classes for p in image_files(ROOT / "pork_dataset/test set" / c)]
            test_scores = []
            for offset in range(0, len(test_files), 16):
                test_scores.extend(predict(tf, model, load_images(tf, test_files[offset:offset+16], size)).flatten().tolist())
            result["test_count"] = len(test_files)
            with (out / "test_predictions.csv").open("w", newline="", encoding="utf-8-sig") as f:
                writer = csv.writer(f)
                writer.writerow(["image", "true_class", "smoke_predicted_class", "smoke_unsafe_score"])
                writer.writerows((str(p.relative_to(ROOT)), p.parent.name, classes[int(s > 0.5)], s) for p, s in zip(test_files, test_scores))
            stage = "unseen"
            unseen = image_files(ROOT / "pork_dataset/unseen data")
            result["unseen_count"] = len(unseen)
            result["unseen_predictions"] = predict(tf, model, load_images(tf, unseen, size)).flatten().tolist()
            stage = "save_reload"
            saved = out / "smoke-checkpoint.keras"
            model.save(saved)
            reloaded = tf.keras.models.load_model(saved, compile=False)
            np.testing.assert_allclose(predict(tf, model, images), predict(tf, reloaded, images), rtol=1e-5, atol=1e-6)
            saved.unlink()  # Only the newly created, isolated smoke checkpoint.
            stage = "diagram"
            tf.keras.utils.plot_model(model, to_file=str(out / "architecture.png"), show_shapes=True)
            result["diagram"] = "architecture.png"
            result["scope"] = "GPU forward, one head update, one fine-tune update, validation samples, all test images, 10 unseen images, save/reload and Graphviz; not full training or trained accuracy evaluation"
        else:
            model = tf.keras.models.load_model(ROOT / filename, compile=False)
            size = tuple(int(x) for x in model.input_shape[1:3])
            result["image_size"] = size
            result["parameters"] = model.count_params()
            result["backbone"] = next((layer.name for layer in model.layers if isinstance(layer, tf.keras.Model)), None)
            stage = "full_test_set"
            test_files = []
            labels = []
            for i, cls in enumerate(classes):
                files = image_files(ROOT / "pork_dataset/test set" / cls)
                test_files.extend(files)
                labels.extend([i] * len(files))
            scores = []
            evaluation_batch = int(os.environ.get("ML_TEST_BATCH_SIZE", "16"))
            result["evaluation_batch_size"] = evaluation_batch
            for offset in range(0, len(test_files), evaluation_batch):
                batch = load_images(tf, test_files[offset:offset+evaluation_batch], size)
                scores.extend(predict(tf, model, batch).flatten().tolist())
            labels = np.array(labels)
            scores = np.array(scores)
            predicted = (scores > 0.5).astype(int)
            result["test_count"] = len(labels)
            result["accuracy"] = float(accuracy_score(labels, predicted))
            result["confusion_matrix"] = confusion_matrix(labels, predicted, labels=[0, 1]).tolist()
            result["classification_report"] = classification_report(labels, predicted, labels=[0, 1], target_names=classes, output_dict=True, zero_division=0)
            eps = 1e-7
            p = np.clip(scores, eps, 1-eps)
            result["binary_crossentropy"] = float(np.mean(-labels*np.log(p)-(1-labels)*np.log(1-p)))
            with (out / "test_predictions.csv").open("w", newline="", encoding="utf-8-sig") as f:
                writer = csv.writer(f)
                writer.writerow(["image", "true_class", "predicted_class", "unsafe_score"])
                writer.writerows((str(path.relative_to(ROOT)), classes[y], classes[yhat], score) for path, y, yhat, score in zip(test_files, labels, predicted, scores))
            stage = "unseen"
            unseen = image_files(ROOT / "pork_dataset/unseen data")
            unseen_scores = predict(tf, model, load_images(tf, unseen, size)).flatten()
            result["unseen_count"] = len(unseen)
            with (out / "unseen_predictions.csv").open("w", newline="", encoding="utf-8-sig") as f:
                writer = csv.writer(f)
                writer.writerow(["image", "predicted_class", "unsafe_score"])
                writer.writerows((p.name, classes[int(s > 0.5)], float(s)) for p, s in zip(unseen, unseen_scores))
            result["scope"] = "Saved weights, compile=False, all test images in fixed order, threshold >0.5, 10 unlabeled unseen images"
        result["status"] = "passed"
    except Exception as error:
        result["stage"] = stage
        result["error"] = str(error)
        result["traceback"] = traceback.format_exc()
        traceback.print_exc()
    result["seconds"] = round(time.monotonic() - start, 2)
    write_json(out / "result.json", result)
    print(json.dumps({k: result.get(k) for k in ["file", "status", "stage", "accuracy", "seconds", "error"]}), flush=True)
    return 0 if result["status"] == "passed" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", choices=["notebook", "saved"])
    parser.add_argument("--file")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--notebooks", nargs="+", help="Test only these notebook filenames; skip saved models")
    args = parser.parse_args()
    if args.worker:
        args.output.mkdir(parents=True, exist_ok=True)
        return worker(args.worker, args.file, args.output)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = args.output or ROOT / "artifacts" / "model-tests" / stamp
    output.mkdir(parents=True, exist_ok=False)
    if args.notebooks:
        tasks = [("notebook", ROOT / name) for name in args.notebooks]
        if any(not p.is_file() or p.suffix != ".ipynb" or p.parent != ROOT for _, p in tasks):
            parser.error("--notebooks must name existing notebooks in the project root")
    else:
        tasks = [("saved", p) for p in sorted(ROOT.glob("*.h5")) + sorted(ROOT.glob("*.keras"))]
        tasks += [("notebook", p) for p in sorted(ROOT.glob("*.ipynb"))]
    summary = {"created_utc": stamp, "dataset": {}, "results": [], "notebook_findings": []}
    for split in ["train set", "valid set", "test set"]:
        summary["dataset"][split] = {c.name: len(image_files(c)) for c in sorted((ROOT / "pork_dataset" / split).iterdir()) if c.is_dir()}
    for kind, path in tasks:
        if kind == "notebook":
            nb = json.loads(path.read_text(encoding="utf-8"))
            issues = []
            if "/mnt/c/Users/jiray/" in source(nb, 1):
                issues.append("Dataset path refers to original user jiray; overridden for this test")
            if "shuffle=True" in source(nb, 2):
                issues.append("Original test dataset shuffles: separate predict and label iterations can misalign the confusion matrix; saved evaluation uses fixed order")
            if "DenseNet" in path.name:
                issues.append("Save cell names a ResNet RMSprop model, risking overwrite")
                issues.append("Unseen cell resizes to 256x256 but the model expects 224x224; test uses model input size")
            if "RMSprop" in path.name:
                issues.append("Fine-tuning switches from RMSprop to Adam")
            summary["notebook_findings"].append({"file": path.name, "issues": issues})
        out = output / (kind + "-" + path.stem)
        out.mkdir()
        print("START", kind, path.name, flush=True)
        try:
            with (out / "run.log").open("w", encoding="utf-8") as log:
                completed = subprocess.run(
                    [sys.executable, str(Path(__file__).resolve()), "--worker", kind, "--file", path.name, "--output", str(out)],
                    cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=600,
                )
            result_path = out / "result.json"
            if result_path.exists():
                result = json.loads(result_path.read_text(encoding="utf-8"))
            else:
                result = {"kind": kind, "file": path.name, "status": "failed", "error": f"Worker exit code {completed.returncode}; see run.log"}
        except subprocess.TimeoutExpired:
            result = {"kind": kind, "file": path.name, "status": "failed", "error": "Exceeded 600-second per-model test limit"}
        result["artifacts"] = str(out.relative_to(ROOT))
        summary["results"].append(result)
        write_json(output / "summary.json", summary)
        print("RESULT", result["file"], result["status"], result.get("accuracy", ""), result.get("error", ""), flush=True)
    print("REPORT", output / "summary.json", flush=True)
    return 0 if all(r["status"] == "passed" for r in summary["results"]) else 1


if __name__ == "__main__":
    sys.exit(main())
