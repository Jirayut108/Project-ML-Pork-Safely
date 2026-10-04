"""Train models 1 through 4 in WSL, then evaluate their best saved models.

source ~/ml/bin/activate
python tools/train_models.py

Original architectures, input rescaling, optimizers and fine-tuning boundaries
come directly from the notebooks. Each phase has at most 100 epochs and stops
after 10 epochs without validation-loss improvement. Test images never select
the checkpoint. Existing notebooks and model files are not overwritten.
"""

import argparse
import contextlib
from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from test_models import ROOT, assignment_values, before_fit, image_files, source, write_json

NOTEBOOKS = ["model_1_ResNet50V2(Adam).ipynb", "model_2_InceptionV3.ipynb", "model_3_MobileNetV2.ipynb", "model_4_DenseNet201.ipynb"]


def train(filename, output, max_epochs, patience, batch_override, resume_head=False, disable_jit=False):
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")
    os.environ.setdefault("TF_CUDNN_USE_AUTOTUNE", "0")
    os.environ.setdefault("MPLBACKEND", "Agg")
    start = time.monotonic()
    result = {"file": filename, "status": "running", "max_epochs_per_phase": max_epochs, "patience": patience}
    previous = json.loads((output / "training-result.json").read_text()) if resume_head else None
    stage = "initialization"
    try:
        import numpy as np
        import tensorflow as tf

        tf.config.threading.set_intra_op_parallelism_threads(8)
        tf.config.threading.set_inter_op_parallelism_threads(2)
        gpus = tf.config.list_physical_devices("GPU")
        assert gpus, "No GPU available"
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        tf.keras.utils.set_random_seed(42)
        result["gpu"] = tf.config.experimental.get_device_details(gpus[0])
        result["tensorflow"] = tf.__version__
        notebook = json.loads((ROOT / filename).read_text(encoding="utf-8"))
        settings = assignment_values(source(notebook, 2))
        size = tuple(settings["IMAGE_SIZE"])
        batch = batch_override or settings["BATCH_SIZE"]
        result["image_size"] = size
        result["batch_size"] = batch
        result["seed"] = 42
        result["changes_from_notebook"] = [
            "Project-relative dataset and output paths",
            "EarlyStopping on val_loss, patience=10, restore_best_weights=True",
            "Independent checkpoints for every model and phase",
            "Reload best head weights before fine-tuning",
            "Select final checkpoint by validation loss, evaluate fixed-order test set afterward",
            "Cache resized datasets in memory; shuffle training samples each epoch",
        ]
        if disable_jit:
            result["changes_from_notebook"].append("Use standard GPU execution with XLA disabled")
        opts = tf.data.Options()
        opts.threading.private_threadpool_size = 8
        datasets = {}
        for split in ["train set", "valid set"]:
            ds = tf.keras.utils.image_dataset_from_directory(
                ROOT / "pork_dataset" / split, class_names=["safe", "unsafe"],
                image_size=size, batch_size=None, shuffle=False,
            )
            ds = ds.with_options(opts).cache()
            if split == "train set":
                ds = ds.shuffle(2100, seed=42, reshuffle_each_iteration=True)
            datasets[split] = ds.batch(batch).prefetch(2)
        result["dataset"] = {
            split: {c: len(image_files(ROOT / "pork_dataset" / split / c)) for c in ["safe", "unsafe"]}
            for split in ["train set", "valid set", "test set"]
        }
        env = {"tf": tf, "np": np, "IMAGE_SIZE": size}
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(source(notebook, 4), filename + ":cell4", "exec"), env)
            exec(compile(source(notebook, 5), filename + ":cell5", "exec"), env)
        model = env["model"]
        result["parameters"] = model.count_params()
        result["backbone"] = env["base_model"].name
        assert "GPU:0" in env["base_model"].weights[0].value.handle.device

        class Progress(tf.keras.callbacks.Callback):
            def __init__(self, phase):
                super().__init__()
                self.phase = phase
            def on_epoch_begin(self, epoch, logs=None):
                self.started = time.monotonic()
            def on_epoch_end(self, epoch, logs=None):
                event = {"model": filename, "phase": self.phase, "epoch": epoch+1,
                         "seconds": round(time.monotonic()-self.started, 2),
                         **{k: float(v) for k,v in (logs or {}).items()}}
                write_json(output / "progress.json", event)
                print("EPOCH " + json.dumps(event), flush=True)

        phases = []
        for phase, cell in [("head", 6), ("fine_tune", 9)]:
            stage = phase
            code = before_fit(source(notebook, cell))
            if cell == 6:
                code = code.split("checkpointer =", 1)[0]
            exec(compile(code, filename + ":cell" + str(cell), "exec"), env)
            if disable_jit:
                model.jit_compile = False
                result["jit_compile"] = False
            if phase == "head" and resume_head:
                model.load_weights(output / "head-best.weights.h5")
                phases.append(previous["phases"][0])
                result["phases"] = phases
                result["resumed_from_head_checkpoint"] = True
                result["changes_from_notebook"].append("Resume from completed head checkpoint; training-data shuffle seed restarts in the new process")
                print("RESUMED " + filename + " from head checkpoint", flush=True)
                continue
            weights_path = output / (phase + "-best.weights.h5")
            callbacks = [
                tf.keras.callbacks.TerminateOnNaN(),
                tf.keras.callbacks.ModelCheckpoint(str(weights_path), monitor="val_loss", save_best_only=True, save_weights_only=True),
                tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True),
                tf.keras.callbacks.CSVLogger(str(output / (phase + "-history.csv"))),
                Progress(phase),
            ]
            print(f"PHASE {filename} {phase} batch={batch} max_epochs={max_epochs}", flush=True)
            history = model.fit(datasets["train set"], validation_data=datasets["valid set"],
                                epochs=max_epochs, callbacks=callbacks, verbose=0)
            if not all(np.isfinite(v).all() for v in history.history.values()):
                raise RuntimeError("Training produced nonfinite metrics")
            model.load_weights(weights_path)
            metrics = model.evaluate(datasets["valid set"], return_dict=True, verbose=0)
            phase_file = output / (phase + "-best.keras")
            model.save(phase_file)
            info = {"phase": phase, "epochs": len(history.history["loss"]),
                    "best_epoch": int(np.argmin(history.history["val_loss"]))+1,
                    "optimizer": model.optimizer.__class__.__name__,
                    "learning_rate": float(model.optimizer.learning_rate.numpy()),
                    "trainable_parameters": sum(int(np.prod(v.shape)) for v in model.trainable_weights),
                    "validation": {k: float(v) for k,v in metrics.items()}, "model_file": str(phase_file)}
            phases.append(info)
            result["phases"] = phases
            write_json(output / "training-result.json", result)
        stage = "select_best"
        best = min(phases, key=lambda p: p["validation"]["loss"])
        selected = tf.keras.models.load_model(best["model_file"], compile=False)
        final_file = output / "best_model.keras"
        selected.save(final_file)
        result["selected_phase"] = best["phase"]
        result["best_model"] = str(final_file)
        result["status"] = "trained"
        result["seconds"] = round(time.monotonic()-start, 2)
        write_json(output / "training-result.json", result)
        print("TRAINED " + json.dumps(result), flush=True)
        return 0
    except Exception as error:
        traceback.print_exc()
        result.update(status="failed", stage=stage, error=str(error), traceback=traceback.format_exc(), seconds=round(time.monotonic()-start,2))
        write_json(output / "training-result.json", result)
        return 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", choices=NOTEBOOKS)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--resume-head", action="store_true")
    parser.add_argument("--disable-jit", action="store_true")
    parser.add_argument("--models", nargs="+", choices=["1", "2", "3", "4"], default=["1", "2", "3", "4"])
    args = parser.parse_args()
    if args.worker:
        return train(args.worker, args.output, args.epochs, args.patience, args.batch_size, args.resume_head, args.disable_jit)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = args.output or ROOT / "artifacts" / "training" / stamp
    output.mkdir(parents=True, exist_ok=False)
    summary = {"created_utc": stamp, "status": "running", "models": []}
    write_json(output / "summary.json", summary)
    print("OUTPUT " + str(output), flush=True)
    for model_id in args.models:
        filename = NOTEBOOKS[int(model_id)-1]
        model_out = output / Path(filename).stem
        model_out.mkdir()
        print("START " + filename, flush=True)
        cmd = [sys.executable, "-u", str(Path(__file__).resolve()), "--worker", filename,
               "--output", str(model_out), "--epochs", str(args.epochs), "--patience", str(args.patience)]
        if args.batch_size:
            cmd += ["--batch-size", str(args.batch_size)]
        if args.disable_jit or "DenseNet" in filename:
            cmd += ["--disable-jit"]
        completed = subprocess.run(cmd, cwd=ROOT)
        info_path = model_out / "training-result.json"
        info = json.loads(info_path.read_text()) if info_path.exists() else {"file": filename, "status": "failed", "error": f"Process exit {completed.returncode}"}
        if info["status"] == "trained":
            evaluation = model_out / "evaluation"
            evaluation.mkdir()
            with (evaluation / "run.log").open("w", encoding="utf-8") as log:
                check = subprocess.run([sys.executable, str(ROOT / "tools/test_models.py"), "--worker", "saved",
                                        "--file", info["best_model"], "--output", str(evaluation)],
                                       cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            evaluated = evaluation / "result.json"
            info["evaluation"] = json.loads(evaluated.read_text()) if evaluated.exists() else {"status": "failed", "error": f"Evaluation exit {check.returncode}"}
            info["status"] = "passed" if info["evaluation"]["status"] == "passed" else "evaluation_failed"
        summary["models"].append(info)
        write_json(output / "summary.json", summary)
        print("RESULT " + json.dumps(info), flush=True)
    summary["status"] = "complete" if all(m["status"] == "passed" for m in summary["models"]) else "failed"
    summary["completed_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(output / "summary.json", summary)
    print("REPORT " + str(output / "summary.json"), flush=True)
    return 0 if summary["status"] == "complete" else 1


if __name__ == "__main__":
    sys.exit(main())
