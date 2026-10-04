"""Create a readable report and curves from completed model training results."""

import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    output = Path(sys.argv[1]).resolve()
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    first = summary["models"][0]
    dataset = first["dataset"]
    train_count = sum(dataset["train set"].values())
    valid_count = sum(dataset["valid set"].values())
    test_count = sum(dataset["test set"].values())
    lines = [
        "# ผลการฝึกและทดสอบโมเดล 1–4", "",
        f"ใช้ train {train_count:,} ภาพ, validation {valid_count:,} ภาพ และ test {test_count:,} ภาพ (safe {dataset['test set']['safe']}, unsafe {dataset['test set']['unsafe']})",
        "ไม่นับ desktop.ini ในโฟลเดอร์ test เพราะไม่ใช่ไฟล์รูปภาพ",
        f"ฝึกส่วนหัวและ fine-tune ช่วงละไม่เกิน {first['max_epochs_per_phase']} epochs โดยหยุดเมื่อ validation loss ไม่ดีขึ้น {first['patience']} epochs",
        "เลือกโมเดลจาก validation loss เท่านั้น แล้วทดสอบภาพตามลำดับคงที่ โดยคะแนน unsafe > 0.5 ถือเป็น unsafe",
        "การอ่านและย่อภาพใช้ขั้นตอนเดียวกับการฝึก: TensorFlow decode_image และ resize แบบ bilinear",
        "ใช้ architecture, Rescaling(1/255), optimizer และขอบเขตการ freeze layers ตาม notebook เดิม",
        "ทุกโมเดลฝึกตามลำดับบน NVIDIA GeForce RTX 4070 Ti ใน WSL2", "",
        "| Model | Head epochs | Fine-tune epochs | Selected phase | Test accuracy | Unsafe recall | Unsafe F1 | Test loss |",
        "|---|---:|---:|---|---:|---:|---:|---:|",
    ]
    rows = []
    fig, axes = plt.subplots(len(summary["models"]), 2, figsize=(13, 3.7 * len(summary["models"])), constrained_layout=True, squeeze=False)
    for index, model in enumerate(summary["models"]):
        name = Path(model["file"]).stem
        evaluation = model.get("evaluation", {})
        if model["status"] != "passed":
            lines.append(f"| {name} | - | - | Failed | - | - | - | - |")
            lines.extend(["", f"Failure: {model.get('error', evaluation.get('error', 'See logs'))}", ""])
            continue
        phases = model["phases"]
        metrics = evaluation["classification_report"]["unsafe"]
        rows.append({"model": name, "head_epochs": phases[0]["epochs"], "fine_tune_epochs": phases[1]["epochs"],
                     "selected_phase": model["selected_phase"], "test_images": evaluation["test_count"],
                     "accuracy": evaluation["accuracy"], "unsafe_recall": metrics["recall"],
                     "unsafe_f1": metrics["f1-score"], "loss": evaluation["binary_crossentropy"]})
        lines.append(f"| {name} | {phases[0]['epochs']} | {phases[1]['epochs']} | {model['selected_phase']} | {evaluation['accuracy']:.2%} | {metrics['recall']:.2%} | {metrics['f1-score']:.2%} | {evaluation['binary_crossentropy']:.4f} |")
        model_dir = output / name
        offset = 0
        histories = []
        for phase in phases:
            with (model_dir / (phase["phase"] + "-history.csv")).open() as f:
                history = list(csv.DictReader(f))
            histories.extend(history)
            offset += len(history)
        epoch = list(range(1, len(histories)+1))
        for ax, metric, title in [(axes[index, 0], "accuracy", "Accuracy"), (axes[index, 1], "loss", "Binary crossentropy")]:
            ax.plot(epoch, [float(h[metric]) for h in histories], label="Training", linewidth=1.5)
            ax.plot(epoch, [float(h["val_"+metric]) for h in histories], label="Validation", linewidth=1.5)
            ax.axvline(phases[0]["epochs"]+0.5, color="gray", linestyle="--", linewidth=1, label="Fine-tuning starts")
            ax.set(title=f"{model['backbone']}: {title}", xlabel="Epoch", ylabel=title)
            ax.grid(alpha=0.2)
            if metric == "accuracy":
                ax.set_ylim(0, 1.02)
            ax.legend(fontsize=8)
    lines.extend(["", "![Training and validation curves](training-curves.png)", ""])
    for model in summary["models"]:
        if model["status"] != "passed":
            continue
        name = Path(model["file"]).stem
        evaluation = model["evaluation"]
        cm = evaluation["confusion_matrix"]
        time_label = "เวลาในรอบ fine-tuning ที่เริ่มต่อ" if model.get("resumed_from_head_checkpoint") else "เวลาฝึก"
        lines.extend([
            f"## {name}", "",
            f"Best model: [{name}/best_model.keras]({name}/best_model.keras)", "",
            "| Actual class | Predicted safe | Predicted unsafe |",
            "|---|---:|---:|",
            f"| safe | {cm[0][0]} | {cm[0][1]} |",
            f"| unsafe | {cm[1][0]} | {cm[1][1]} |", "",
            f"Per-image test predictions: [{name}/evaluation/test_predictions.csv]({name}/evaluation/test_predictions.csv)",
            f"Unlabeled unseen-image predictions: [{name}/evaluation/unseen_predictions.csv]({name}/evaluation/unseen_predictions.csv)", "",
            f"{time_label}: {model['seconds']/60:.1f} นาที ขนาด input: {model['image_size']} Batch size: {model['batch_size']}", "",
        ])
    lines.extend([
        "## ปัญหาใน notebook ที่จัดการในการรันครั้งนี้", "",
        "- พาธข้อมูลและพาธบันทึกโมเดลอ้างโฟลเดอร์ของผู้ใช้เดิม ตัวรันใช้โฟลเดอร์โปรเจกต์ปัจจุบัน",
        "- ทุก notebook ใช้ชื่อ checkpoint เดียวกัน ครั้งนี้แยกไฟล์ตามโมเดลและช่วงฝึก",
        "- DenseNet201 บันทึกด้วยชื่อไฟล์ของ ResNet RMSprop ครั้งนี้แยกบันทึก DenseNet201",
        "- DenseNet201 ย่อ unseen เป็น 256x256 แต่โมเดลรับ 224x224 ครั้งนี้ใช้ขนาด input ของโมเดล",
        "- Notebook สุ่มลำดับ test; การทำนายและอ่าน labels คนละรอบอาจทำให้ confusion matrix จับคู่ผิด ครั้งนี้จับคู่ภาพและ label ตามลำดับคงที่",
        "- Unseen 10 ภาพไม่มี labels ที่ยืนยันแล้ว จึงรายงานการทำนายแต่ไม่รายงาน accuracy",
        "- ไม่ได้แก้ notebook ต้นฉบับหรือเขียนทับไฟล์โมเดลที่มีอยู่ก่อน",
        "- DenseNet201 ปิด XLA เพื่อแก้การคอมไพล์ fine-tuning ที่ใช้เวลานาน และเริ่มต่อจาก checkpoint ส่วนหัวที่ฝึกเสร็จแล้ว",
        "- ตัวเลขในรายงานมาจากการฝึกหนึ่งครั้งและ test set ที่มีอยู่ในโปรเจกต์", "",
    ])
    fig.savefig(output / "training-curves.png", dpi=160)
    plt.close(fig)
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    if rows:
        with (output / "metrics.csv").open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps(rows, indent=2))
    print(output / "report.md")


if __name__ == "__main__":
    main()
