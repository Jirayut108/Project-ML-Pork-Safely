# Project ML Pork Safely

โปรเจกต์จำแนกภาพหมูตามคลาส `safe` และ `unsafe` ของชุดข้อมูล ชุดโมเดล 1–4 ใช้ ResNet50V2 (Adam), InceptionV3, MobileNetV2 และ DenseNet201

## ผลฝึกและทดสอบโมเดล 1–4

รอบฝึกวันที่ 4 ตุลาคม 2026 บน WSL2, Python 3.12, TensorFlow 2.21.0 และ NVIDIA GeForce RTX 4070 Ti ใช้ train 2,100 ภาพ, validation 300 ภาพ และ test 599 ภาพ เลือก checkpoint จาก validation loss ก่อนประเมิน test set โดยไม่สุ่มลำดับ test

| โมเดล | Epochs ส่วนหัว + fine-tune | Test accuracy | Unsafe recall | unsafe ที่ทำนายเป็น safe |
|---|---:|---:|---:|---:|
| 1: ResNet50V2 (Adam) | 92 + 25 | 81.97% | 99.00% | 3/300 |
| 2: InceptionV3 | 15 + 18 | 84.81% | 97.67% | 7/300 |
| 3: MobileNetV2 | 14 + 18 | 78.96% | 99.67% | 1/300 |
| 4: DenseNet201 | 39 + 29 | 85.98% | 99.00% | 3/300 |

[รายงาน กราฟ confusion matrix และผลรายภาพ](artifacts/training/20261004T123848Z/report.md) · [สรุปตัวเลข](artifacts/training/20261004T123848Z/metrics.csv) · [ตำแหน่งโมเดลและ input](models.json)

ทั้งสี่โมเดลผ่านการโหลดจากไฟล์และทำนายบน GPU แล้ว พร้อมสำหรับทดลองและพัฒนาต่อ คะแนนนี้เป็นผลจาก test set ของโปรเจกต์หนึ่งชุด ยังไม่มีการประเมินกับชุดภาพภายนอกที่ยืนยัน labels แล้ว ในรอบนี้โมเดล 1 ใช้ ResNet50V2 (Adam) ส่วน notebook RMSprop และ VGG19 เป็นการทดลองเดิมที่ยังเก็บไว้

คำว่า `safe` เป็นชื่อคลาสของ dataset ไม่ใช่การรับรองว่าหมูปลอดภัยต่อการบริโภค ลักษณะและสีเพียงอย่างเดียวไม่ยืนยันการเน่าเสีย ตาม [USDA FSIS](https://www.fsis.usda.gov/food-safety/safe-food-handling-and-preparation/food-safety-basics/color-meat-and-poultry) ควรตรวจสอบผลกับวิธีประเมินคุณภาพที่เหมาะสมก่อนนำไปใช้ตัดสินใจจริง

## ติดตั้งและดาวน์โหลดโมเดล

คำสั่งต่อไปนี้ใช้ใน Ubuntu/WSL2 โมเดลขนาดใหญ่เก็บผ่าน Git LFS:

```bash
git clone https://github.com/Jirayut108/Project-ML-Pork-Safely.git
cd Project-ML-Pork-Safely
git lfs install
git lfs pull
python3.12 -m venv ~/ml
source ~/ml/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-ml.txt
```

หากใช้ชุดผลนี้ก่อนรวมเข้า `main` ให้รัน `git switch codex/model-1-4-training-results` หลังเข้าโฟลเดอร์โปรเจกต์ แล้วจึงรัน `git lfs pull` เพื่อดาวน์โหลดโมเดลชุดนี้

หากมี `~/ml` ที่ติดตั้งแล้ว ให้ activate ตัวเดิมได้ ไม่ต้องสร้างใหม่ CPU ใช้ทำนายได้ ส่วนการฝึกและทดสอบด้วยตัวรันในโปรเจกต์ต้องใช้ GPU ติดตั้ง Graphviz เพิ่มหากต้องการทดสอบวาดโครงสร้างโมเดลจาก notebook

ตรวจ GPU ด้วย `python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"` หากไม่พบ ให้ตรวจ driver และขั้นตอนใน [คู่มือ TensorFlow](https://www.tensorflow.org/install/pip) ก่อนฝึก

## ทำนายภาพใหม่

```bash
source ~/ml/bin/activate
# เปรียบเทียบโมเดลทั้งสี่กับภาพ unseen 10 ภาพ
python tools/predict.py --model all --input "pork_dataset/unseen data"
# ภาพเดียวกับ DenseNet201
python tools/predict.py --model 4 --input "/path/to/pork.jpg"
# ใช้ CPU และระบุไฟล์ผลลัพธ์
python tools/predict.py --model 3 --device cpu --input "/path/to/images" --output predictions.csv
```

ไฟล์ผลลัพธ์เริ่มต้นคือ `artifacts/predictions/predictions.csv` มีชื่อภาพ โมเดล คลาสที่ทำนาย และคะแนน `unsafe` ใช้ threshold > 0.5 คะแนนไม่ได้ผ่านการปรับเทียบให้เป็นความน่าจะเป็นด้านความปลอดภัย

โมเดลย่อภาพเป็น 256×256 สำหรับ ResNet50V2, 128×128 สำหรับ InceptionV3 และ 224×224 สำหรับ MobileNetV2/DenseNet201 ด้วย TensorFlow bilinear resize การหาร 255 อยู่ในโมเดลแล้ว จึงไม่ต้องหารซ้ำ ผลตัวเลขอาจคลาดเคลื่อนเล็กน้อยตามอุปกรณ์และ batch size

## ไฟล์ `.h5` สำหรับโมเดล 2–4

ไฟล์เหล่านี้แปลงจาก `best_model.keras` ของรอบ `20261004T123848Z` โดยเก็บโครงสร้างและน้ำหนักที่ฝึกแล้วครบ เหมาะสำหรับโหลดทำนาย ไม่ได้เก็บสถานะ optimizer สำหรับฝึกต่อ:

| โมเดล | ไฟล์ | ขนาดภาพเข้า |
|---|---|---|
| 2: InceptionV3 | [model2_InceptionV3.h5](model2_InceptionV3.h5) | 128×128 |
| 3: MobileNetV2 | [model3_MobileNetV2.h5](model3_MobileNetV2.h5) | 224×224 |
| 4: DenseNet201 | [model4_DenseNet201.h5](model4_DenseNet201.h5) | 224×224 |

ดาวน์โหลดไฟล์จริงด้วย `git lfs pull` แล้วโหลดได้ด้วย:

```python
import tensorflow as tf

model = tf.keras.models.load_model("model2_InceptionV3.h5", compile=False)
```

ใช้ภาพ RGB ช่วงค่า 0–255 และย่อด้วย TensorFlow bilinear resize ตามขนาดของแต่ละโมเดล การหาร 255 อยู่ในโมเดลแล้ว ผลลัพธ์เป็นคะแนน `unsafe` โดยใช้ค่า > 0.5 เป็นคลาส `unsafe`

ตรวจสอบด้วย TensorFlow 2.21.0 โดยโหลด `.h5` กลับ ตรวจน้ำหนักทุกตัวว่าเท่ากับต้นฉบับ และเปรียบเทียบผลทำนายภาพ unseen ทั้ง 10 ภาพ ดู [ผลตรวจสอบการแปลง](artifacts/training/20261004T123848Z/h5-export-verification.json)

## ฝึกและตรวจสอบใหม่

```bash
python tools/train_models.py
```

ฝึกโมเดล 1–4 ตามลำดับ ช่วงละสูงสุด 100 epochs และใช้ EarlyStopping เมื่อ validation loss ไม่ดีขึ้น 10 epochs ทุกโมเดลมี checkpoint แยกกัน เลือกผลที่ดีที่สุดจากส่วนหัวและ fine-tune แล้วทดสอบครบ 599 ภาพ ผลอยู่ใน `artifacts/training/<run>/` DenseNet201 ปิด XLA เพื่อหลีกเลี่ยงการคอมไพล์ที่ใช้เวลานาน ใช้ `--models 1` เพื่อฝึกเฉพาะโมเดล 1 หรือ `--models 2 3 4` เพื่อฝึกเฉพาะสามตัวที่เหลือ

สร้างรายงานหลังฝึกด้วย:

```bash
python tools/summarize_training.py artifacts/training/<run>
```

เมื่อฝึกใหม่และตรวจสอบผลแล้ว ให้อัปเดตพาธและตัวเลขใน `models.json` ให้ชี้ไปยังรอบใหม่ก่อนใช้คำสั่งทำนาย ปัจจุบันไฟล์นี้ชี้ไปยังรอบที่ทดสอบแล้ว `20261004T123848Z`

ทดสอบโมเดลที่บันทึกไว้โดยไม่ฝึกใหม่:

```bash
python tools/test_models.py --worker saved \
  --file artifacts/training/20261004T123848Z/model_4_DenseNet201/best_model.keras \
  --output artifacts/model-tests/densenet201
```

ตัวรันใช้ architecture, optimizer และการ freeze layers จาก notebook ต้นฉบับ แต่ใช้พาธโปรเจกต์ปัจจุบัน แยกชื่อ checkpoint และจับคู่ภาพ test กับ labels ตามลำดับคงที่ การเปิด notebook เดิมแล้วรันทุก cell ยังต้องแก้พาธเก่าและชื่อไฟล์บันทึกเอง รายงาน JSON บันทึกพาธของเครื่องที่ฝึกเป็นข้อมูลประวัติ ใช้ `models.json` สำหรับพาธที่ใช้งานได้หลัง clone

## งานต่อยอด

1. เพิ่มชุดทดสอบภายนอกจากกล้อง แสง และสถานที่ที่จะใช้จริง พร้อม labels ที่ตรวจสอบแล้ว
2. ปรับ preprocessing/augmentation และ learning rate โดยใช้ train/validation เท่านั้น แล้วใช้ test ชุดใหม่ประเมินรอบถัดไป
3. เลือกโมเดลและ threshold จาก validation ตามต้นทุนของการพลาด `unsafe` และการแจ้งเตือนผิด พร้อมวัดเวลา inference
4. เชื่อม `tools/predict.py` กับหน้ารับภาพหรือ API โดยแสดงผลเป็นการจำแนกภาพและรองรับกรณีที่ข้อมูลไม่เพียงพอ

## ไฟล์ที่เผยแพร่

- `tools/`: ตัวฝึก ตัวประเมิน ตัวทำนาย และตัวสร้างรายงาน
- `models.json`: พาธโมเดล ขนาดภาพ และผลทดสอบ
- `model2_InceptionV3.h5`, `model3_MobileNetV2.h5`, `model4_DenseNet201.h5`: โมเดล 2–4 ในรูปแบบ HDF5 สำหรับโหลดทำนาย
- `requirements-ml.txt` และ `requirements-ml-lock.txt`: dependency หลักและเวอร์ชัน environment ที่ใช้ทดสอบ
- `artifacts/training/20261004T123848Z/`: best_model.keras ทั้งสี่ตัว ประวัติการฝึก รายงาน กราฟ และผลทำนายรายภาพ

ไฟล์ optimizer/checkpoint ระหว่างฝึกเก็บไว้ในเครื่องและไม่นำขึ้น GitHub ส่วนโมเดลที่เลือกเก็บด้วย Git LFS
