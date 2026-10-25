import streamlit as st
import numpy as np
from PIL import Image, ImageOps
import os
import time
import pandas as pd
from datetime import datetime
import keras

# ==============================================================================
# กำหนดค่าหน้าเว็บ (Page Configuration)
# ==============================================================================
st.set_page_config(
    page_title="ระบบประเมินความสดและการเน่าเสียของเนื้อหมู",
    layout="wide",
    initial_sidebar_state="collapsed"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ==============================================================================
# โมเดลหลักสำหรับการ Deploy (Champion Model: DenseNet201)
# ==============================================================================
MODEL_CONFIG = {
    "name": "DenseNet201",
    "filename": "model4_DenseNet201_v2.keras",
    "input_size": (224, 224),
    "accuracy": "95.00%",
    "recall_unsafe": "95.33%",
    "size": "73.0 MB",
    "desc": "โมเดลโครงข่าย DenseNet201 มีความแม่นยำสูงและดักจับเนื้อเน่าเสียได้อย่างมีประสิทธิภาพ"
}

@st.cache_resource(show_spinner="กำลังโหลดโมเดล DenseNet201...")
def load_deployment_model():
    candidates = [
        os.path.join(BASE_DIR, "New_train_Model", "model4_DenseNet201_v2.keras"),
        os.path.join(BASE_DIR, "model4_DenseNet201_v2.keras"),
        os.path.join(BASE_DIR, "New_train_Model", "model4_DenseNet201.h5"),
        os.path.join(BASE_DIR, "model4_DenseNet201.h5"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return keras.models.load_model(p, compile=False)
    return None

model = load_deployment_model()

# ==============================================================================
# สไตล์ CSS กลาง: Responsive ออกแบบให้สวยทั้งบนจอคอมและมือถือ (Mobile-First UX/UI)
# ==============================================================================
st.markdown("""
<style>
    /* Google Fonts & Modern Typography */
    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    }
    
    .stApp {
        background-color: #F8FAFC;
    }

    /* ซ่อนแถบ Sidebar อย่างสมบูรณ์ เพื่อความคลีนของแอปหน้าเดียว */
    [data-testid="collapsedControl"] { display: none !important; }
    section[data-testid="stSidebar"] { display: none !important; }

    /* ปรับแต่ง Container กลางให้พอดีสายตา */
    .block-container {
        max-width: 1080px;
        padding-top: 1.2rem !important;
        padding-bottom: 2.5rem !important;
    }

    /* กล่องหัวเรื่องหลัก (Modern App Header) */
    .app-header {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 16px;
        padding: 18px 22px;
        margin-bottom: 16px;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.03);
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 12px;
    }
    .app-header-title {
        font-size: 1.4rem;
        font-weight: 800;
        color: #0F172A;
        letter-spacing: -0.02em;
    }
    .app-header-sub {
        color: #64748B;
        font-size: 0.86rem;
        margin-top: 3px;
    }
    .header-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: #F1F5F9;
        border: 1px solid #CBD5E1;
        padding: 6px 14px;
        border-radius: 24px;
        font-size: 0.8rem;
        font-weight: 600;
        color: #334155;
    }
    .status-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background-color: #10B981;
        box-shadow: 0 0 0 3px rgba(16, 185, 129, 0.2);
    }

    /* กล่องแจ้งเตือนข้อจำกัด (Strict Notice) */
    .warning-box {
        background: linear-gradient(135deg, #FFFBEB 0%, #FEF3C7 100%);
        border: 1px solid #FCD34D;
        border-left: 5px solid #F59E0B;
        border-radius: 12px;
        padding: 13px 18px;
        margin-bottom: 18px;
        font-size: 0.86rem;
        line-height: 1.55;
        color: #92400E;
        box-shadow: 0 2px 6px rgba(245, 158, 11, 0.05);
    }

    /* การ์ดแดชบอร์ดสรุปตัวเลข (Modern Stat Summary Grid) */
    .stat-summary-container {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 12px;
        margin: 16px 0 22px 0;
    }
    .stat-card {
        background: #FFFFFF;
        border-radius: 14px;
        padding: 15px 16px;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.03);
        border: 1px solid #E2E8F0;
        position: relative;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        transition: transform 0.15s ease;
    }
    .stat-card.total {
        background: linear-gradient(145deg, #FFFFFF 0%, #F8FAFC 100%);
        border-color: #CBD5E1;
        border-top: 3px solid #64748B;
    }
    .stat-card.safe {
        background: linear-gradient(145deg, #F0FDF4 0%, #DCFCE7 100%);
        border-color: #86EFAC;
        border-top: 3px solid #16A34A;
    }
    .stat-card.unsafe {
        background: linear-gradient(145deg, #FEF2F2 0%, #FEE2E2 100%);
        border-color: #FCA5A5;
        border-top: 3px solid #DC2626;
    }
    .stat-card-label {
        font-size: 0.8rem;
        font-weight: 700;
        letter-spacing: 0.01em;
        display: flex;
        align-items: center;
        gap: 6px;
    }
    .stat-card.total .stat-card-label { color: #475569; }
    .stat-card.safe .stat-card-label { color: #166534; }
    .stat-card.unsafe .stat-card-label { color: #991B1B; }

    .stat-card-value {
        font-size: 2.1rem;
        font-weight: 800;
        line-height: 1.1;
        margin: 6px 0 2px 0;
        letter-spacing: -0.02em;
    }
    .stat-card.total .stat-card-value { color: #0F172A; }
    .stat-card.safe .stat-card-value { color: #15803D; }
    .stat-card.unsafe .stat-card-value { color: #B91C1C; }

    .stat-card-unit {
        font-size: 0.88rem;
        font-weight: 600;
        color: #64748B;
        margin-left: 3px;
    }
    .stat-card-sub {
        font-size: 0.78rem;
        font-weight: 600;
        margin-top: 4px;
    }
    .stat-card.total .stat-card-sub { color: #64748B; }
    .stat-card.safe .stat-card-sub { color: #15803D; }
    .stat-card.unsafe .stat-card-sub { color: #B91C1C; }

    .stat-card-progress {
        height: 5px;
        background: rgba(0, 0, 0, 0.06);
        border-radius: 4px;
        margin-top: 8px;
        overflow: hidden;
    }
    .stat-card.total .stat-card-fill { height: 100%; background: #64748B; border-radius: 4px; }
    .stat-card.safe .stat-card-fill { height: 100%; background: #16A34A; border-radius: 4px; }
    .stat-card.unsafe .stat-card-fill { height: 100%; background: #DC2626; border-radius: 4px; }

    /* ป้าย Badge ผลลัพธ์ */
    .safe-badge {
        background-color: #DCFCE7;
        color: #15803D;
        border: 1px solid #86EFAC;
        padding: 5px 12px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.84rem;
        display: inline-flex;
        align-items: center;
        gap: 5px;
    }
    .unsafe-badge {
        background-color: #FEE2E2;
        color: #B91C1C;
        border: 1px solid #FCA5A5;
        padding: 5px 12px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.84rem;
        display: inline-flex;
        align-items: center;
        gap: 5px;
    }

    /* การ์ดผลลัพธ์การประเมิน (Single Image Mode) */
    .result-card-safe {
        background: linear-gradient(145deg, #F0FDF4 0%, #FFFFFF 100%);
        border: 2px solid #22C55E;
        border-radius: 16px;
        padding: 20px;
        margin-bottom: 14px;
        box-shadow: 0 4px 14px rgba(34, 197, 94, 0.08);
    }
    .result-card-unsafe {
        background: linear-gradient(145deg, #FEF2F2 0%, #FFFFFF 100%);
        border: 2px solid #EF4444;
        border-radius: 16px;
        padding: 20px;
        margin-bottom: 14px;
        box-shadow: 0 4px 14px rgba(239, 68, 68, 0.08);
    }

    /* การ์ดข้อมูลเทคนิค */
    .tech-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 14px 18px;
        margin-top: 14px;
        font-size: 0.84rem;
        color: #475569;
        box-shadow: 0 1px 4px rgba(0, 0, 0, 0.02);
    }

    /* กล่องหมายเหตุขอบเขตโครงการ */
    .scope-box {
        background-color: #FFFBEB;
        border: 1px solid #FDE68A;
        border-radius: 10px;
        padding: 13px 16px;
        margin-top: 14px;
        font-size: 0.82rem;
        color: #92400E;
        line-height: 1.5;
    }

    /* ปรับแต่งสำหรับหน้าจอมือถือโดยเฉพาะ (Mobile Specific) */
    @media (max-width: 768px) {
        .block-container {
            padding-left: 0.65rem !important;
            padding-right: 0.65rem !important;
            padding-top: 0.75rem !important;
        }
        .app-header {
            padding: 14px 16px;
            border-radius: 12px;
        }
        .app-header-title {
            font-size: 1.15rem;
        }
        .header-badge {
            width: 100%;
            justify-content: center;
        }
        .warning-box {
            padding: 12px 14px;
            font-size: 0.82rem;
        }
        .stat-summary-container {
            grid-template-columns: repeat(3, 1fr) !important;
            gap: 6px !important;
            margin: 12px 0 16px 0 !important;
        }
        .stat-card {
            padding: 10px 7px !important;
            border-radius: 10px !important;
        }
        .stat-card-label {
            font-size: 0.68rem !important;
            gap: 3px !important;
        }
        .stat-card-value {
            font-size: 1.35rem !important;
            margin: 4px 0 2px 0 !important;
        }
        .stat-card-unit {
            font-size: 0.72rem !important;
        }
        .stat-card-sub {
            font-size: 0.68rem !important;
        }
        .stat-card-progress {
            margin-top: 5px !important;
            height: 4px !important;
        }
        .result-card-safe, .result-card-unsafe {
            padding: 14px;
        }
        .stButton > button {
            width: 100% !important;
            padding: 0.65rem 1rem !important;
            font-size: 0.95rem !important;
            border-radius: 8px !important;
        }
        .tech-card div {
            grid-template-columns: 1fr !important;
        }
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# ส่วนหัวเว็บแอปพลิเคชัน (App Header)
# ==============================================================================
st.markdown("""
<div class="app-header">
    <div>
        <div class="app-header-title">
            ระบบประเมินความสดและการเน่าเสียของเนื้อหมู
        </div>
        <div class="app-header-sub">
            Pork Freshness & Spoilage AI Assessment &bull; โครงการระบบจำแนกความปลอดภัยอาหาร
        </div>
    </div>
    <div class="header-badge">
        <span class="status-dot"></span>
        <span>DenseNet201 v2 &bull; ความแม่นยำ 95.0%</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ==============================================================================
# กล่องคำเตือนข้อกำหนดการใช้งานเฉพาะเนื้อหมู (Domain Notice)
# ==============================================================================
st.markdown("""
<div class="warning-box">
    <div style="font-weight: 800; font-size: 0.92rem; margin-bottom: 4px; color: #9A3412;">
        คำเตือนสำคัญ: ระบบนี้ออกแบบสำหรับตรวจ "เฉพาะเนื้อหมูดิบ" เท่านั้น (Raw Pork Meat Only)
    </div>
    <div>
        &bull; <b>เป้าหมายการใช้งาน:</b> ใช้สำหรับชิ้นเนื้อหมูดิบจริง เช่น สันนอก สันใน หมูสามชั้น สะโพก ซี่โครง หรือหมูบด<br>
        &bull; <b>ไม่รองรับภาพบุคคล (Selfie), สัตว์เลี้ยง, หรือสิ่งของทั่วไป:</b> เนื่องจากโมเดลเป็น Binary Classifier ที่ไม่มีคลาสปฏิเสธสิ่งแปลกปลอม หากอัปโหลดภาพคนหรือวัตถุอื่น โมเดลจะยังคงพยายามคำนวณและอาจมองโทนสีผิวของคนเป็นเนื้อหมูสดได้
    </div>
</div>
""", unsafe_allow_html=True)

# ==============================================================================
# การจัดการ Session State
# ==============================================================================
if "analyzed_results" not in st.session_state:
    st.session_state.analyzed_results = None

if "last_analyzed_key" not in st.session_state:
    st.session_state.last_analyzed_key = ""

if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

# ==============================================================================
# ส่วนนำเข้าภาพ (Input Section)
# ==============================================================================
st.markdown("### เลือกวิธีนำเข้าภาพ")
input_mode = st.radio(
    "เลือกแหล่งที่มาของภาพ:",
    options=["อัปโหลดไฟล์รูปภาพ", "ถ่ายภาพด้วยกล้อง", "เลือกจากภาพตัวอย่างในระบบ"],
    index=0,
    horizontal=True,
    label_visibility="collapsed"
)

images_to_process = []

if input_mode == "อัปโหลดไฟล์รูปภาพ":
    st.caption("คำแนะนำ: ควรถ่ายภาพเน้นเฉพาะตัวชิ้นเนื้อหมูดิบเป็นหลัก หลีกเลี่ยงพื้นหลังสีดำทึบ ถาดสีสด หรือสิ่งของประดับ เพื่อผลการวิเคราะห์ที่แม่นยำ")
    uploaded_files = st.file_uploader(
        "แนบรูปภาพเนื้อหมู (เลือกได้ทั้ง 1 รูป หรือพร้อมกันหลายรูป)",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key=f"uploader_{st.session_state.uploader_key}",
        help="ใช้เฉพาะภาพถ่ายชิ้นเนื้อหมูดิบเท่านั้น (ไม่รองรับภาพบุคคลหรือสิ่งของอื่น)"
    )
    if uploaded_files:
        col_up_stat, col_up_clr = st.columns([3.5, 1.5])
        with col_up_stat:
            st.success(f"แนบไฟล์สำเร็จ {len(uploaded_files)} ภาพ (พร้อมสำหรับการตรวจวิเคราะห์)")
        with col_up_clr:
            if st.button("ล้างรูปทั้งหมด (Clear)", key="btn_clear_uploader", use_container_width=True):
                st.session_state.uploader_key += 1
                st.session_state.analyzed_results = None
                st.session_state.last_analyzed_key = ""
                st.rerun()
        for f in uploaded_files:
            try:
                img = Image.open(f)
                images_to_process.append((img, f.name))
            except Exception as e:
                st.error(f"ไม่สามารถเปิดไฟล์ {f.name} ได้: {e}")

elif input_mode == "ถ่ายภาพด้วยกล้อง":
    camera_file = st.camera_input("กดถ่ายภาพเนื้อหมูผ่านกล้อง (เฉพาะชิ้นเนื้อหมูดิบ)", label_visibility="collapsed")
    if camera_file is not None:
        try:
            img = Image.open(camera_file)
            images_to_process.append((img, f"ภาพถ่ายกล้อง_{datetime.now().strftime('%H%M%S')}.png"))
            st.success("บันทึกภาพจากกล้องเรียบร้อย (พร้อมสำหรับการตรวจวิเคราะห์)")
        except Exception as e:
            st.error(f"เกิดข้อผิดพลาดในการโหลดภาพจากกล้อง: {e}")

elif input_mode == "เลือกจากภาพตัวอย่างในระบบ":
    sample_dir = os.path.join(BASE_DIR, "pork_dataset", "unseen data")
    if not os.path.exists(sample_dir):
        sample_dir = os.path.join(BASE_DIR, "..", "dataset", "unseen data")
        
    if os.path.exists(sample_dir):
        sample_files = sorted([f for f in os.listdir(sample_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
        col_s1, col_s2 = st.columns([1.3, 1], gap="medium")
        with col_s1:
            sample_choice = st.selectbox(
                "เลือกภาพตัวอย่างเพื่อทดสอบ:",
                options=[f"{i+1}. {fn}" for i, fn in enumerate(sample_files)],
                index=0
            )
            selected_idx = int(sample_choice.split(".")[0]) - 1
            chosen_file = sample_files[selected_idx]
            chosen_path = os.path.join(sample_dir, chosen_file)
            chosen_img = Image.open(chosen_path)
            images_to_process.append((chosen_img, chosen_file))
            st.success(f"เลือกภาพ: {chosen_file} (พร้อมสำหรับการตรวจวิเคราะห์)")
            st.caption(f"ขนาดภาพต้นฉบับ: {chosen_img.size[0]} x {chosen_img.size[1]} พิกเซล")
        with col_s2:
            st.image(chosen_img, caption=f"พรีวิวตัวอย่าง: {chosen_file}", use_container_width=True)

# ตรวจสอบ Cache Key: หากเปลี่ยนภาพ ให้ล้างผลลัพธ์เก่าเพื่อรอการกดปุ่มวิเคราะห์ใหม่
current_cache_key = f"{input_mode}_{len(images_to_process)}_{'_'.join([n for _, n in images_to_process])}"
if st.session_state.last_analyzed_key != current_cache_key:
    st.session_state.analyzed_results = None

# ==============================================================================
# ฟังก์ชันการประมวลผลโมเดล (Inference)
# ==============================================================================
def run_inference(img: Image.Image, fname: str):
    input_size = MODEL_CONFIG["input_size"]
    img_rgb = img.convert("RGB")
    resized = img_rgb.resize(input_size)
    # ไม่หาร 255.0 ซ้ำ เนื่องจากโมเดล v2 มี Rescaling Layer ภายในโครงสร้างอยู่แล้ว
    arr = np.expand_dims(np.array(resized, dtype=np.float32), axis=0)
    
    t0 = time.time()
    pred_val = float(model.predict(arr, verbose=0)[0][0])
    latency = (time.time() - t0) * 1000
    
    threshold = 0.50
    prob_unsafe = pred_val
    prob_safe = 1.0 - prob_unsafe
    is_unsafe = prob_unsafe >= threshold
    conf = (prob_unsafe if is_unsafe else prob_safe) * 100
    
    return {
        "filename": fname,
        "is_unsafe": is_unsafe,
        "confidence": conf,
        "prob_safe": prob_safe,
        "prob_unsafe": prob_unsafe,
        "latency": latency,
        "image": img
    }

# ==============================================================================
# ปุ่มกดเริ่มตรวจวิเคราะห์
# ==============================================================================
st.markdown("<div style='margin-top: 12px;'></div>", unsafe_allow_html=True)

if not images_to_process:
    st.info("กรุณาแนบไฟล์รูปภาพ ถ่ายภาพ หรือเลือกภาพตัวอย่างก่อน แล้วปุ่มตรวจวิเคราะห์จะเปิดใช้งาน")
    st.button("เริ่มตรวจวิเคราะห์ความสดและการเน่าเสีย", disabled=True, use_container_width=True)
else:
    btn_text = f"กดตรวจวิเคราะห์ ({len(images_to_process)} ภาพ)" if len(images_to_process) > 1 else "กดตรวจวิเคราะห์ภาพนี้"
    col_act1, col_act2 = st.columns([3.5, 1.5])
    with col_act1:
        if st.button(btn_text, type="primary", use_container_width=True):
            with st.spinner("กำลังวิเคราะห์เนื้อหมูด้วย DenseNet201..."):
                results = [run_inference(img, name) for img, name in images_to_process]
                st.session_state.analyzed_results = results
                st.session_state.last_analyzed_key = current_cache_key
    with col_act2:
        if st.button("ล้างข้อมูลทั้งหมด", key="btn_clear_action", use_container_width=True):
            st.session_state.uploader_key += 1
            st.session_state.analyzed_results = None
            st.session_state.last_analyzed_key = ""
            st.rerun()

# ==============================================================================
# การแสดงผลลัพธ์ (หลังกดปุ่มวิเคราะห์เท่านั้น)
# ==============================================================================
if st.session_state.analyzed_results is not None:
    results = st.session_state.analyzed_results
    st.markdown("---")
    st.markdown("### ผลการตรวจวิเคราะห์")
    
    # กรณีตรวจภาพเดี่ยว (Single Image Mode) -> รูปอยู่ซ้าย ข้อมูลอยู่ขวาบนจอคอม และเรียงต่อกันบนมือถือ
    if len(results) == 1:
        res = results[0]
        img_item = res["image"]
        name_item = res["filename"]
        
        col_res1, col_res2 = st.columns([1, 1.25], gap="large")
        
        with col_res1:
            st.image(img_item, caption=f"ภาพที่ตรวจ: {name_item}", use_container_width=True)
            st.caption(f"ขนาดภาพต้นฉบับ: {img_item.size[0]} x {img_item.size[1]} พิกเซล | ประมวลผลใน {res['latency']:.1f} ms")
            
        with col_res2:
            if res["is_unsafe"]:
                st.markdown(f"""
                <div class="result-card-unsafe">
                    <span class="unsafe-badge">เนื้อเริ่มเน่าเสีย (ไม่ควรบริโภค)</span>
                    <div style="font-size: 1.35rem; font-weight: 800; color: #991B1B; margin: 10px 0 6px 0;">
                        เนื้อหมูเริ่มเน่าเสีย ไม่ปลอดภัยต่อการบริโภค
                    </div>
                    <div style="font-size: 0.95rem; color: #450A0A; line-height: 1.55;">
                        ระดับความมั่นใจ: <b>{res['confidence']:.2f}%</b><br>
                        <b>ผลประเมินทางกายภาพ:</b> สีและพื้นผิวของเนื้อหมูมีลักษณะซีด คล้ำ หรือเริ่มมีเมือกจากการย่อยสลายของแบคทีเรีย<br>
                        <b>คำแนะนำด้านสุขอนามัย:</b> <u>ไม่ควรนำมารับประทานหรือประกอบอาหารโดยเด็ดขาด</u> เพื่อป้องกันภาวะอาหารเป็นพิษ แม้จะนำไปผ่านความร้อนแล้ว สารพิษจากแบคทีเรียบางชนิดก็อาจยังคงตกค้างได้
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-card-safe">
                    <span class="safe-badge">สดตามเกณฑ์สี (พร้อมปรุงสุก)</span>
                    <div style="font-size: 1.35rem; font-weight: 800; color: #166534; margin: 10px 0 6px 0;">
                        เนื้อหมูสดตามเกณฑ์สี ไม่พบการเน่าเสีย
                    </div>
                    <div style="font-size: 0.95rem; color: #052E16; line-height: 1.55;">
                        ระดับความมั่นใจ: <b>{res['confidence']:.2f}%</b><br>
                        <b>ผลประเมินทางกายภาพ:</b> สีของเนื้อหมูอยู่ในเกณฑ์สีชมพูสดตามธรรมชาติ สภาพเนื้อสมบูรณ์ ไม่พบการเสื่อมสภาพ<br>
                        <b>คำแนะนำด้านความปลอดภัย:</b> สามารถนำไปประกอบอาหารได้ <u>แต่ต้องปรุงให้สุกด้วยความร้อนสูงทั่วถึงทุกครั้ง</u> เพื่อป้องกันเชื้อโรคหรือพยาธิที่อาจแฝงอยู่ในเนื้อหมูดิบตามธรรมชาติ
                    </div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("**การกระจายค่าน้ำหนักความน่าจะเป็น:**")
            p1, p2 = st.columns(2)
            with p1:
                st.metric("ความสดตามเกณฑ์สี (Freshness)", f"{res['prob_safe']*100:.2f}%")
                st.progress(res["prob_safe"])
            with p2:
                st.metric("ความเสี่ยงเน่าเสีย (Spoilage Risk)", f"{res['prob_unsafe']*100:.2f}%")
                st.progress(res["prob_unsafe"])
                
            # การ์ดรายละเอียดทางเทคนิค
            st.markdown(f"""
            <div class="tech-card">
                <div style="font-weight: 700; font-size: 0.88rem; color: #1E293B; margin-bottom: 8px;">
                    รายละเอียดทางเทคนิค (Technical Details)
                </div>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 6px 14px;">
                    <div>&bull; <b>โมเดลที่ใช้:</b> {MODEL_CONFIG['name']}</div>
                    <div>&bull; <b>เวลาประมวลผล:</b> {res['latency']:.1f} ms</div>
                    <div>&bull; <b>ขนาด Input โมเดล:</b> {MODEL_CONFIG['input_size'][0]} &times; {MODEL_CONFIG['input_size'][1]} px</div>
                    <div>&bull; <b>ขนาดภาพต้นฉบับ:</b> {img_item.size[0]} &times; {img_item.size[1]} px</div>
                    <div>&bull; <b>ค่า Raw Sigmoid:</b> {res['prob_unsafe']:.4f}</div>
                    <div>&bull; <b>เกณฑ์ตัดสินใจ:</b> 0.50</div>
                </div>
            </div>
            
            <div class="scope-box">
                <b>ข้อจำกัดและขอบเขตของโครงงาน (Project Scope & Domain Constraints):</b><br>
                &bull; <b>ขอบเขตวัตถุ:</b> โมเดลรองรับเฉพาะภาพถ่ายชิ้นเนื้อหมูดิบเท่านั้น ไม่มีระบบปฏิเสธภาพทั่วไป (Out-of-Distribution Rejection) หากนำภาพบุคคล สัตว์เลี้ยง หรือสิ่งของอื่นมาตรวจ ระบบจะบังคับจัดเข้ากลุ่มหมูสดหรือหมูเน่าตามโทนสีเสมอ<br>
                &bull; <b>การประเมินทางจุลชีววิทยา:</b> ประเมินจาก <i>ความสดและการเน่าเสียทางกายภาพ (สีและสภาพพื้นผิว)</i> เท่านั้น ไม่ครอบคลุมการตรวจหาเชื้อโรคระดับจุลชีววิทยาเชิงลึก (เช่น แบคทีเรียสเตรปโตคอกคัส ซูอิส) หรือพยาธิที่มองไม่เห็นด้วยตาเปล่า เนื้อหมูดิบสดต้องผ่านการปรุงสุกด้วยความร้อนสูงทุกครั้งก่อนรับประทาน
            </div>
            """, unsafe_allow_html=True)

    # กรณีตรวจหลายภาพพร้อมกัน (Batch Mode)
    else:
        safe_count = sum(1 for r in results if not r["is_unsafe"])
        unsafe_count = len(results) - safe_count
        total_count = len(results)
        safe_pct = (safe_count / total_count) * 100 if total_count > 0 else 0
        unsafe_pct = (unsafe_count / total_count) * 100 if total_count > 0 else 0
        
        stat_html = (
            f'<div class="stat-summary-container">'
            f'<div class="stat-card total">'
            f'<div class="stat-card-label"><span>&bull;</span> ตรวจทั้งหมด</div>'
            f'<div class="stat-card-value">{total_count}<span class="stat-card-unit">ภาพ</span></div>'
            f'<div class="stat-card-sub">เสร็จสิ้น 100%</div>'
            f'<div class="stat-card-progress"><div class="stat-card-fill" style="width: 100%;"></div></div>'
            f'</div>'
            f'<div class="stat-card safe">'
            f'<div class="stat-card-label"><span>&bull;</span> สดตามเกณฑ์สี</div>'
            f'<div class="stat-card-value">{safe_count}<span class="stat-card-unit">ภาพ</span></div>'
            f'<div class="stat-card-sub">{safe_pct:.1f}% ของทั้งหมด</div>'
            f'<div class="stat-card-progress"><div class="stat-card-fill" style="width: {safe_pct:.1f}%;"></div></div>'
            f'</div>'
            f'<div class="stat-card unsafe">'
            f'<div class="stat-card-label"><span>&bull;</span> เริ่มเน่าเสีย</div>'
            f'<div class="stat-card-value">{unsafe_count}<span class="stat-card-unit">ภาพ</span></div>'
            f'<div class="stat-card-sub">{unsafe_pct:.1f}% ของทั้งหมด</div>'
            f'<div class="stat-card-progress"><div class="stat-card-fill" style="width: {unsafe_pct:.1f}%;"></div></div>'
            f'</div>'
            f'</div>'
        )
        st.markdown(stat_html, unsafe_allow_html=True)
        
        st.markdown("<div style='font-size: 1.05rem; font-weight: 700; color: #0F172A; margin: 12px 0 8px 0;'>รายละเอียดผลการตรวจแต่ละภาพ:</div>", unsafe_allow_html=True)
        tab_cards, tab_table = st.tabs(["แกลเลอรีการ์ด (Card Gallery)", "ตารางสรุปผลทั้งหมด (Summary Table)"])
        
        with tab_cards:
            cols_per_row = 3
            for row_idx in range(0, len(results), cols_per_row):
                cols = st.columns(cols_per_row, gap="medium")
                for c_i in range(cols_per_row):
                    i = row_idx + c_i
                    if i < len(results):
                        r = results[i]
                        with cols[c_i]:
                            with st.container(border=True):
                                # Thumbnail ขนาดมาตรฐานเท่ากันทุกรูป (สัดส่วน 16:10 ไม่ยืด ไม่เตี้ย ไม่ดึงจอ)
                                thumb = ImageOps.fit(r["image"], (400, 250), method=Image.Resampling.LANCZOS)
                                st.image(thumb, use_container_width=True)
                                
                                if r["is_unsafe"]:
                                    st.markdown(
                                        f'<div style="display: flex; justify-content: space-between; align-items: center; margin-top: 6px;">'
                                        f'<span class="unsafe-badge">เริ่มเน่าเสีย ({r["confidence"]:.1f}%)</span>'
                                        f'<span style="font-size: 0.75rem; font-weight: 600; color: #DC2626;">เสี่ยง {r["prob_unsafe"]*100:.0f}%</span>'
                                        f'</div>',
                                        unsafe_allow_html=True
                                    )
                                else:
                                    st.markdown(
                                        f'<div style="display: flex; justify-content: space-between; align-items: center; margin-top: 6px;">'
                                        f'<span class="safe-badge">สดตามเกณฑ์ ({r["confidence"]:.1f}%)</span>'
                                        f'<span style="font-size: 0.75rem; font-weight: 600; color: #16A34A;">สด {r["prob_safe"]*100:.0f}%</span>'
                                        f'</div>',
                                        unsafe_allow_html=True
                                    )
                                
                                st.progress(r["prob_unsafe"])
                                
                                fname_short = r['filename'][:18] + '...' if len(r['filename']) > 20 else r['filename']
                                st.markdown(
                                    f'<div style="display: flex; justify-content: space-between; align-items: center; background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; padding: 4px 8px; font-size: 0.74rem; color: #475569; margin-top: 6px;">'
                                    f'<span style="font-weight: 600; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;" title="{r["filename"]}">{fname_short}</span>'
                                    f'<span style="color: #64748B; font-size: 0.7rem;">{r["latency"]:.0f} ms</span>'
                                    f'</div>',
                                    unsafe_allow_html=True
                                )
        
        with tab_table:
            table_data = []
            for idx, r in enumerate(results, 1):
                table_data.append({
                    "ลำดับ": idx,
                    "ชื่อไฟล์": r["filename"],
                    "ผลการประเมิน": "เนื้อเริ่มเน่าเสีย (ไม่ควรบริโภค)" if r["is_unsafe"] else "สดตามเกณฑ์สี (พร้อมปรุงสุก)",
                    "ความมั่นใจ": f"{r['confidence']:.1f}%",
                    "ความสด (Freshness)": f"{r['prob_safe']*100:.1f}%",
                    "ความเสี่ยงเน่า (Spoilage)": f"{r['prob_unsafe']*100:.1f}%",
                    "เวลาประมวลผล": f"{r['latency']:.1f} ms",
                    "Raw Sigmoid": f"{r['prob_unsafe']:.4f}"
                })
            df_batch = pd.DataFrame(table_data)
            st.dataframe(df_batch, use_container_width=True, hide_index=True)
            
        st.markdown("""
        <div class="scope-box" style="margin-top: 14px;">
            <b>ข้อจำกัดและขอบเขตของโครงงาน (Project Scope & Domain Constraints):</b><br>
            &bull; <b>ขอบเขตวัตถุ:</b> โมเดลรองรับเฉพาะภาพถ่ายชิ้นเนื้อหมูดิบเท่านั้น ไม่มีระบบปฏิเสธภาพทั่วไป (Out-of-Distribution Rejection) หากนำภาพบุคคล สัตว์เลี้ยง หรือสิ่งของอื่นมาตรวจ ระบบจะบังคับจัดเข้ากลุ่มหมูสดหรือหมูเน่าตามโทนสีเสมอ<br>
            &bull; <b>การประเมินทางจุลชีววิทยา:</b> ประเมินจาก <i>ความสดและการเน่าเสียทางกายภาพ (สีและสภาพพื้นผิว)</i> เท่านั้น ไม่ครอบคลุมการตรวจหาเชื้อโรคระดับจุลชีววิทยาเชิงลึก หรือพยาธิที่มองไม่เห็นด้วยตาเปล่า เนื้อหมูดิบสดต้องผ่านการปรุงสุกด้วยความร้อนสูงทุกครั้งก่อนรับประทาน
        </div>
        """, unsafe_allow_html=True)
