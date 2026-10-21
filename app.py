import os
import io
import json
import time
import zipfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image
import streamlit as st
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

# ----------------- 頁面基本配置 -----------------
st.set_page_config(
    page_title="AI 智慧多比例攝影裁切神器",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 響應式 CSS
st.markdown("""
<style>
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        max-width: 1050px;
    }
    .hero-container {
        background: linear-gradient(135deg, #eff6ff 0%, #fdfbf7 100%);
        border: 1px solid #bfdbfe;
        border-radius: 16px;
        padding: 20px 24px;
        margin-bottom: 20px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.03);
    }
    .hero-title {
        font-size: 2.1rem;
        font-weight: 800;
        color: #1d4ed8;
        margin-bottom: 6px;
        line-height: 1.2;
    }
    .hero-subtitle {
        color: #4b5563;
        font-size: 1rem;
        margin: 0;
        line-height: 1.5;
    }
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #ffffff;
        border-radius: 14px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
        padding: 14px;
        margin-bottom: 14px;
    }
    .ratio-desc {
        color: #6b7280;
        font-size: 0.8rem;
        line-height: 1.3;
        margin-top: -6px;
        margin-bottom: 10px;
    }
    @media (max-width: 768px) {
        .block-container {
            padding-left: 0.8rem !important;
            padding-right: 0.8rem !important;
            padding-top: 1rem !important;
        }
        .hero-title { font-size: 1.4rem; }
        .hero-subtitle { font-size: 0.88rem; }
        div[data-testid="stHorizontalBlock"] {
            flex-wrap: wrap !important;
            gap: 8px !important;
        }
        div[data-testid="stColumn"] {
            min-width: 100% !important;
            flex: 1 1 100% !important;
        }
    }
</style>
""", unsafe_allow_html=True)

api_key = st.secrets.get("GEMINI_API_KEY", os.environ.get("GEMINI_API_KEY"))
client = genai.Client(api_key=api_key) if api_key else None

class GroupSubjectAnalysis(BaseModel):
    subject_category: str = Field(description="Dominant category: 'people', 'animal', 'architecture', 'vehicle'")
    group_box_2d: list[int] = Field(description="Bounding box of main group [ymin, xmin, ymax, xmax] scaled 0 to 1000")
    face_top_y: int = Field(description="Highest point of faces/heads (0-1000)")

def get_ultra_low_res_ai_input(img: Image.Image) -> Image.Image:
    """縮小至最長邊 512px，使單張上傳體積降至數十 KB，大幅提升速度"""
    max_dim = 512
    w, h = img.size
    if max(w, h) > max_dim:
        scale = max_dim / max(w, h)
        return img.resize((int(w * scale), int(h * scale)), Image.Resampling.BILINEAR)
    return img

def analyze_single_image(pil_img: Image.Image) -> dict:
    """單張圖片 AI 分析，容錯高且穩定"""
    default_res = {"subject_category": "people", "group_box_2d": [250, 200, 950, 950], "face_top_y": 380}
    if not client:
        return default_res

    low_img = get_ultra_low_res_ai_input(pil_img)
    prompt = (
        "You are an expert photography director. Analyze composition.\n"
        "1. If humans present, return 'group_box_2d' [ymin, xmin, ymax, xmax] (0-1000) enclosing ALL persons.\n"
        "2. Identify 'face_top_y' (0-1000).\n"
        "3. If no humans, apply to animal > architecture > vehicle.\n"
        "Output JSON matching schema."
    )

    try:
        response = client.models.generate_content(
            model="gemini-3.8-flash",
            contents=[low_img, prompt],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=GroupSubjectAnalysis,
                temperature=0.1
            ),
        )
        return json.loads(response.text)
    except Exception:
        return default_res

def calculate_adaptive_crop(img_size: tuple[int, int], target_ratio: float, info: dict) -> tuple[int, int, int, int]:
    orig_w, orig_h = img_size
    orig_ratio = orig_w / orig_h

    ymin, xmin, ymax, xmax = info["group_box_2d"]
    face_top = info.get("face_top_y", ymin)

    s_top = (ymin / 1000.0) * orig_h
    s_bottom = (ymax / 1000.0) * orig_h
    s_left = (xmin / 1000.0) * orig_w
    s_right = (xmax / 1000.0) * orig_w
    face_top_px = (face_top / 1000.0) * orig_h

    s_cx = (s_left + s_right) / 2.0
    s_w = s_right - s_left

    if orig_ratio > target_ratio:
        crop_h = orig_h
        crop_w = crop_h * target_ratio
    else:
        crop_w = orig_w
        crop_h = crop_w / target_ratio

    left = s_cx - (crop_w / 2.0)
    right = left + crop_w

    if crop_w >= s_w:
        if left > s_left:
            left = s_left - 20
            right = left + crop_w
        if right < s_right:
            right = s_right + 20
            left = right - crop_w

    if target_ratio > 1.3:
        top = face_top_px - (crop_h * 0.20)
    elif target_ratio < 0.7:
        top = face_top_px - (crop_h * 0.35)
    else:
        top = face_top_px - (crop_h * 0.25)

    bottom = top + crop_h

    if crop_h >= (s_bottom - face_top_px):
        if top > face_top_px:
            top = face_top_px - 20
            bottom = top + crop_h
        if bottom < s_bottom:
            bottom = s_bottom + 20
            top = bottom - crop_h

    if left < 0:
        right += -left
        left = 0
    if right > orig_w:
        left -= (right - orig_w)
        right = orig_w
    if top < 0:
        bottom += -top
        top = 0
    if bottom > orig_h:
        bottom = orig_h

    return int(left), int(top), int(right), int(bottom)

def process_single_task(raw_bytes: bytes, file_name: str, target_ratios: dict):
    """單一圖片任務：AI 解析 + 多尺寸裁切"""
    with Image.open(io.BytesIO(raw_bytes)) as img:
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")
        
        base_name = Path(file_name).stem
        info = analyze_single_image(img)
        
        results = []
        for label, ratio in target_ratios.items():
            l, t, r, b = calculate_adaptive_crop(img.size, ratio, info)
            c = img.crop((l, t, r, b))
            
            buf = io.BytesIO()
            c.save(buf, format='JPEG', quality=88)
            filename = f"{base_name}_{label}.jpg"
            results.append((filename, buf.getvalue(), c))
            
        return file_name, results

# ----------------- 介面排版 -----------------

st.markdown("""
<div class="hero-container">
    <div class="hero-title">⚡ AI 智慧多比例攝影裁切神器</div>
    <p class="hero-subtitle">智能鎖定人物主角群體，極速多核心並行構圖裁切，全面支援手機、平板與電腦！</p>
</div>
""", unsafe_allow_html=True)

if not client:
    st.warning("⚠️ 系統尚未偵測到 GEMINI_API_KEY，請在雲端後台 Settings ➔ Secrets 中進行設定。")

RATIO_KEYS = ["cb_16_9", "cb_9_16", "cb_3_4", "cb_4_3", "cb_1_1"]
for k in RATIO_KEYS:
    if k not in st.session_state:
        st.session_state[k] = True

def set_all_checkboxes(status: bool):
    for k in RATIO_KEYS:
        st.session_state[k] = status

with st.container(border=True):
    col_t1, col_t2 = st.columns([6, 4])
    with col_t1:
        st.markdown("#### 📌 第一步：勾選你想產出的尺寸規格")
    with col_t2:
        c_btn1, c_btn2 = st.columns(2)
        with c_btn1:
            st.button("全選尺寸", on_click=set_all_checkboxes, args=(True,), width="stretch")
        with c_btn2:
            st.button("清除勾選", on_click=set_all_checkboxes, args=(False,), width="stretch")

    cols = st.columns(5)
    selected_ratios = {}

    with cols[0]:
        c1 = st.checkbox("16:9 橫幅寬螢幕", key="cb_16_9")
        st.markdown('<p class="ratio-desc">適用 YouTube / 簡報 / 電腦桌布</p>', unsafe_allow_html=True)
        if c1: selected_ratios["16-9"] = 16 / 9

    with cols[1]:
        c2 = st.checkbox("9:16 直幅限動", key="cb_9_16")
        st.markdown('<p class="ratio-desc">適用 IG Reels / 短影音 / 限時動態</p>', unsafe_allow_html=True)
        if c2: selected_ratios["9-16"] = 9 / 16

    with cols[2]:
        c3 = st.checkbox("3:4 人像直幅", key="cb_3_4")
        st.markdown('<p class="ratio-desc">適用 IG 直幅貼文 / 手機閱讀</p>', unsafe_allow_html=True)
        if c3: selected_ratios["3-4"] = 3 / 4

    with cols[3]:
        c4 = st.checkbox("4:3 標準橫幅", key="cb_4_3")
        st.markdown('<p class="ratio-desc">經典相機規格 / 部落格文章首圖</p>', unsafe_allow_html=True)
        if c4: selected_ratios["4-3"] = 4 / 3

    with cols[4]:
        c5 = st.checkbox("1:1 正方形", key="cb_1_1")
        st.markdown('<p class="ratio-desc">適用 社群頭貼 / 經典方形貼文</p>', unsafe_allow_html=True)
        if c5: selected_ratios["1-1"] = 1.0

with st.container(border=True):
    st.markdown("#### 📁 第二步：上傳照片（上限 10 張）")
    uploaded_files = st.file_uploader(
        "拖曳或點選上傳（支援 JPG、PNG、WEBP）：", 
        type=["jpg", "jpeg", "png", "webp"], 
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

    MAX_FILES = 10
    has_file_count_error = False

    if uploaded_files:
        if len(uploaded_files) > MAX_FILES:
            st.error(f"⚠️ 為確保雲端運作速度與伺服器穩定，一次最多只能處理 {MAX_FILES} 張照片！目前已選取 {len(uploaded_files)} 張，請刪除多餘照片再試。")
            has_file_count_error = True
        else:
            st.success(f"✨ 已成功載入 **{len(uploaded_files)}** 張照片，已準備好進行處理。")

if uploaded_files and not has_file_count_error:
    with st.container(border=True):
        st.markdown("#### 🚀 第三步：極速產出")
        if not selected_ratios:
            st.warning("⚠️ 請在上方第一步中至少勾選一種你想裁切的尺寸規格喔！")
        else:
            if st.button("⚡ 開始極速並行裁切", type="primary", width="stretch"):
                zip_buffer = io.BytesIO()
                preview_list = []

                progress_bar = st.progress(0.0)
                status_box = st.empty()
                status_box.info("🚀 啟動多核心極速處理中...")

                t_start = time.time()
                total_files = len(uploaded_files)
                completed_count = 0

                # 讀取檔案二進位數據
                file_items = [(f.read(), f.name) for f in uploaded_files]

                # 4 執行緒平行運算
                max_workers = min(4, total_files)
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = [executor.submit(process_single_task, raw_bytes, name, selected_ratios) for raw_bytes, name in file_items]

                    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
                        for future in as_completed(futures):
                            file_name, items = future.result()
                            completed_count += 1

                            for filename, img_bytes, preview_img in items:
                                zip_file.writestr(filename, img_bytes)
                                preview_list.append((preview_img, filename))

                            progress = completed_count / total_files
                            progress_bar.progress(progress)
                            status_box.markdown(f"⏳ **已完成 {completed_count} / {total_files} 張**：`{file_name}`")

                elapsed = time.time() - t_start
                progress_bar.empty()
                status_box.empty()
                st.success(f"⚡ 處理完成！共生成 **{len(preview_list)}** 張照片（耗時 {elapsed:.1f} 秒）。")

                st.download_button(
                    label="📥 一鍵下載所有裁切成果包 (.ZIP)",
                    data=zip_buffer.getvalue(),
                    file_name="smart_cropped_photos.zip",
                    mime="application/zip",
                    width="stretch"
                )

                st.markdown("##### 🖼️ 裁切成果預覽")
                cols_preview = st.columns(3)
                for idx, (c_img, title) in enumerate(preview_list):
                    with cols_preview[idx % 3]:
                        st.image(c_img, caption=title, width="stretch")
