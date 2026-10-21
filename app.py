import os
import io
import json
import time
import zipfile
from pathlib import Path
from PIL import Image
import streamlit as st
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

# ----------------- 頁面基本配置 -----------------
st.set_page_config(
    page_title="AI 智慧多比例攝影裁切神器",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 注入卡片式區塊 CSS
st.markdown("""
<style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1050px;
    }
    
    /* 標題與簡介 */
    .hero-container {
        background: linear-gradient(135deg, #f0f7ff 0%, #fdfbf7 100%);
        border: 1px solid #dbeafe;
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 24px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.03);
    }
    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1e40af;
        margin-bottom: 6px;
    }
    .hero-subtitle {
        color: #4b5563;
        font-size: 1.05rem;
        margin: 0;
    }

    /* 區塊卡片外框 */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #ffffff;
        border-radius: 14px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
        padding: 12px;
        margin-bottom: 12px;
    }

    /* 尺寸小卡片說明文字 */
    .ratio-desc {
        color: #6b7280;
        font-size: 0.82rem;
        line-height: 1.3;
        margin-top: -6px;
        min-height: 38px;
    }
</style>
""", unsafe_allow_html=True)

# 讀取後台設定的 API Key
api_key = st.secrets.get("GEMINI_API_KEY", os.environ.get("GEMINI_API_KEY"))
client = genai.Client(api_key=api_key) if api_key else None

class GroupSubjectAnalysis(BaseModel):
    subject_category: str = Field(description="Dominant category: 'people', 'animal', 'architecture', 'vehicle'")
    group_box_2d: list[int] = Field(description="Bounding box of main group [ymin, xmin, ymax, xmax] scaled 0 to 1000")
    face_top_y: int = Field(description="Highest point of faces/heads (0-1000)")

def get_optimized_ai_input(img: Image.Image) -> Image.Image:
    """縮小圖片尺寸以加快上傳與分析"""
    max_dim = 1024
    w, h = img.size
    if max(w, h) > max_dim:
        scale = max_dim / max(w, h)
        return img.resize((int(w * scale), int(h * scale)), Image.Resampling.BILINEAR)
    return img

def analyze_all_subjects(pil_img: Image.Image):
    """呼叫 Gemini 進行主角群體判定與透氣留白分析"""
    if not client:
        return {"subject_category": "people", "group_box_2d": [250, 200, 950, 950], "face_top_y": 380}

    ai_img = get_optimized_ai_input(pil_img)
    prompt = (
        "You are an expert director of photography. Analyze composition.\n"
        "1. If humans present, return 'group_box_2d' [ymin, xmin, ymax, xmax] (0-1000) enclosing ALL persons.\n"
        "2. Identify 'face_top_y' (0-1000).\n"
        "3. If no humans, apply to animal > architecture > vehicle.\n"
        "Output JSON matching schema."
    )

    for attempt in range(2):
        try:
            response = client.models.generate_content(
                model="gemini-3.8-flash",
                contents=[ai_img, prompt],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=GroupSubjectAnalysis,
                    temperature=0.1
                ),
            )
            return json.loads(response.text)
        except Exception:
            if attempt == 0:
                time.sleep(1.5)
                continue
            return {"subject_category": "people", "group_box_2d": [250, 200, 950, 950], "face_top_y": 380}

def calculate_adaptive_crop(img_size: tuple[int, int], target_ratio: float, info: dict) -> tuple[int, int, int, int]:
    """計算自適應構圖裁切框，嚴防切斷主角群"""
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

# ----------------- 介面排版 -----------------

# 頂部 Hero 卡片
st.markdown("""
<div class="hero-container">
    <div class="hero-title">✨ AI 智慧多比例攝影裁切神器</div>
    <p class="hero-subtitle">智能鎖定人物主角群體，自動套用最佳攝影構圖，一鍵批量產出多種社群完美比例！</p>
</div>
""", unsafe_allow_html=True)

if not client:
    st.warning("⚠️ 系統尚未偵測到 GEMINI_API_KEY，請在雲端後台 Settings ➔ Secrets 中進行設定。")

# 狀態管理函式：解決全選/清除失靈問題
RATIO_KEYS = ["cb_16_9", "cb_9_16", "cb_3_4", "cb_4_3", "cb_1_1"]
for k in RATIO_KEYS:
    if k not in st.session_state:
        st.session_state[k] = True

def set_all_checkboxes(status: bool):
    for k in RATIO_KEYS:
        st.session_state[k] = status

# 區塊一：尺寸規格設定卡片
with st.container(border=True):
    col_t1, col_t2 = st.columns([6, 4])
    with col_t1:
        st.markdown("#### 📌 第一步：勾選你想產出的尺寸規格")
    with col_t2:
        c_btn1, c_btn2 = st.columns(2)
        with c_btn1:
            st.button("全選尺寸", on_click=set_all_checkboxes, args=(True,), use_container_width=True)
        with c_btn2:
            st.button("清除勾選", on_click=set_all_checkboxes, args=(False,), use_container_width=True)

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

# 區塊二：照片上傳卡片
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

# 區塊三：執行與結果卡片
if uploaded_files and not has_file_count_error:
    with st.container(border=True):
        st.markdown("#### 🚀 第三步：開始產出")
        if not selected_ratios:
            st.warning("⚠️ 請在上方第一步中至少勾選一種你想裁切的尺寸規格喔！")
        else:
            if st.button("開始 AI 智慧構圖與極速裁切", type="primary", use_container_width=True):
                zip_buffer = io.BytesIO()
                preview_list = []
                
                progress_bar = st.progress(0)
                status_box = st.empty()

                total_files = len(uploaded_files)
                with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
                    for idx, file in enumerate(uploaded_files):
                        status_box.markdown(f"🤖 **正在深度分析第 {idx + 1}/{total_files} 張照片的主角與空間佈局...** `{file.name}`")
                        
                        img = Image.open(file)
                        base_name = Path(file.name).stem
                        info = analyze_all_subjects(img)

                        for label, ratio in selected_ratios.items():
                            l, t, r, b = calculate_adaptive_crop(img.size, ratio, info)
                            cropped = img.crop((l, t, r, b))
                            
                            if cropped.mode in ("RGBA", "P"):
                                cropped = cropped.convert("RGB")

                            img_byte_arr = io.BytesIO()
                            cropped.save(img_byte_arr, format='JPEG', quality=92, optimize=True)
                            filename = f"{base_name}_{label}.jpg"
                            zip_file.writestr(filename, img_byte_arr.getvalue())
                            preview_list.append((cropped, filename))

                        progress_bar.progress((idx + 1) / total_files)

                status_box.empty()
                st.success(f"🎉 全部處理完成！共為你產出 **{len(preview_list)}** 張自適應裁切照片。")

                st.download_button(
                    label="📥 一鍵下載所有裁切成果包 (.ZIP)",
                    data=zip_buffer.getvalue(),
                    file_name="smart_cropped_photos.zip",
                    mime="application/zip",
                    use_container_width=True
                )

                st.markdown("##### 🖼️ 裁切成果預覽")
                cols_preview = st.columns(3)
                for idx, (c_img, title) in enumerate(preview_list):
                    with cols_preview[idx % 3]:
                        st.image(c_img, caption=title, use_container_width=True)
