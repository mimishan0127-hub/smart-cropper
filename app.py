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

# 頁面基本配置
st.set_page_config(page_title="AI 智慧多比例攝影裁切器", layout="wide", page_icon="📷")

# 讀取後台設定的 API Key
api_key = st.secrets.get("GEMINI_API_KEY", os.environ.get("GEMINI_API_KEY"))
client = genai.Client(api_key=api_key) if api_key else None

class GroupSubjectAnalysis(BaseModel):
    subject_category: str = Field(description="Dominant category: 'people', 'animal', 'architecture', 'vehicle'")
    group_box_2d: list[int] = Field(description="Bounding box of main group [ymin, xmin, ymax, xmax] scaled 0 to 1000")
    face_top_y: int = Field(description="Highest point of faces/heads (0-1000)")

def get_optimized_ai_input(img: Image.Image) -> Image.Image:
    """將原圖縮小上傳給 AI 分析，大幅節省頻寬與等待時間"""
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
    """計算最佳裁切座標，保護主角群體完整不被裁切"""
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
        top -= (bottom - orig_h)
        bottom = orig_h

    return int(left), int(top), int(right), int(bottom)

# ----------------- 前端 UI 設計 -----------------
st.title("📷 AI 智慧攝影多比例裁切工具")
st.write("上傳照片後，AI 會自動鎖定主要人物與主角群，並依攝影美學進行智慧裁切。")

if not client:
    st.warning("⚠️ 系統尚未偵測到 GEMINI_API_KEY，請在雲端後台 Secrets 中進行設定。")

# 所有支援的尺寸比例映射
ALL_RATIOS = {
    "16:9 (橫幅寬螢幕)": ("16-9", 16 / 9),
    "9:16 (直幅限動/Reels)": ("9-16", 9 / 16),
    "3:4 (直幅人像)": ("3-4", 3 / 4),
    "4:3 (標準橫幅)": ("4-3", 4 / 3),
    "1:1 (正方形)": ("1-1", 1.0)
}

# 1. 尺寸複選功能
selected_labels = st.multiselect(
    "📐 請勾選欲輸出的裁切尺寸（可複選）：",
    options=list(ALL_RATIOS.keys()),
    default=list(ALL_RATIOS.keys())  # 預設全選
)

# 2. 上傳照片
uploaded_files = st.file_uploader(
    "📁 選擇或拖曳照片上傳（支援 JPG、PNG、WEBP）：", 
    type=["jpg", "jpeg", "png", "webp"], 
    accept_multiple_files=True
)

# 檢查張數限制（最多 10 張）
MAX_FILES = 10
has_file_count_error = False

if uploaded_files:
    if len(uploaded_files) > MAX_FILES:
        st.error(f"⚠️ 為維護系統效能與處理速度，一次最多只能上傳 {MAX_FILES} 張照片！目前已選取 {len(uploaded_files)} 張，請刪除部分照片後再試。")
        has_file_count_error = True
    else:
        st.info(f"已選取 {len(uploaded_files)} 張照片。")

# 3. 執行按鈕
if uploaded_files and not has_file_count_error:
    if not selected_labels:
        st.warning("⚠️ 請至少勾選一種裁切尺寸！")
    else:
        if st.button("🚀 開始 AI 智慧分析與裁切", type="primary"):
            target_ratios = {ALL_RATIOS[label][0]: ALL_RATIOS[label][1] for label in selected_labels}

            zip_buffer = io.BytesIO()
            preview_list = []
            progress_bar = st.progress(0)
            status_text = st.empty()

            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
                total_files = len(uploaded_files)
                for idx, file in enumerate(uploaded_files):
                    status_text.text(f"⏳ 正在分析第 {idx + 1}/{total_files} 張照片的主角與構圖：{file.name}...")
                    
                    img = Image.open(file)
                    base_name = Path(file.name).stem
                    info = analyze_all_subjects(img)

                    for label, ratio in target_ratios.items():
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

            status_text.empty()
            st.success(f"🎉 全部處理完成！共生成 {len(preview_list)} 張照片。")

            # 下載壓縮檔
            st.download_button(
                label="📦 下載全部裁切打包檔 (.ZIP)",
                data=zip_buffer.getvalue(),
                file_name="all_cropped_photos.zip",
                mime="application/zip"
            )

            # 即時預覽成果
            st.subheader("🖼️ 成果預覽")
            cols = st.columns(3)
            for idx, (c_img, title) in enumerate(preview_list):
                with cols[idx % 3]:
                    st.image(c_img, caption=title, use_container_width=True)
