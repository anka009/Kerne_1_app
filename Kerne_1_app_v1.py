import streamlit as st
from streamlit_drawable_canvas import st_canvas
import cv2
import numpy as np
import io
from PIL import Image, ImageEnhance

# 🧾 Seitenlayout
st.set_page_config(page_title="🧽 Bildbearbeitung", layout="centered")
st.title("🧼 Bildbearbeitung mit ZOI, Reglern & Canvas")

# 📤 Einmaliger Upload
uploaded_file = st.file_uploader("📤 Bild hochladen", type=["jpg", "jpeg", "png", "tif", "tiff"])

if uploaded_file:
    # 🖼️ PIL-Image laden und konvertieren
    pil_img = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(pil_img)

    # 🧼 Buffer für Canvas (ohne fehlerhafte image_to_url)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    buf.seek(0)
    canvas_result = st_canvas(
        fill_color="rgba(255, 0, 0, 0.3)",
        stroke_width=2,
        background_image=None,  # ✅ Kein Fehler durch image_to_url
        update_streamlit=True,
        height=pil_img.height,
        width=pil_img.width,
        drawing_mode="rect",
        key="canvas_key"
    )

    # 🎯 ZOI auslesen, wenn vorhanden
    if canvas_result.json_data:
        objects = canvas_result.json_data["objects"]
        if objects:
            obj = objects[-1]
            x, y = int(obj["left"]), int(obj["top"])
            w, h = int(obj["width"]), int(obj["height"])
            roi = image_np[y:y+h, x:x+w]
            st.image(roi, caption="🎯 Ausgewählte Zone of Interest (ZOI)")

    # 📸 Originalbild anzeigen
    st.image(image_np, caption="📷 Originalbild")

    # 🎚️ Sidebar-Regler
    st.sidebar.subheader("🎛 Bildregler")
    brightness = st.sidebar.slider("🌞 Helligkeit", -100, 100, 0)
    contrast = st.sidebar.slider("🌗 Kontrast", -100, 100, 0)
    saturation = st.sidebar.slider("🌈 Sättigung", 0.0, 3.0, 1.0)
    min_gray = st.sidebar.slider("🖤 Graufilter ab Intensität", 0, 255, 0)

    # ⚙️ Helligkeit & Kontrast mit OpenCV
    image_proc = cv2.convertScaleAbs(image_np, alpha=1 + contrast / 100.0, beta=brightness)

    # 🎨 Sättigung mit PIL
    pil_proc = Image.fromarray(image_proc)
    pil_proc = ImageEnhance.Color(pil_proc).enhance(saturation)
    image_proc = np.array(pil_proc)

    # 🧊 Graustufen + Schwellenfilter
    gray = cv2.cvtColor(image_proc, cv2.COLOR_RGB2GRAY)
    filtered_gray = cv2.threshold(gray, min_gray, 255, cv2.THRESH_BINARY)[1]

    # 🖼️ Ergebnis anzeigen
    st.image(image_proc, caption="🎨 Bild nach Regler")
    st.image(filtered_gray, caption="🖤 Graufilter-Ergebnis", channels="GRAY")
