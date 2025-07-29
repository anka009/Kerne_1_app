import streamlit as st
from streamlit_drawable_canvas import st_canvas
import cv2
import numpy as np
import io
from PIL import Image, ImageEnhance

# 🔧 App-Konfiguration
st.set_page_config(page_title="🧽 Bildbearbeitung", layout="centered")
st.title("🧽 Bildbearbeitung mit Canvas & Reglern")

# 📤 Upload
uploaded_file = st.file_uploader("📤 Bild hochladen", type=["jpg", "jpeg", "png", "tif", "tiff"])

if uploaded_file:
    # 🖼️ Öffnen & Konvertieren
    pil_img = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(pil_img)

    # 🎨 Canvas-Setup ohne Fehlerquelle
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    buf.seek(0)
    background_image = Image.open(buf)

    canvas_result = st_canvas(
        fill_color="rgba(255, 0, 0, 0.3)",
        stroke_width=2,
        background_image=background_image,
        update_streamlit=True,
        height=pil_img.height,
        width=pil_img.width,
        drawing_mode="rect",
        key="canvas_key"
    )

    # 🎯 ZOI auslesen & anzeigen
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

    # 🎛️ Sidebar-Regler
    st.sidebar.subheader("🎚 Bildregler")
    brightness = st.sidebar.slider("🌞 Helligkeit", -100, 100, 0)
    contrast = st.sidebar.slider("🌗 Kontrast", -100, 100, 0)
    saturation = st.sidebar.slider("🌈 Sättigung", 0.0, 3.0, 1.0)
    min_gray = st.sidebar.slider("🖤 Min. Grauintensität", 0, 255, 0)

    # 🧠 Verarbeitung: Helligkeit + Kontrast
    image_proc = cv2.convertScaleAbs(image_np, alpha=1 + contrast / 100.0, beta=brightness)

    # 🌈 Sättigung mit PIL
    image_pil = Image.fromarray(image_proc)
    image_pil = ImageEnhance.Color(image_pil).enhance(saturation)
    image_proc = np.array(image_pil)

    # 🖤 Graustufen & Filter
    gray_image = cv2.cvtColor(image_proc, cv2.COLOR_RGB2GRAY)
    filtered_gray = cv2.threshold(gray_image, min_gray, 255, cv2.THRESH_BINARY)[1]

    # 🖼️ Ergebnisanzeigen
    st.image(image_proc, caption="🎨 Bearbeitetes Bild")
    st.image(filtered_gray, caption="🖤 Graufilter-Ergebnis", channels="GRAY")
