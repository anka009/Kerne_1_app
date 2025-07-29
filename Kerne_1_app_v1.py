from streamlit_drawable_canvas import st_canvas
import streamlit as st
import cv2
import numpy as np
import io
from PIL import Image

uploaded_file = st.file_uploader("Bild hochladen", type=["jpg", "jpeg", "png", "tif", "tiff"])
if uploaded_file:
    pil_img = Image.open(uploaded_file).convert("RGB")

    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    byte_img = buf.getvalue()

    canvas_result = st_canvas(
        fill_color="rgba(255, 0, 0, 0.3)",
        stroke_width=2,
        background_image=Image.open(io.BytesIO(byte_img)),
        update_streamlit=True,
        height=pil_img.height,
        width=pil_img.width,
        drawing_mode="rect",
        key="zoi"
    )

    # Ausgewählte Zone extrahieren…

if canvas_result.json_data:
    objects = canvas_result.json_data["objects"]
    if objects:
        obj = objects[-1]  # letztes gezeichnetes Objekt
        x, y = int(obj["left"]), int(obj["top"])
        w, h = int(obj["width"]), int(obj["height"])
        roi = image_np[y:y+h, x:x+w]

        st.image(roi, caption="🎯 Ausgewählte Zone of Interest (ZOI)")


# 🧭 Seiteneinstellungen
st.set_page_config(page_title="🧬 Konturen entfernen", layout="centered")
st.title("🧬 Konturen aus dem Bild entfernen")

# 📥 Bildupload
uploaded_file = st.file_uploader("📤 Bild hochladen", type=["jpg", "jpeg", "png", "tif", "tiff"])

if uploaded_file:
    # 📷 TIFF-Konvertierung für OpenCV
    if uploaded_file.name.endswith((".tif", ".tiff")):
        pil_img = Image.open(uploaded_file).convert("RGB")
        image = np.array(pil_img)
    else:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)

    # 🖼️ Anzeige des Originalbildes
    st.image(image, caption="📷 Originalbild", channels="BGR")

    # 🖤 Graustufen-Konvertierung
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 🎚️ Grauintensitätsfilter
    min_gray = st.sidebar.slider("Minimale Grauintensität", 0, 255, 0)
    max_gray = st.sidebar.slider("Maximale Grauintensität", 0, 255, 255)

    # 🧽 Maske basierend auf Grauintensität
    intensity_mask = cv2.inRange(gray_image, min_gray, max_gray)

    # 🎯 Gefilterte Bildanzeige
    filtered_image = cv2.bitwise_and(image, image, mask=intensity_mask)
    st.image(filtered_image, caption="🎯 Gefilterte Grauintensitäten", channels="BGR")

    # 📍 Auswahl zwischen Farbbild und Graustufen
    ansicht = st.radio("🔍 Bildanzeige:", ["Farbe", "Graustufen"])
    if ansicht == "Farbe":
        st.image(image, caption="📷 Originalbild", channels="BGR")
    else:
        st.image(gray_image, caption="🖤 Graustufenbild", channels="GRAY")

    # ⚙️ Sidebar-Einstellungen
    st.sidebar.header("🧪 Einstellungen")
    th1 = st.sidebar.slider("Kanten-Schwelle 1", 0, 255, 50)
    th2 = st.sidebar.slider("Kanten-Schwelle 2", 0, 255, 150)
    repair_radius = st.sidebar.slider("Inpaint Radius", 1, 10, 3)

    # 🧼 Vorbereitung zur Konturenentfernung
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, th1, th2)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, contours, -1, 255, thickness=cv2.FILLED)

    # 🎯 Konturen entfernen durch Inpainting
    result = cv2.inpaint(image, mask, repair_radius, cv2.INPAINT_TELEA)

    # 🎛️ Anzeigeauswahl
    mode = st.radio("🖼️ Anzeige", ["Bild ohne Konturen", "Maske", "Konturenbild"])
    if mode == "Bild ohne Konturen":
        st.image(result, caption="✅ Konturen entfernt", channels="BGR")
    elif mode == "Maske":
        st.image(mask, caption="🎭 Konturenmaske", channels="GRAY")
    else:
        contour_img = np.zeros_like(image)
        cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 1)
        st.image(contour_img, caption="📏 Konturendarstellung", channels="BGR")
else:
    st.info("⬆️ Bitte lade ein Bild hoch, um loszulegen.")
