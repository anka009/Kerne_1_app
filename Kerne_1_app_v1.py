import cv2
import numpy as np
from PIL import Image
import streamlit as st

st.set_page_config(page_title="🧼 Konturen entfernen", layout="centered")
st.title("🧽 Konturen aus dem Bild entfernen")

uploaded_file = st.file_uploader("📤 Bild hochladen", type=["jpg", "jpeg", "png", "tif", "tiff"])

if uploaded_file:
    # TIFF-Konvertierung für OpenCV
    if uploaded_file.name.endswith((".tif", ".tiff")):
        pil_img = Image.open(uploaded_file).convert("RGB")
        image = np.array(pil_img)
    else:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)

    st.image(image, caption="📷 Originalbild", channels="BGR")
# Konvertierung zu Graustufen
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Anzeige der Graustufenversion
st.image(gray_image, caption="🖤 Graustufenbild", channels="GRAY")
ansicht = st.radio("🔍 Bildanzeige:", ["Farbe", "Graustufen"])
if ansicht == "Farbe":
    st.image(image, caption="Originalbild", channels="BGR")
else:
    st.image(gray_image, caption="Graustufenbild", channels="GRAY")

    # 🎚️ Schwellenwerte für Canny-Kanten
    st.sidebar.header("🧪 Einstellungen")
    th1 = st.sidebar.slider("Kanten-Schwelle 1", 0, 255, 50)
    th2 = st.sidebar.slider("Kanten-Schwelle 2", 0, 255, 150)
    repair_radius = st.sidebar.slider("Inpaint Radius", 1, 10, 3)

    # Vorbereitung
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blurred, th1, th2)

    # Konturen finden
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, contours, -1, 255, thickness=cv2.FILLED)

    # Konturen entfernen durch "Inpainting"
    result = cv2.inpaint(image, mask, repair_radius, cv2.INPAINT_TELEA)

    # Auswahl anzeigen
    mode = st.radio("🖼️ Anzeige", ["Bild ohne Konturen", "Maske", "Konturenbild"])

    if mode == "Bild ohne Konturen":
        st.image(result, caption="✅ Konturen entfernt", channels="BGR")
    elif mode == "Maske":
        st.image(mask, caption="🎭 Konturenmaske", channels="GRAY")
    else:
        contour_img = np.zeros_like(image)
        cv2.drawContours(contour_img, contours, -1, (0,255,0), 1)
        st.image(contour_img, caption="📏 Konturendarstellung", channels="BGR")
else:
    st.info("⬆️ Bitte lade ein Bild hoch, um loszulegen.")
