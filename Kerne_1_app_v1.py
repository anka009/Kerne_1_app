import cv2
import numpy as np
from PIL import Image
import streamlit as st

st.title("🧩 Konturen anzeigen")

uploaded_file = st.file_uploader("Bild hochladen", type=["jpg", "jpeg", "png", "tif", "tiff"])
if uploaded_file:
    # TIFF-Unterstützung
    if uploaded_file.name.endswith((".tif", ".tiff")):
        pil_img = Image.open(uploaded_file).convert("RGB")
        image = np.array(pil_img)
    else:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)

    # Vorverarbeitung: Graustufen & Kanten
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blurred, threshold1=50, threshold2=150)

    # Konturen finden
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Leeres Bild für Konturen
    contour_img = np.zeros_like(image)
    cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 1)

    st.image(contour_img, caption="🖼️ Nur Konturen", channels="BGR")
else:
    st.info("⬆️ Bitte lade ein Bild hoch, um Konturen zu sehen.")
