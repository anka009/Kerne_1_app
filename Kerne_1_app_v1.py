import cv2
import numpy as np
from PIL import Image
import streamlit as st

st.title("🧼 Bild ohne Konturen")

uploaded_file = st.file_uploader("Bild hochladen", type=["jpg", "jpeg", "png", "tif", "tiff"])
if uploaded_file:
    # TIFF-Unterstützung
    if uploaded_file.name.endswith((".tif", ".tiff")):
        pil_img = Image.open(uploaded_file).convert("RGB")
        image = np.array(pil_img)
    else:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)

    # Graustufen und Kantenerkennung
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    # Konturen finden
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Maske für Konturen erzeugen
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, contours, -1, 255, thickness=cv2.FILLED)

    # Bildbereiche an Konturen "reparieren"
    result = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)

    st.image(result, caption="🖼️ Bild ohne sichtbare Konturen", channels="BGR")
else:
    st.info("⬆️ Lade bitte ein Bild hoch.")
