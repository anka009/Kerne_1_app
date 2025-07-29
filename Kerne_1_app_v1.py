import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile

# 🔬 Zellkernform prüfen (Rundheitskriterium)
def is_round(contour):
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    if perimeter == 0:
        return False
    circularity = 4 * np.pi * (area / (perimeter ** 2))
    return circularity > 0.6  # Schwellenwert für Rundheit

# 📌 Farbklassifikation nach mittlerem RGB-Wert
def classify_color(masked_image):
    mean_color = cv2.mean(masked_image)
    if mean_color[2] > mean_color[0] + 50:
        return 'rot'
    elif mean_color[0] > mean_color[2] + 50:
        return 'blau'
    else:
        return 'unklar'

# 🖼️ Bildanalyse-Funktion
def analyze_image(pil_img):
    image = np.array(pil_img)
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # 👉 Bild verbessern
    enhanced_image = enhance_contrast_and_brightness(image)

    # 👉 Mit verbessertem Bild weiterarbeiten
    gray = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    total_nuclei = 0
    red_nuclei = 0
    blue_nuclei = 0

    for cnt in contours:
        if is_round(cnt):
            total_nuclei += 1
            mask = np.zeros(gray.shape, dtype=np.uint8)
            cv2.drawContours(mask, [cnt], -1, 255, -1)
            masked_img = cv2.bitwise_and(image, image, mask=mask)
            color_type = classify_color(masked_img)
            if color_type == 'rot':
                red_nuclei += 1
            elif color_type == 'blau':
                blue_nuclei += 1

    # 📊 Prozent berechnen
    color_total = red_nuclei + blue_nuclei
    red_percent = (red_nuclei / color_total) * 100 if color_total else 0

    return {
        'gesamt': total_nuclei,
        'rot': red_nuclei,
        'blau': blue_nuclei,
        'rot_prozent': round(red_percent, 2)
    }

# 📥 Streamlit UI
st.title("🔬 Zellkernanalyse-App")
uploaded_files = st.file_uploader(
    "Bilder hochladen (jpg, jpeg, tif, tiff, bmp, png)",
    type=["jpg", "jpeg", "tif", "tiff", "bmp", "png"],
    accept_multiple_files=True
)

if uploaded_files:
    for file in uploaded_files:
        st.subheader(f"🖼️ {file.name}")
        img = Image.open(file)
        st.image(img, caption=f"Originalbild: {file.name}", use_column_width=True)

        results = analyze_image(img)

        st.write(f"🧮 **Gesamtzahl erkannter Zellkerne:** {results['gesamt']}")
        st.write(f"🔴 **Rote Kerne:** {results['rot']}")
        st.write(f"🔵 **Blaue Kerne:** {results['blau']}")
        st.write(f"📊 **Prozentanteil rote Zellkerne:** {results['rot_prozent']}%")
else:
    st.info("⬆️ Bitte lade ein oder mehrere Bilder hoch.")
