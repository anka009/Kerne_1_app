import streamlit as st
import cv2
import numpy as np
from PIL import Image
from streamlit_cropper import st_cropper

# 🎛️ Regler in Sidebar
alpha = st.sidebar.slider("Kontrast (alpha)", 0.5, 3.0, 1.5, 0.1)
beta = st.sidebar.slider("Helligkeit (beta)", -100, 100, 20, 5)

# 🔧 Bildverbesserung
def enhance_contrast_and_brightness(image, alpha, beta):
    image_np = np.array(image)
    if image_np.ndim == 2:
        image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2BGR)
    enhanced = cv2.convertScaleAbs(image_np, alpha=alpha, beta=beta)
    return enhanced

# 🔵 Farbklassifikation
def classify_color(masked_image):
    mean_color = cv2.mean(masked_image)
    if mean_color[2] > mean_color[0] + 50:
        return 'rot'
    elif mean_color[0] > mean_color[2] + 50:
        return 'blau'
    else:
        return 'unklar'

# 🟢 Rundheitsprüfung
def is_round(contour):
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    if perimeter == 0:
        return False
    circularity = 4 * np.pi * (area / (perimeter ** 2))
    return circularity > 0.6

# 🔬 Hauptanalysefunktion
def analyze_image(pil_img, alpha, beta):
    image = np.array(pil_img)
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    enhanced_image = enhance_contrast_and_brightness(image, alpha, beta)
    gray = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    total_nuclei, red_nuclei, blue_nuclei = 0, 0, 0
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

    color_total = red_nuclei + blue_nuclei
    red_percent = (red_nuclei / color_total) * 100 if color_total else 0

    return {
        'gesamt': total_nuclei,
        'rot': red_nuclei,
        'blau': blue_nuclei,
        'rot_prozent': round(red_percent, 2)
    }

# 🧪 Streamlit UI
st.title("🧬 Zellkernanalyse mit ZOI & Optimierung")

uploaded_file = st.file_uploader("📤 Bild hochladen", type=["jpg", "jpeg", "png", "tif", "tiff", "bmp"])
if uploaded_file:
    img = Image.open(uploaded_file)
    st.write("📍 Wähle deine Zone of Interest aus:")
    cropped_img = st_cropper(img, box_color='red', aspect_ratio=None)

    enhanced_img = enhance_contrast_and_brightness(cropped_img, alpha, beta)

    col1, col2 = st.columns(2)
    col1.image(cropped_img, caption="Zone of Interest")
    col2.image(enhanced_img, caption="Optimiert")

    results = analyze_image(cropped_img, alpha, beta)
    st.write(f"🔢 **Zellkerne gesamt:** {results['gesamt']}")
    st.write(f"🔴 **Rote Kerne:** {results['rot']}")
    st.write(f"🔵 **Blaue Kerne:** {results['blau']}")
    st.write(f"📊 **Anteil rote Zellkerne:** {results['rot_prozent']}%")
else:
    st.info("⬆️ Bitte lade ein Bild hoch.")
