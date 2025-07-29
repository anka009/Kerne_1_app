import streamlit as st
import cv2
import numpy as np
from PIL import Image

gesamt_counter = 0
farb_counter = {"rot": 0, "blau": 0, "unbekannt": 0}

st.title("🟠 Farbige Kreise erkennen mit HSV und Durchmesser-Regler")

uploaded_file = st.file_uploader("Lade ein Bild hoch", type=["jpg", "jpeg", "png", "tif", "tiff"])

if uploaded_file:
    # Unterstütze TIFF via PIL
    if uploaded_file.name.lower().endswith((".tif", ".tiff")):
        pil_img = Image.open(uploaded_file).convert("RGB")
        image = np.array(pil_img)
    else:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)

    # Helligkeit/Kontrast
    alpha = st.slider("Kontrast (α)", 0.5, 3.0, 1.2)
    beta = st.slider("Helligkeit (β)", -100, 100, 20)
    adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    # Graustufen & Blur
    gray = cv2.cvtColor(adjusted, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)

    # Konturen finden
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Durchmesser-Regler
    min_radius, max_radius = st.slider("Kreis-Durchmesser (Pixel)", 1, 1000, (30, 300))

    # Farbwahl
    farbwahl = st.selectbox("Zeige nur Kreise mit Farbe:", ["alle", "rot", "blau"])

    output = adjusted.copy()

    def classify_color_hsv(roi):
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        hue = cv2.mean(hsv)[0]
        if 0 <= hue <= 15 or 160 <= hue <= 180:
            return "rot"
        elif 90 <= hue <= 130:
            return "blau"
        return "unbekannt"

    for cnt in contours:
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        if min_radius <= radius <= max_radius:
            x, y, r = int(x), int(y), int(radius)
            roi = output[y - r:y + r, x - r:x + r]
            if roi.shape[0] > 0 and roi.shape[1] > 0:
                farbe = classify_color_hsv(roi)
                farb_counter[farbe] += 1
                gesamt_counter += 1

                if farbwahl == "alle" or farbe == farbwahl:
                    cv2.circle(output, (x, y), r, (0, 255, 0), 2)
                    cv2.putText(output, farbe, (x - r, y - r), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Ergebnisse anzeigen
    st.image(output, caption="Erkannte Kreise", channels="BGR")
    st.subheader("🧮 Kernauszählung (live)")

    for farbe in ["rot", "blau", "unbekannt"]:
        st.write(f"{farbe.capitalize()}: {farb_counter[farbe]}")

    if gesamt_counter > 0:
        anteil_rot = 100 * farb_counter["rot"] / gesamt_counter
        st.write(f"🔴 Anteil roter Kerne: **{anteil_rot:.1f}%**")
    else:
        st.write("🔴 Anteil roter Kerne: 0.0%")
else:
    st.info("📂 Bitte ein Bild hochladen, um zu starten.")
