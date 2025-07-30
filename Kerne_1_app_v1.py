import streamlit as st
import cv2
import numpy as np
from PIL import Image
from collections import defaultdict

st.set_page_config(layout="wide")
st.title("🔴🔵 Fleckanalyse mit Farberkennung, Korrektur und ROI")

# 🔢 Zähler initialisieren
farb_counter = defaultdict(int)
korrektur_hinzufuegen = []
korrektur_loeschen = []

# 📂 Bild hochladen
uploaded_file = st.file_uploader("Bild hochladen", type=["jpg", "jpeg", "png", "tif", "tiff"])
if uploaded_file:
    pil_img = Image.open(uploaded_file).convert("RGB")
    image = np.array(pil_img)

    # 🔆 Helligkeit & Kontrast
    col1, col2 = st.columns(2)
    alpha = col1.slider("Kontrast (alpha)", 0.5, 3.0, 1.2)
    beta = col2.slider("Helligkeit (beta)", -100, 100, 10)
    adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    # ✂️ ROI-Auswahl
    st.subheader("📘 Manuelle ROI-Auswahl")
    x = st.slider("X-Position", 0, adjusted.shape[1], adjusted.shape[1] // 4)
    y = st.slider("Y-Position", 0, adjusted.shape[0], adjusted.shape[0] // 4)
    w = st.slider("Breite", 10, adjusted.shape[1] - x, adjusted.shape[1] // 2)
    h = st.slider("Höhe", 10, adjusted.shape[0] - y, adjusted.shape[0] // 2)
    roi = adjusted[y:y+h, x:x+w]
    st.image(roi, caption="Ausgewählte ROI", use_column_width=True)

    # 🎨 Farbfilter
    st.subheader("🎛️ Farbfilter (HSV)")
    col1, col2, col3 = st.columns(3)
    hmin = col1.slider("H min", 0, 180, 0)
    hmax = col1.slider("H max", 0, 180, 15)
    smin = col2.slider("S min", 0, 255, 70)
    smax = col2.slider("S max", 0, 255, 255)
    vmin = col3.slider("V min", 0, 255, 50)
    vmax = col3.slider("V max", 0, 255, 255)

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    maske = cv2.inRange(hsv, (hmin, smin, vmin), (hmax, smax, vmax))
    konturen, _ = cv2.findContours(maske, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    radius_min, radius_max = st.slider("Kreisradius (Pixel)", 1, 500, (10, 80))
    farbwahl = st.selectbox("Nur diese Farbe anzeigen", ["alle", "rot", "blau"])

    out = roi.copy()
    for cnt in konturen:
        (cx, cy), r = cv2.minEnclosingCircle(cnt)
        cx, cy, r = int(cx), int(cy), int(r)
        if radius_min <= r <= radius_max:
            kreis_roi = roi[cy - r:cy + r, cx - r:cx + r]
            if kreis_roi.size == 0:
                continue
            roi_hsv = cv2.cvtColor(kreis_roi, cv2.COLOR_BGR2HSV)
            h_mittel = cv2.mean(roi_hsv)[0]

            farbe = "unbekannt"
            if 0 <= h_mittel <= 15 or 160 <= h_mittel <= 180:
                farbe = "rot"
            elif 90 <= h_mittel <= 130:
                farbe = "blau"

            farb_counter[farbe] += 1  # zählt auch unbekannte automatisch

            if farbwahl == "alle" or farbe == farbwahl:
                cv2.circle(out, (cx, cy), r, (0, 255, 0), 2)
                cv2.putText(out, farbe, (cx - r, cy - r), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    st.image(out, caption="Flecken innerhalb der ROI", channels="BGR")

    # 📈 Auswertung anzeigen
    gesamt = sum(farb_counter.values())
    st.subheader("📊 Auswertung")
    st.metric("⚪ Gesamtzahl Kreise", gesamt)

    col1, col2, col3 = st.columns(3)
    col1.metric("🔴 Rot", farb_counter["rot"])
    col2.metric("🔵 Blau", farb_counter["blau"])
    prozent_rot = 100 * farb_counter["rot"] / gesamt if gesamt > 0 else 0
    col3.metric("% Rot", f"{prozent_rot:.1f}%")

    # 🧾
