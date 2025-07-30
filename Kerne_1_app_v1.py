import streamlit as st
import cv2
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas

st.set_page_config(layout="wide")
st.title("🔴🟦 Fleckanalyse mit Farberkennung, Korrektur und ROI")

# Initiale Zähler
farb_counter = {"rot": 0, "blau": 0}
korrektur_hinzufuegen = []
korrektur_loeschen = []

# Bild hochladen
uploaded_file = st.file_uploader("Bild hochladen", type=["jpg", "jpeg", "png", "tif", "tiff"])
if uploaded_file:
    # Bild einlesen
    pil_img = Image.open(uploaded_file).convert("RGB")
    image = np.array(pil_img)
    image_display = image.copy()

    # Helligkeit/Kontrast
    col1, col2 = st.columns(2)
    with col1:
        alpha = st.slider("Kontrast (alpha)", 0.5, 3.0, 1.2)
    with col2:
        beta = st.slider("Helligkeit (beta)", -100, 100, 10)
    adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    # HSV-Schieberegler
    st.subheader("🎛️ Farbfilter (HSV)")
    col1, col2, col3 = st.columns(3)
    with col1:
        hmin = st.slider("H min", 0, 180, 0)
        hmax = st.slider("H max", 0, 180, 15)
    with col2:
        smin = st.slider("S min", 0, 255, 70)
        smax = st.slider("S max", 0, 255, 255)
    with col3:
        vmin = st.slider("V min", 0, 255, 50)
        vmax = st.slider("V max", 0, 255, 255)

    # ROI-Auswahl via Canvas
    st.subheader("📍 Region of Interest (ROI) auswählen")
    canvas_result = st_canvas(
        fill_color="rgba(255, 0, 0, 0.3)",
        stroke_width=3,
        background_image=Image.fromarray(adjusted),
        update_streamlit=True,
        height=adjusted.shape[0],
        width=adjusted.shape[1],
        drawing_mode="circle",
        key="roi_canvas"
    )

    # ROI Maske generieren
    mask_roi = np.zeros(adjusted.shape[:2], dtype=np.uint8)
    if canvas_result.json_data and "objects" in canvas_result.json_data:
        for obj in canvas_result.json_data["objects"]:
            if obj["type"] == "circle":
                cx = int(obj["left"] + obj["radius"])
                cy = int(obj["top"] + obj["radius"])
                r = int(obj["radius"])
                cv2.circle(mask_roi, (cx, cy), r, 255, -1)

    hsv = cv2.cvtColor(adjusted, cv2.COLOR_BGR2HSV)
    maske = cv2.inRange(hsv, (hmin, smin, vmin), (hmax, smax, vmax))
    if np.any(mask_roi):
        maske = cv2.bitwise_and(maske, mask_roi)

    # Kreise erkennen
    konturen, _ = cv2.findContours(maske, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    radius_min, radius_max = st.slider("Kreisradius (Pixel)", 1, 500, (10, 80))

    farbwahl = st.selectbox("Nur diese Farbe anzeigen", ["alle", "rot", "blau"])
    out = adjusted.copy()

    for cnt in konturen:
        (x, y), r = cv2.minEnclosingCircle(cnt)
        x, y, r = int(x), int(y), int(r)
        if radius_min <= r <= radius_max:
            roi = adjusted[y - r:y + r, x - r:x + r]
            if roi.size == 0:
                continue
            roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            h_mittel = cv2.mean(roi_hsv)[0]
            farbe = "unbekannt"
            if 0 <= h_mittel <= 15 or 160 <= h_mittel <= 180:
                farbe = "rot"
            elif 90 <= h_mittel <= 130:
                farbe = "blau"

            if farbe in farb_counter:
                farb_counter[farbe] += 1

            if farbwahl == "alle" or farbe == farbwahl:
                cv2.circle(out, (x, y), r, (0, 255, 0), 2)
                cv2.putText(out, farbe, (x - r, y - r), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    st.image(out, caption="Erkannte Flecken", channels="BGR")

    # Ergebnis-Anzeige
    gesamt = farb_counter["rot"] + farb_counter["blau"]
    st.subheader("📊 Auswertung")
    col1, col2, col3 = st.columns(3)
    col1.metric("🔴 Rot", farb_counter["rot"])
    col2.metric("🔵 Blau", farb_counter["blau"])
    if gesamt > 0:
        prozent_rot = 100 * farb_counter["rot"] / gesamt
        col3.metric("% Rot", f"{prozent_rot:.1f}%")
    else:
        col3.metric("% Rot", "0.0%")
else:
    st.info("⬆️ Bitte ein Bild hochladen.")
