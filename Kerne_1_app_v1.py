import streamlit as st
import cv2, numpy as np
from PIL import Image
import pandas as pd
from datetime import datetime
from streamlit_drawable_canvas import st_canvas

st.set_page_config(layout="wide")
st.title("🔍 Fleck-Analyse mit Zone of Interest & Korrektur")

uploaded = st.file_uploader("Bild hochladen", type=["jpg","jpeg","png","tif","tiff"])
if not uploaded:
    st.info("Bitte ein Bild hochladen.")
    st.stop()

img = Image.open(uploaded).convert("RGB")
img_np = np.array(img)
hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)

st.sidebar.header("🧭 Analyse-Optionen")
# ZOI-Modus: Ellipse zeichnen
zoi_mode = st.sidebar.checkbox("Zone of Interest zeichnen", value=True)

# Farbauswahl manuell
manual_color = st.sidebar.selectbox("Manuelle Fleckenfarbe", ["rot", "blau"])

# Zeichenmodus
modus = st.sidebar.radio("Modus:", ["Hinzufügen", "Löschen"])

# Farbfilter
settings = {}
for col_name, default_h in [("rot", 0), ("blau", 100)]:
    with st.sidebar.expander(col_name.capitalize()):
        h1 = st.slider(f"{col_name} H min", 0, 179, default_h)
        h2 = st.slider(f"{col_name} H max", 0, 179, default_h + (20 if col_name=="blau" else 15))
        s1 = st.slider("S min", 0, 255, 70)
        s2 = st.slider("S max", 0, 255, 255)
        v1 = st.slider("V min", 0, 255, 50)
        v2 = st.slider("V max", 0, 255, 255)
        settings[col_name] = ((h1,s1,v1), (h2,s2,v2), col_name)

# ZOI & Canvas zeichnen
background = np.array(img.convert("RGBA"))
draw_mode = "ellipse" if zoi_mode else "circle"
canvas_result = st_canvas(
    background_image=background,
    height=img_np.shape[0], width=img_np.shape[1],
    drawing_mode=draw_mode,
    stroke_width=2,
    stroke_color="#00FF00",
    key="canvas",
    update_streamlit=True,
)

# Bestimme Zone
mask_zone = np.ones(img_np.shape[:2], dtype=np.uint8) * 255
if canvas_result.json_data and "objects" in canvas_result.json_data:
    for o in canvas_result.json_data["objects"]:
        if o["type"] == "ellipse":
            # Ellipsenparameter
            left, top = int(o["left"]), int(o["top"])
            rx, ry = int(o["width"]/2), int(o["height"]/2)
            cy, cx = top + ry, left + rx
            mask_zone = np.zeros_like(mask_zone)
            cv2.ellipse(mask_zone, (cx, cy), (rx, ry), 0, 0, 360, 255, -1)

# Automatische Fleckenkennung
output = img_np.copy()
counts = {"rot":0,"blau":0}
manuals = []
deletes = []

kernel = np.ones((5,5),np.uint8)
for col, (low, high, key) in settings.items():
    low = np.array(low); high = np.array(high)
    mask = cv2.inRange(hsv, low, high)
    mask = cv2.bitwise_and(mask, mask, mask=mask_zone)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    cnts,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        (x,y),r = cv2.minEnclosingCircle(c)
        if r>5:
            center = (int(x), int(y))
            counts[col]+=1
            color = (0,0,255) if col=="rot" else (255,0,0)
            cv2.circle(output, center, int(r), color, 2)

# Manuelle Punkte & Löschen
if canvas_result.json_data and "objects" in canvas_result.json_data:
    for o in canvas_result.json_data["objects"]:
        x,y = int(o["left"]), int(o["top"])
        if o["type"]=="circle":
            if modus=="Hinzufügen":
                manuals.append((x,y, manual_color))
            else:
                deletes.append((x,y))

for x,y,col in manuals:
    clr = (0,0,255) if col=="rot" else (255,0,0)
    cv2.circle(output, (x,y), 8, clr, 2)
    counts[col]+=1

for x,y in deletes:
    cv2.circle(output, (x,y), 8, (0,255,255), 2)
    # keine Zählung bei löschen

# Anzeige
st.image(output, caption="Analyse mit Zone, automatisch & manuell", use_column_width=True)

# Ergebnisse / Prozent
rot = counts["rot"]; blau = counts["blau"]
st.write(f"🔴 Rot: {rot}   🔵 Blau: {blau}")
total = rot + blau
if total>0:
    pct = rot/total*100
    st.write(f"🔍 Anteil rot: **{pct:.1f}%**")
else:
    st.write("Keine Flecken erkannt.")

# CSV Export
df = pd.DataFrame([{"rot":rot,"blau":blau,"anteil_rot":pct if total>0 else 0}])
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
st.download_button("CSV herunterladen", df.to_csv(index=False), file_name=f"ergebnis_{ts}.csv", mime="text/csv")
