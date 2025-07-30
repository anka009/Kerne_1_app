import streamlit as st
from streamlit_drawable_canvas import st_canvas
import cv2
import numpy as np
from PIL import Image
import io

# 🎨 Seitenlayout
st.set_page_config(page_title="🧬 Fleckenanalyse", layout="centered")
st.title("🧪 Rote & Blaue Flecken in ZOI erkennen")

# 📤 Bild-Upload
uploaded_file = st.file_uploader("📷 Bild hochladen", type=["jpg", "jpeg", "png"])
if uploaded_file:
    pil_img = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(pil_img)

    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    buf.seek(0)

    # 🖌️ Zeichenmodus auswählen
    drawing_mode = st.selectbox("🖌️ Zeichenmodus", ["rect", "circle"])
    
    # 🖼️ Canvas zum Zeichnen
   canvas_result = st_canvas(
       fill_color="rgba(255, 0, 0, 0.3)",
       stroke_width=2,
       background_image=pil_img,  # ← Hier statt Image.open(buf)
       height=pil_img.height,
       width=pil_img.width,
       drawing_mode=drawing_mode,
       key="canvas_key",
       update_streamlit=True
   )


    # 🧭 ZOI analysieren
    if canvas_result.json_data and canvas_result.json_data["objects"]:
        obj = canvas_result.json_data["objects"][-1]
        x, y = int(obj["left"]), int(obj["top"])
        w, h = int(obj["width"]), int(obj["height"])
        roi = image_np[y:y+h, x:x+w]

        # 🔄 In HSV-Farbraum umwandeln
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)

        # 🎯 Farbdefinitionen
        lower_red1 = np.array([0, 70, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 70, 50])
        upper_red2 = np.array([180, 255, 255])
        lower_blue = np.array([100, 150, 0])
        upper_blue = np.array([140, 255, 255])

        # 🧪 Masken
        mask_red = cv2.inRange(hsv_roi, lower_red1, upper_red1) | cv2.inRange(hsv_roi, lower_red2, upper_red2)
        mask_blue = cv2.inRange(hsv_roi, lower_blue, upper_blue)

        # 🔢 Flecken zählen
        red_count, _ = cv2.connectedComponents(mask_red)
        blue_count, _ = cv2.connectedComponents(mask_blue)
        st.write(f"🔴 Rote Flecken: **{red_count - 1}**")
        st.write(f"🔵 Blaue Flecken: **{blue_count - 1}**")

        # ✨ Hervorheben
        output_roi = roi.copy()
        contours_red, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_blue, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        cv2.drawContours(output_roi, contours_red, -1, (255, 0, 0), 2)       # Leuchtend Rot
        cv2.drawContours(output_roi, contours_blue, -1, (173, 216, 230), 2)  # Hellblau

        st.image(output_roi, caption="📍 Flecken in ZOI hervorgehoben")

    else:
        st.warning("Bitte zeichne eine Zone of Interest (ZOI) ins Bild.")

else:
    st.info("🔼 Bitte lade ein Bild hoch.")
