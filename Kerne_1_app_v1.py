import cv2
import numpy as np
from skimage import measure
from matplotlib import pyplot as plt

# 1. Bild laden
image = cv2.imread('zellkerne.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 2. Thresholding zur Segmentierung runder Formen
ret, thresh = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY)

# 3. Konturen finden
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 4. Filter auf runde/geometrisch passende Konturen
def is_round(cnt):
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    if perimeter == 0: return False
    circularity = 4 * np.pi * (area / (perimeter * perimeter))
    return circularity > 0.6  # Schwelle für Rundheit

kern_count = 0
kern_mask = np.zeros(gray.shape, dtype=np.uint8)

for cnt in contours:
    if is_round(cnt):
        cv2.drawContours(kern_mask, [cnt], -1, 255, -1)
        kern_count += 1

# 5. Farbklassifikation (rot/blau)
# Beispiel: rote Kerne haben hohen Rotwert > Blauwert
red_kern_count = 0
blue_kern_count = 0
for cnt in contours:
    mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.drawContours(mask, [cnt], -1, 255, -1)
    mean_color = cv2.mean(image, mask=mask)
    if mean_color[2] > mean_color[0] + 50:  # R > B deutlich
        red_kern_count += 1
    elif mean_color[0] > mean_color[2] + 50:  # B > R deutlich
        blue_kern_count += 1

# 6. Prozentanteil berechnen
total_color_kern = red_kern_count + blue_kern_count
if total_color_kern > 0:
    red_percent = red_kern_count / total_color_kern * 100
    print(f"🔴 Anteil roter Zellkerne: {red_percent:.2f}%")
else:
    print("Keine farblich klassifizierbaren Kerne gefunden.")

print(f"🟢 Gesamtzahl geometrisch erkannter Zellkerne: {kern_count}")
