
"""
heatmap_analyzer.py

This script analyzes a heatmap image to detect the most crowded aisles in a store layout.
It uses HSV color filtering to detect red intensity (heat zones) in predefined aisle regions.

How it works:
- Reads the heatmap image
- Converts to HSV to isolate red hues
- Defines aisle bounding boxes manually
- Calculates red pixel ratio for each aisle
- Highlights the most crowded aisle on the image


drawback : - Aisle coordinates are hardcoded; automatic aisle detection is not implemented.
  This requires manual adjustment if the layout or resolution changes.
"""



!pip install opencv-python-headless matplotlib numpy


from google.colab import files
uploaded = files.upload()
filename = list(uploaded.keys())[0]


import cv2
import numpy as np
import matplotlib.pyplot as plt


img = cv2.imread(filename)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


plt.imshow(img)
plt.title("Uploaded Heatmap")
plt.axis("off")
plt.show()


img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)


aisle_boxes = [
    (560,  60, 80, 250),   
    (160, 120, 80, 250),   
    (280, 100, 60, 120),
    (360, 100, 60, 250),   
    (480, 70, 40, 180),   
    (240, 340, 200, 80),   
]


crowd_levels = []


for i, (x, y, w, h) in enumerate(aisle_boxes):
    aisle_crop = img_hsv[y:y+h, x:x+w]

    red_mask1 = cv2.inRange(aisle_crop, (0, 100, 100), (10, 255, 255))
    red_mask2 = cv2.inRange(aisle_crop, (160, 100, 100), (180, 255, 255))
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)

    red_ratio = np.sum(red_mask > 0) / (w * h)
    crowd_levels.append(red_ratio)

    print(f"Aisle {i+1} crowd level (red ratio): {red_ratio:.2f}")


most_crowded_index = np.argmax(crowd_levels)
print(f"\n🛒 Most crowded aisle: Aisle {most_crowded_index + 1}")


img_marked = img.copy()
for i, (x, y, w, h) in enumerate(aisle_boxes):
    color = (255, 0, 0)  
    if i == most_crowded_index:
        color = (255, 0, 255)  
    cv2.rectangle(img_marked, (x, y), (x+w, y+h), color, 2)
    cv2.putText(img_marked, f"A{i+1}", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

plt.figure(figsize=(10, 6))
plt.imshow(img_marked)
plt.title("Detected Crowded Aisle(s)")
plt.axis("off")
plt.show()
