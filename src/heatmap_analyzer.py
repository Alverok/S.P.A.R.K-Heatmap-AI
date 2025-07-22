
!pip install opencv-python-headless matplotlib numpy


from google.colab import files
uploaded = files.upload()
filename = list(uploaded.keys())[0]


import cv2
import numpy as np
import matplotlib.pyplot as plt


img = cv2.imread(filename)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Show uploaded image
plt.imshow(img)
plt.title("Uploaded Heatmap")
plt.axis("off")
plt.show()


img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)


