import cv2
import numpy as np

photo = np.zeros((300, 300, 3), dtype=np.uint8)

# photo[:] = 255, 0, 0

cv2.rectangle(photo, (50, 50), (250, 250), (0, 255, 0), 3)

cv2.imshow('Photo', photo)
cv2.waitKey(0)