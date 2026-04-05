import cv2
import numpy as np

# cap = cv2.VideoCapture("video/video2.mov")

# while True:
#     success, img = cap.read()
#     if not success:
#         break
    
#     img = cv2.Canny(img, 90, 90)
    
#     kernel = np.ones((5, 5), np.uint8)
#     img = cv2.dilate(img, kernel, iterations=1)
    
#     img = cv2.erode(img, kernel, iterations=1)
    
#     cv2.imshow('Video', img)
    
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

img = cv2.imread('images/image.jpg')

img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.GaussianBlur(img, (5, 5), 0)

img = cv2.Canny(img, 80, 80)

con, fir = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

print(con)

cv2.imshow('Image', img)
cv2.waitKey(0)