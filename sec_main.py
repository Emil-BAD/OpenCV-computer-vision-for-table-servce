import cv2

image = cv2.imread('images/image.jpg')

image = cv2.resize(image, (image.shape[1] // 2, image.shape[0] // 2))

image = cv2.Canny(image, 80, 80)

cv2.imshow('Image', image)
cv2.waitKey(0)
