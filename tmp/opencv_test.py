import cv2 as cv

src = cv.imread('E:/MMpose/mmpose/tmp/vis_109.jpg', 0)
cv.imshow('src', src)
dst = cv.rectangle(src, (290, 80), (290 + 70, 80 + 100), (255, 255, 255), 2)
cv.imshow('dst', dst)
cv.waitKey(0)
cv.destroyAllWindows()
