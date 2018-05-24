# coding: utf-8
from matplotlib import pyplot as plt
import numpy as np
import cv2
import math


# 讀入檔案

img = cv2.imread('./001.jpg',  cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('./002.jpg', cv2.IMREAD_GRAYSCALE)
img.shape
img2.shape


# 進行直方圖平均化 (左為原圖，右為平均化結果，使用numpy hstack合併）

equ = cv2.equalizeHist(img)
res = np.hstack((img,equ))
cv2.imwrite('001-equalizeHist.png', equ)


# 執行X和Y方向的Sobel算子

x = cv2.Sobel(img,cv2.CV_16S,1,0)
y = cv2.Sobel(img,cv2.CV_16S,0,1)
absX = cv2.convertScaleAbs(x)
absY = cv2.convertScaleAbs(y)
dst = cv2.addWeighted(absX,0.5,absY,0.5,0)
gaussian_img = cv2.GaussianBlur(img,(3,3),0)
canny = cv2.Canny(gaussian_img, 50, 150)
cv2.imwrite('001-sobelX.png', absX)
cv2.imwrite('001-sobelY.png', absY)
cv2.imwrite('001-sobel.png', dst)
cv2.imwrite('002-GaussianBlur.png', gaussian_img)
cv2.imwrite('002-canny.png', canny)

# 第二張圖也一樣

equ = cv2.equalizeHist(img2)
cv2.imwrite('002-equalizeHist.png', equ)
x = cv2.Sobel(img2,cv2.CV_16S,1,0)
y = cv2.Sobel(img2,cv2.CV_16S,0,1)
absX = cv2.convertScaleAbs(x)
absY = cv2.convertScaleAbs(y)
dst = cv2.addWeighted(absX,0.5,absY,0.5,0)
gaussian_img = cv2.GaussianBlur(img2,(3,3),0)
canny = cv2.Canny(gaussian_img, 50, 150)
cv2.imwrite('002-sobelX.png', absX)
cv2.imwrite('002-sobelY.png', absY)
cv2.imwrite('002-sobel.png', dst)
cv2.imwrite('002-GaussianBlur.png', gaussian_img)
cv2.imwrite('002-canny.png', canny)
