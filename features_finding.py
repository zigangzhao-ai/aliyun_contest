#基于FlannBasedMatcher近邻近似匹配的surf实现
#接口定义：输入图像img1和img2的路径，返回一组DMatch类型（每组两个DMatch）
#测试方法：将返回的DMatch用cv2.drawMatchesKnn方法绘制在水平拼接（hstack）的两图片上

import numpy as np 
import cv2

#输入接口部分
img1_path = input("Enter the path of img1：")
img2_path = input("Enter the path of img2：")

img1 = cv2.imread(img1_path)
img2 = cv2.imread(img2_path)

#对图像进行灰度处理
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

#创建surf对象
surf = cv2.xfeatures2d.SURF_creat()

#寻找关键点和描述符
keypoints1, features1 = surf.detectAndCompute(img1,None)
keypoints2, features2 = surf.detectAndCompute(img2,None)

#设置FLANN参数
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm=FLANN_INDEX_KDTREE,trees=5)
search_params = dict(checks=50)

#将FlannBasedMatcher方法实例化
flann = cv2.FlannBasedMatcher(index_parameters, search_parameters)

#利用knnMatch匹配处理，并将结果返回给matches
matches = flann.knnMatch(features1,features2,k=2)

#对符合要求的匹配结果进行输出
good = []
for m,n in matches:
    if m.distance < 0.7*n.distance:
        good.append([m])

result = cv2.drawMatchesKnn(img1,features1,img2,features1,good,None,flags=2)
cv2.imshow("SURF", result)
cv2.waitKey(0)