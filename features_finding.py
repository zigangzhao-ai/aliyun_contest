# ——*——coding:utf-8——*——
# author: 胡润龙、hhhfccz(胡珈魁) time:2020/8/25
import numpy as np
import cv2

'''
基于FlannBasedMatcher近邻近似匹配的sift实现
接口定义：输入图像img1和img2的路径，返回一组DMatch类型（每组两个DMatch）
测试方法：将返回的DMatch用cv2.drawMatchesKnn方法绘制在水平拼接的两图片上
'''

# 输入接口
img_left = cv2.imread(input("Enter the path of img_left："))
img_right = cv2.imread(input("Enter the path of img_right："))

# 对图像进行灰度处理
img_left_g = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
img_right_g = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

# 创建sift对象
sift = cv2.xfeatures2d.SIFT_create()

# 寻找关键点和描述符
keypoints1, features1 = sift.detectAndCompute(img_left_g, None)
keypoints2, features2 = sift.detectAndCompute(img_right_g, None)

# 设置FLANN参数
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)

# 将FlannBasedMatcher方法实例化
flann = cv2.FlannBasedMatcher(index_params, search_params)

# 利用knnMatch匹配处理，并将结果返回给matches
matches = flann.knnMatch(features1, features2, k=2)

# 比值判别法删除离群点，将符合要求的匹配结果进行输出
good = []
for m, n in matches:
    if m.distance < 0.75*n.distance:
        good.append([m])

result = cv2.drawMatchesKnn(img_left, keypoints1, img_right, keypoints2, good, None, flags=2)
result = cv2.resize(result, (int(img_left.shape[0]/2), int(img_left.shape[1])), interpolation=cv2.INTER_CUBIC)
cv2.imshow("SIFT_left", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
