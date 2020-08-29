# ——*——coding:utf-8——*——
# author: hhhfccz(胡珈魁) time:2020/8/29
import cv2
import numpy as np
from rotation_according_to_word_direction import rotated_img_with_fft
from image_matching import SIFT

if __name__ == "__main__":
    # 输入接口（有点小问题，有空再改吧）
    number = input("choose your text number: ")
    img_right = cv2.imread("./pic_left/"+str(number)+".jpg")
    img_left = cv2.imread("./pic_right/"+str(number)+".jpg")
    print("----start the image processing----\n")

    # 对图像进行灰度处理
    img_left_g = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
    img_right_g = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

    # 将两个图片拼接在一起
    matching, result = SIFT(img_left, img_right, img_left_g, img_right_g, norm=0.6)

    # 利用FFT进行文字方向矫正
    img_gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    rotated = rotated_img_with_fft(result, img_gray)
    print("----finished----")

    cv2.imshow("rotated", rotated)
    cv2.imwrite("./rotated_result/result"+str(number)+".jpg", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
