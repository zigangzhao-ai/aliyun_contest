# ——*——coding:utf-8——*——
# author: 胡润龙、hhhfccz(胡珈魁) time:2020/8/25
import numpy as np
import cv2
import math


def rotated_img_with_fft(img, img_gray):
    # 对图像进行边界扩充
    top_size, bottom_size, left_size, right_size = 50, 50, 50, 50
    constant = cv2.copyMakeBorder(img_gray, top_size, bottom_size, left_size, right_size,
                                  borderType=cv2.BORDER_CONSTANT, value=0)

    # 获取图片大小
    h, w = img_gray.shape[:2]

    # 对图像进行傅里叶变换
    dft = np.fft.fft2(constant)
    shift = np.fft.fftshift(dft)

    fft_img = np.log(np.abs(shift))
    fft_img = (fft_img - np.amin(fft_img)) / (np.amax(fft_img) - np.amin(fft_img))
    fft_img *= 255

    # 边缘检测，关键问题1，边界的判定对最终结果影响较大
    blur = cv2.GaussianBlur(img, (3, 3), 1)
    thresh = cv2.Canny(blur.astype(np.uint8), 80, 120)

    # 霍夫变换，关键问题2，需要对函数数值做特定变换
    lines = cv2.HoughLinesP(thresh.astype(np.uint8), 1.0, np.pi / 180, 20, minLineLength=20, maxLineGap=10)
    if lines is not None:
        lines1 = lines[:, 0, :]
    else:
        lines1 = []

    piThresh = np.pi / 180
    pi2 = np.pi / 2
    angle = 0
    for line in lines1:
        x1, y1, x2, y2 = line

        if abs(x2 - x1) < 1e-2:
            continue
        else:
            theta = (y2 - y1) / (x2 - x1)
            if abs(theta) < piThresh or abs(theta - pi2) < piThresh:
                continue
            else:
                angle = abs(theta)
                break

    angle = math.atan(angle)
    angle = angle / piThresh
    center = (w // 2, h // 2)
    height_1 = int(w * math.fabs(math.sin(math.radians(angle))) + h * math.fabs(math.cos(math.radians(angle))))
    width_1 = int(h * math.fabs(math.sin(math.radians(angle))) + w * math.fabs(math.cos(math.radians(angle))))
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    M[0, 2] += (width_1 - w) / 2
    M[1, 2] += (height_1 - h) / 2

    # 旋转
    rotated = cv2.warpAffine(img_gray, M, (width_1, height_1), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    rotated = cv2.resize(rotated, (int(width_1/2), int(height_1/2)), interpolation=cv2.INTER_CUBIC)

    return rotated


if __name__ == "__main__":
    pass
