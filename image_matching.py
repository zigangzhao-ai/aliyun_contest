# ——*——coding:utf-8——*——
# author: 胡润龙、hhhfccz(胡珈魁) time:2020/8/25
import numpy as np
import cv2


def SIFT(img_left, img_right, img_left_g, img_right_g, MIN_MATCH_COUNT=10, norm=0.75):
    # 获取图片大小
    global result
    h, w = img_left.shape[:2]

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
        if m.distance < norm * n.distance:
            good.append([m])

    matching = cv2.drawMatchesKnn(img_left, keypoints1, img_right, keypoints2, good, None, flags=2)
    matching = cv2.resize(matching, (int(w / 2), int(h / 2)),
                          interpolation=cv2.INTER_CUBIC)

    if len(good) > MIN_MATCH_COUNT:
        # 得到两幅待拼接图的匹配点集
        # src = source;dst = destination;pts = points
        good = []
        for m, n in matches:
            if m.distance < norm * n.distance:
                good.append(m)
        src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        # cv2.findHomography：传入两个图像里的点集合,返回目标的透视转换
        # 第三个参数用于计算单应矩阵的方法
        # 第四个参数为误差阈值
        # 返回值：H为变换矩阵,mask为掩膜
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        # cv2.warpPerspective：利用M矩阵对原图进行透视变换（M点乘位置向量）
        wrap = cv2.warpPerspective(img_left, H,
                                   (2 * w, 2 * h))
        wrap[0:h, 0:w] = img_right

        # 去除黑色无用部分
        rows, cols = np.where(wrap[:, :, 0] != 0)
        min_row, max_row = min(rows), max(rows) + 1
        min_col, max_col = min(cols), max(cols) + 1

        result = wrap[min_row:max_row, min_col:max_col, :]
        result = cv2.resize(result, (int(w / 2), int(h / 2)),
                            interpolation=cv2.INTER_CUBIC)

    return matching, result


if __name__ == '__main__':
    pass
