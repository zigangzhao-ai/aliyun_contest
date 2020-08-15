import numpy as np  
import cv2 

#最小匹配阈值
MIN_MATCH_COUNT = 10
 
#输入接口部分 
img1_path = input("Enter the path of img1：") 
img2_path = input("Enter the path of img2：") 
 
img1 = cv2.imread(img1_path) 
img2 = cv2.imread(img2_path) 
 
#对图像进行灰度处理 
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) 
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) 

def SURF(): 
    #创建surf对象 
    surf = cv2.xfeatures2d.SURF_creat() 
 
    #设置FLANN参数 
    FLANN_INDEX_KDTREE = 0 
    index_params = dict(algorithm=FLANN_INDEX_KDTREE,trees=5) 
    search_params = dict(checks=50) 
 
    #寻找关键点和描述符 
    keypoints1, features1 = surf.detectAndCompute(img1,None) 
    keypoints2, features2 = surf.detectAndCompute(img2,None) 
 
    #将FlannBasedMatcher方法实例化 
    flann = cv2.FlannBasedMatcher(index_parameters, search_parameters) 
 
    #利用knnMatch匹配处理，并将结果返回给matches 
    matches = flann.knnMatch(features1,features2,k=2) 
 
    #对符合要求的匹配结果进行输出 
    good = [] 
    for m,n in matches: 
        if m.distance < 0.7*n.distance: 
            good.append(m) 
 
    matching = cv2.drawMatchesKnn(img1,features1,img2,features1,good,None,flags=2) 
    

    if len(good) > MIN_MATCH_COUNT:
        #得到两幅待拼接图的匹配点集
        #src = source;dst = destination;pts = points
        src_pts = np.float32([ keypoints1[m.queryIdx].pt for m in good ]).reshape(-1, 1, 2)
        dst_pts = np.float32([ keypoints2[m.trainIdx].pt for m in good ]).reshape(-1, 1, 2)

        #cv2.findHomography：传入两个图像里的点集合,返回目标的透视转换
        #第三个参数用于计算单应矩阵的方法
        #第四个参数为误差阈值
        #返回值：H为变换矩阵,mask为掩膜
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        #cv2.warpPerspective：利用M矩阵对原图进行透视变换（M点乘位置向量）
        wrap = cv2.warpPerspective(img2, H, (img2.shape[1]+img2.shape[1] , img2.shape[0]+img2.shape[0]))
        wrap[0:img2.shape[0], 0:img2.shape[1]] = img1

        rows, cols = np.where(wrap[:,:,0] !=0)
        min_row, max_row = min(rows), max(rows) +1
        min_col, max_col = min(cols), max(cols) +1
        #去除黑色无用部分
        result = wrap[min_row:max_row,min_col:max_col,:]

        return matching, result

if __name__ == '__main__':
    matching, result = SURF()
    cv2.imshow('matching',matching)
    cv2.imshow('result',result)
    cv2.waitKey(0)
    cv2.destroyAllWindows() 
    cv2.waitKey(1)
    cv2.waitKey(1)
    cv2.waitKey(1)
    cv2.waitKey(1)