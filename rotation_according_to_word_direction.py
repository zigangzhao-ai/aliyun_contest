import numpy as np
import cv2

def rotated_img_with_fft(gray_img):
    #cut & extend image so it will be easier for FFT algorithm to calculate
    h, w = gray.shape[:2]
    new_h = cv2.getOptimalDFTSize(h)
    new_w = cv2.getOptimalDFTSize(w)
    right = new_w - w
    bottom = new_h - h
    nimg = cv2.copyMakeBorder(gray, 0, bottom, 0, right, borderType=cv2.BORDER_CONSTANT, value=0)

    #perform FFT
    f = np.fft.fft2(nimg)
    fshift = np.fft.fftshift(f)

    fft_img = np.log(np.abs(fshift))
    fft_img = (fft_img - np.amin(fft_img)) / (np.amax(fft_img) - np.amin(fft_img))

    fft_img *= 255

    #binarization
    ret, thresh = cv2.threshold(fft_img, 150, 255, cv2.THRESH_BINARY)

    #perform Hough transform
    thresh = thresh.astype(np.uint8)
    lines = cv2.HoughLinesP(thresh, 1, np.pi / 180, 30, minLineLength=40, maxLineGap=100)
    try:
        lines1 = lines[:, 0, :]
    except Exception as e:
        lines1 = []

    # create a new image using the lines before
    # lineimg = np.ones(nimg.shape,dtype=np.uint8)
    # lineimg = lineimg * 255

    piThresh = np.pi / 180
    pi2 = np.pi / 2
    angle = 0
    for line in lines1:
        # x1, y1, x2, y2 = line[0]
        x1, y1, x2, y2 = line
        # cv2.line(lineimg, (x1, y1), (x2, y2), (0, 255, 0), 2)

        if x2 - x1 == 0:
            continue

        #little bias is allowed
        else:
            #theta is approximately equal to tan(theta)
            theta = (y2 - y1) / (x2 - x1)
        if abs(theta) < piThresh or abs(theta - pi2) < piThresh:
            continue
        else:
            angle = abs(theta)
            break
    
    angle = math.atan(angle)
    angle = angle * (180 / np.pi)
    print(angle)
    # cv2.imshow("line image", lineimg)
    center = (w // 2, h // 2)
    height_1 = int(w * fabs(sin(radians(angle))) + h * fabs(cos(radians(angle))))
    width_1 = int(h * fabs(sin(radians(angle))) + w * fabs(cos(radians(angle))))

    #get the matrix to transform the image 
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    M[0, 2] += (width_1 - w) / 2
    M[1, 2] += (height_1 - h) / 2

    #rotate the image using the matrix before
    rotated = cv2.warpAffine(gray, M, (width_1, height_1), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    cv2.imshow('rotated', rotated)
    cv2.waitKey(0)
    return rotated