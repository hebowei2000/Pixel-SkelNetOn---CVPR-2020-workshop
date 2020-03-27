import cv2
import os
import glob
import matplotlib.pyplot as plt
import numpy as np
import math

sigma1 = sigma2 = 1
sum = 0
gaussian = np.zeros([5,5])
for i in range(5):
    for j in range(5):
        gaussian[i,j] = math.exp(-1/2 * (np.square(i-3)/np.square(sigma1)
            +(np.square(j-3)/np.squre(sigma2)))) / (2*math.pi*sigma1*sigma2)
        sum = sum + gaussian[i,j]

gaussian = gaussian/sum

PATH = '../../data/result25'
paths = glob.glob(os.path.join(PATH,'*.png'))
paths.sort()

#canny operator process
for path in paths:
    #Step1:Gaussion filter
    img = cv2.imread(path)
    W,H = img.shape
    new_gray = np.zeros([W-5,H-5])
    for i in range(W-5):
        for j in range(H-5):
            new_gray[i,j] = np.sum(img[i:i+5,j:j+5]*gaussian)

    #Step2: get the gradient
    W1,H1 = new_gray.shape
    dx = np.zeros([W1-1,H1-1])
    dy = np.zeros([W1-1,H1-1])
    d = np.zeros([W1-1,H1-1])
    for i in range(W1-1):
        for j in range(H1-1):
            dx[i,j] = new_gray[i,j+1] - new_gray[i,j]
            dy[i,j] = new_gray[i+1,j] - new_gray[i,j]
            d[i,j] = np.sqrt(np.square(dx[i,j])+np.square(dy[i,j]))

    #step3 non-maximal suppression
    W2,H2 = d.shape
    NMS = np.copy(d)
    NMS[0,:] = NMS[W2-1,:] = NMS[:,0] = NMS[:,H2-2] = 0
    for i in range(1,W2-1):
        for j in range(1,H2-1):

            if d[i,j] == 0:
                NMS[i,j] = 0
            else:
                gradX = dx[i,j]
                gradY = dy[i,j]
                gradTemp = d[i,j]

                # if gradient value on the y axis is larger
                if np.abs(graY) > np.abs(gradX):
                    weight = np.abs(gradX)/np.abs(gradY)
                    grad2 = d[i-1,j]
                    grad4 = d[i+1,j]
                    # if the sign of gradX corrsponds to gradY
                    if gradX*gradY > 0:
                        grad1 = d[i-1,j-1]
                        grad3 = d[i+1,j+1]
                    #if the sign of gradX not correspond to gradY
                    else:
                        grad1 = d[i-1,j+1]
                        grad3 = d[i+1,j-1]
                
                #if gradient value on the x axix is larger
                else:
                    weight = np.abs(gradX) > np.abs(gradY)
                    grad2 = d[i,j-1]
                    grad4 = d[i,j+1]

                    #if the sign of gradX corresponds to gradY
                    if gradX*gradY > 0:
                        grad1 = d[i+1,j-1]
                        grad3 = d[i-1,j+1]
                    #
                    else:
                        grad1 = d[i-1,j-1]
                        grad3 = d[i+1,j+1]

          
            gradTemp1 = weight*grad1 + (1-weight)*grad2
            gradTemp2 = weight*grad3 + (1-weight)*grad4
            if gradTemp >= gradTemp1 and gradTemp >= gradTemp2:
                NMS[i,j] = gradTemp
            else:
                NMS[i,j] = 0


    #step4 bi-threshold algorithm detection and link the borders
    W3,H3 = NMS.shape
    DT = np.zeros([W3,H3])
    #define the high and low threshold
    TL = 0.2*np.max(NMS)
    TH = 0.3*np.max(NMS)
    for i in range(1,W3-1):
        for j in range(1,H3-1):
            if (NMS[i,j] < TL):
                DT[i,j] = 0;
            elif (NMS[i,j] > TH):
                DT[i,j] = 1;
            elif ((NMS[i-1,j-1:j+1] < TH).any() or (NMS[i+1,j-1:j+1]).any()
                    or (NMS[i,[j-1,j+1]] < TH).any()):
                DT[i,j] = 1
   #store the canny operated image
   cv2.imwrite(path,DT)
                    



