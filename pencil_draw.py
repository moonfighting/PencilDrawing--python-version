import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.sparse import spdiags
from scipy.sparse import dia_matrix
from scipy.sparse.linalg import cg

import math
import sys
import os


def rotate_img(img, angle):
    row, col = img.shape
    M = cv2.getRotationMatrix2D((row / 2, col / 2), angle, 1)
    res = cv2.warpAffine(img, M, (row, col))
    return res

def get_eight_directions(l_len):    
    L = np.zeros((8, l_len, l_len))
    half_len = (l_len + 1) / 2
    for i in range(8):
        if i == 0 or i == 1 or i == 2 or i == 7:
            for x in range(l_len):
                    y = half_len - int(round((x + 1 - half_len) * math.tan(math.pi * i / 8)))
                    if y >0 and y <= l_len:
                        L[i, x, y - 1 ] = 1
            if i != 7:
                L[i + 4] = np.rot90(L[i])
    L[3] = np.rot90(L[7], 3)
    return L



# compute and get the stroke of the raw img
def get_stroke(img, ks,  dirNum):
    height , width = img.shape[0], img.shape[1]
    img = np.float32(img) / 255.0
    img = cv2.medianBlur(img, 3)
    #cv2.imshow('blur', img)
    imX = np.append(np.absolute(img[:, 0 : width - 1] - img[:, 1 : width]), np.zeros((height, 1)), axis = 1)
    imY = np.append(np.absolute(img[0 : width - 1, :] - img[1 : width, :]), np.zeros((1, width)), axis = 0)
    #img_gredient = np.sqrt((imX ** 2 + imY ** 2))
    img_gredient = imX + imY

    kernel_Ref = np.zeros((ks * 2 + 1, ks * 2 + 1))
    kernel_Ref [ks, :] = 1

    response = np.zeros((dirNum, height, width))
    L = get_eight_directions(2 * ks + 1)
    for n in range(dirNum):
        ker = rotate_img(kernel_Ref, n * 180 / dirNum)
        response[n, :, :] = cv2.filter2D(img, -1, ker)

    Cs = np.zeros((dirNum, height, width))
    for x in range(width):
        for y in range(height):
            i = np.argmax(response[:,y,x])
            Cs[i, y, x] = img_gredient[y,x]

    spn = np.zeros((8, img.shape[0], img.shape[1]))

    kernel_Ref = np.zeros((2 * ks + 1, 2 * ks + 1))
    kernel_Ref [ks, :] = 1
    for n in range(width):
        if (ks - n) >= 0:
            kernel_Ref[ks  - n, :] = 1
        if (ks + n)  < ks * 2:
            kernel_Ref[ks + n, :] = 1

    kernel_Ref = np.zeros((2*ks + 1, 2 * ks + 1))
    kernel_Ref [ks, :] = 1

    for i in range(8):
        ker = rotate_img(kernel_Ref, i * 180 / dirNum)
        spn[i]= cv2.filter2D(Cs[i], -1, ker)

    sp = np.sum(spn, axis = 0)
    sp =  (sp - np.min(sp)) / (np.max(sp) - np.min(sp))
    S = 1 -  sp

    return S


#for histogram matching
def natural_histogram_matching(img):
    ho = np.zeros( 256)
    po = np.zeros( 256)
    for i in range(256):
        po[i] = np.sum(img == i)
    po = po / np.sum(po)
    ho[0] = po[0]
    for i in range(1,256):
        ho[i] = ho[i - 1] + po[i]

    p1 = lambda x : (1 / 9.0) * np.exp(-(255 - x) / 9.0)
    p2 = lambda x : (1.0 / (225 - 105)) * (x >= 105 and x <= 225)
    p3 = lambda x : (1.0 / np.sqrt(2 * math.pi *11) ) * np.exp(-((x - 90) ** 2) / float((2 * (11 **2))))
    p = lambda x : (76 * p1(x) +22 * p2(x) + 2 * p3(x)) * 0.01
    prob = np.zeros(256)
    histo = np.zeros(256)
    total = 0
    for i in range(256):
        prob[i] = p(i)
        total = total + prob[i]
    prob = prob / np.sum(prob)

    histo[0] = prob[0]
    for i in range(1, 256):
        histo[i] = histo[i - 1] + prob[i]

    Iadjusted = np.zeros((img.shape[0], img.shape[1]))
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            histogram_value = ho[img[x,y]]
            i = np.argmin(np.absolute(histo - histogram_value))
            Iadjusted[x, y] = i

    Iadjusted = np.float64(Iadjusted) / 255.0
    return Iadjusted


#compute the tone map
def get_toneMap(img, P):
    P = np.float64(P) / 255.0
    J = natural_histogram_matching(img)
    J = cv2.blur(J, (10, 10))
    theta = 0.2

    height, width = img.shape

    P = cv2.resize(P, (height, width))
    P = P.reshape((1, height * width))
    logP = np.log(P)
    logP = spdiags(logP, 0, width * height, width * height)


    J = cv2.resize(J, (height, width))
    J = J.reshape( height * width)
    logJ = np.log(J)

    e = np.ones(width * height)

    Dx = spdiags([-e, e], np.array([0, height]), width *height, width * height)
    Dy = spdiags([-e, e], np.array([0, 1]), width * height, width * height)


    A = theta * (Dx.dot(Dx.transpose()) + Dy.dot(Dy.transpose())) + logP.dot(logP.transpose())

    b= logP.transpose().dot(logJ)
    beta, info = cg(A, b , tol = 1e-6, maxiter = 60)

    beta = beta.reshape((height, width))
    P = P.reshape((height, width))
    T = np.power(P, beta)

    return T

def pencil_drawing(img_path, pencil_texture):
    P = cv2.imread(pencil_texture, cv2.IMREAD_GRAYSCALE)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    S = get_stroke(img, 3, 8)
    T = get_toneMap(img, P)

    res = S * T

    return res



if __name__ == '__main__':

    if len(sys.argv) < 3:
        print 'usage: python %s [img] [pencil texture]' % os.path.basename(sys.argv[0])
    img_path = sys.argv[1]
    pencil_texture = sys.argv[2]

    #img_path = 'lena.jpg'
    #pencil_texture = 'pencil0.jpg'
    res = pencil_drawing(img_path, pencil_texture)
    cv2.imshow('res', res)
    cv2.waitKey(0)