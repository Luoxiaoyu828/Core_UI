import numpy as np
import matplotlib.pyplot as plt


def brightness_distribution(x, y, v, x0, y0, v0, sigma, A, p):
    s = np.sqrt((x - x0) ** 2 + (y - y0) ** 2)
    temp1 = 1 / (1 + (s / sigma) ** p)
    temp2 = np.exp(-(v - v0) ** 2 / (2 * sigma ** 2))
    T = A * temp1 * temp2
    return T


"""
TT = np.zeros([IMG_x, IMG_y, IMG_v], np.float64)
for i in range(100):
    if i%10==0:
        print(i)
    for j in range(100):
        for k in range(100):
            x = xx[i, j, k]
            y = yy[i, j, k]
            v = vv[i, j, k]
            TT[i, j, k] = brightness_distribution(x, y, v, x0, y0, v0, sigma, A)
"""


def make_clumps(n_clumps, sigma_, A_, IMG_x_, IMG_y_, IMG_v_, p):
    xx, yy, vv = np.mgrid[0:IMG_x_, 0:IMG_y_, 0:IMG_v_]
    T = np.zeros([IMG_x_, IMG_y_, IMG_v_], np.float64)
    info_dict = []
    for item in range(n_clumps):
        [x0, y0, v0] = np.random.uniform([15, 15, 15], [IMG_x_ - 15, IMG_y_ - 15, IMG_v_ - 15])
        A = np.random.uniform(low=A_[0], high=A_[1], size=[1])
        sigma = np.random.uniform(low=sigma_[0], high=sigma_[1], size=[1])
        temp = brightness_distribution(xx, yy, vv, x0, y0, v0, sigma[0], A[0], p)
        peak1, peak2, peak3 = np.where(temp == temp.max())
        Sum = temp.sum()
        Volume = len(np.where(temp > 0.01)[0])
        temp1 = [peak1[0], peak2[0], peak3[0], x0, y0, v0, sigma[0], sigma[0], sigma[0], Sum, A[0], Volume, -1]
        T += temp
        info_dict.append(temp1)
    return T, info_dict



