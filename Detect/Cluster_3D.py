import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import ndimage
from skimage import transform, filters, segmentation, measure, morphology
from astropy.io import fits


def Get_Neighbors(delta_i_xyz, xres, yres, zres, r):
    neighbors = []
    it = delta_i_xyz[0]
    jt = delta_i_xyz[1]
    zt = delta_i_xyz[2]
    x_arange = np.arange(max(0, it - r), min(xres, it + r + 1))
    y_arange = np.arange(max(0, jt - r), min(yres, jt + r + 1))
    z_arange = np.arange(max(0, zt - r), min(zres, zt + r + 1))
    [xx, yy, zz] = np.meshgrid(x_arange, y_arange, z_arange)
    xxyyzz = np.column_stack([xx.flat, yy.flat, zz.flat])
    for i in range(xxyyzz.shape[0]):
        if 1 - all(delta_i_xyz == xxyyzz[i]):
            neighbors.append([xxyyzz[i][0], xxyyzz[i][1], xxyyzz[i][2]])
    neighbors = np.array(neighbors)
    return neighbors


def Dist_AB(point_a, point_b):
    temp = point_a - point_b
    distance = math.sqrt(temp[0] ** 2 + temp[1] ** 2 + temp[2] ** 2)
    return distance


def Get_Near_Index(distance, gradient):
    distance = np.array(distance)
    gradient = np.array(gradient)
    unique_dist = sorted(list(set(distance)))
    nmg_index = -1
    for u_dist in unique_dist:
        near_u_dist = np.where(distance == u_dist)
        near_max_gradient = gradient[near_u_dist].max()
        if near_max_gradient >= 0:
            gradient_index = np.where(gradient[near_u_dist] == near_max_gradient)[0][0]
            nmg_index = near_u_dist[0][gradient_index]
            break
    return nmg_index


def Build_DGI(origin_data, rms, xyz):
    xres, yres, zres = origin_data.shape
    maxd = xres + yres + zres
    ravel_data = origin_data.ravel()
    rd_sort_index = np.argsort(-ravel_data)
    rd_sorted = ravel_data[rd_sort_index]
    length = len(ravel_data)
    Distance = np.zeros(length)
    IndNearNeigh = np.zeros(length)
    Gradient = np.zeros(length)
    IndNearNeigh[rd_sort_index[0]] = -1
    r = 1
    for i in range(1, length):
        print(i)
        rdsi_i = rd_sort_index[i]
        rd_i = rd_sorted[i]
        if rd_i >= rms:
            distance = []
            gradient = []
            Distance[rdsi_i] = maxd
            delta_i_xyz = xyz[rdsi_i, :]
            neighbors = Get_Neighbors(delta_i_xyz, xres, yres, zres, r)
            for neighbor in neighbors:
                dist = Dist_AB(delta_i_xyz, neighbor)
                distance.append(dist)
                rd_neighbor = origin_data[neighbor[0], neighbor[1], neighbor[2]]
                #                 gradient.append((rd_neighbor - rd_i)/dist)
                gradient.append(rd_neighbor - rd_i)
            nmg_index = Get_Near_Index(distance, gradient)
            if nmg_index != -1:
                Distance[rdsi_i] = distance[nmg_index]
                Gradient[rdsi_i] = gradient[nmg_index]
                IndNearNeigh[rdsi_i] = neighbors[nmg_index][0] * yres * zres \
                                       + neighbors[nmg_index][1] * zres + neighbors[nmg_index][2];

            if Distance[rdsi_i] == maxd:
                distance_1 = []
                gradient_1 = []
                for j in range(i):
                    rdsi_j = rd_sort_index[j]
                    rd_j = rd_sorted[j]

                    dist = Dist_AB(delta_i_xyz, xyz[rdsi_j, :])
                    distance_1.append(dist)
                    #                     gradient_1.append((rdsi_j - rdsi_i) / dist)
                    gradient_1.append(rd_j - rd_i)
                nmg_index = Get_Near_Index(distance_1, gradient_1)
                #             print(nmg_index)
                if nmg_index != -1:
                    Distance[rdsi_i] = distance_1[nmg_index]
                    Gradient[rdsi_i] = gradient_1[nmg_index]
                    neighbor_j = xyz[rd_sort_index[nmg_index], :]
                    IndNearNeigh[rdsi_i] = neighbor_j[0] * yres * zres + neighbor_j[1] * zres + neighbor_j[2]
        else:
            IndNearNeigh[rdsi_i] = length
    IndNearNeigh = np.asarray(IndNearNeigh, dtype=np.int64)  # 正方体边长最大值21845
    Distance[rd_sort_index[0]] = Distance.max()

    return Distance, Gradient, IndNearNeigh,


def Get_Infor(origin_data, Distance, Gradient, IndNearNeigh, rhomin, v_min, deltamin):
    NCLUST = 0
    ravel_data = origin_data.ravel()
    rd_sort_index = np.argsort(-ravel_data)
    rd_sorted = ravel_data[rd_sort_index]
    length = len(ravel_data)
    clust_order = -1 * np.ones(length + 1)

    # 将满足密度，距离最低值的点选出来
    clust_index = np.where(np.logical_and(ravel_data >= rhomin, Distance > deltamin) == True)[0]
    #     print(clust_index)
    clust_num = clust_index.shape[0]
    for i in range(clust_num):
        j = clust_index[i]
        clust_order[j] = NCLUST
        NCLUST += 1

    for i in range(length):
        rdsi_i = rd_sort_index[i]
        if clust_order[rdsi_i] == -1:
            clust_order[rdsi_i] = clust_order[IndNearNeigh[rdsi_i]]
        else:
            Gradient[rdsi_i] = -1
    clust_order = clust_order[:length]
    clustVolume = np.zeros(NCLUST)
    for i in range(NCLUST):
        clustVolume[i] = clust_order[clust_order == i].shape[0]
    #     print(clustVolume)
    centInd = clust_index[clustVolume >= v_min]
    centNum = np.where(clustVolume >= v_min)[0]

    #     cluster_info=[Distance[clust_index],ravel_data[clust_index],clustVolume];

    return clust_order, centInd, centNum


def Get_Mask_Out(origin_data, xyz, clust_order, Gradient, gradmin, centNum):
    xres, yres, zres = origin_data.shape
    ravel_data = origin_data.ravel()
    mask = np.zeros([xres, yres, zres])
    output = np.zeros([xres, yres, zres])
    co_real = -1 * np.ones(clust_order.shape[0])
    mask_grad = np.where(Gradient >= gradmin)[0]
    for i in range(centNum.shape[0]):  # centNum.shape[0]
        rd_clust_i = np.zeros(ravel_data.shape[0])
        index_clust_i = np.where(clust_order == centNum[i])[0]  # 得到第i类所有点的索引
        #         print(index_clust_i.shape[0])
        index_cc = set(mask_grad).intersection(index_clust_i)  # 得到第i类梯度大于gradtmin的点的索引
        index_cc = np.asarray(list(index_cc))
        #         print(index_cc.shape[0])

        rd_clust_i[index_clust_i] = ravel_data[index_clust_i]  # 得到第i类所有点的密度
        rd_cc_mean = ravel_data[index_cc].mean()  # 得到第i类梯度大于gradtmin的点的密度的均值
        index_cc_rd = np.where(rd_clust_i > rd_cc_mean)[0]  # 得到第i类密度大于该阈值的点的索引
        index_clust_rd = set(index_cc).union(index_cc_rd)  # 得到两个条件都满足的点的索引
        index_clust_rd = np.asarray(list(index_clust_rd))
        co_real[index_clust_rd] = centNum[i]
        cl_i_point = xyz[index_clust_rd, :]  # 得到第i类梯度大于gradtmin的点的坐标

        mask_out = np.zeros([xres, yres, zres])
        for j in range(cl_i_point.shape[0]):
            mask_out[cl_i_point[j, 0], cl_i_point[j, 1], cl_i_point[j, 2]] = 1
        mo_close = morphology.closing(mask_out)  # 闭运算
        mo_fill = ndimage.binary_fill_holes(mo_close).astype(int)
        mo_fill_label = measure.label(mo_fill)
        regions = measure.regionprops(mo_fill_label)
        area = []
        for region in regions:
            area.append(region.area)
        area = np.array(area)
        area_index = np.where(area == area.max())[0][0] + 1
        mo_fill[mo_fill_label != area_index] = 0
        mask_clust = mo_fill * (i + 1)
        mask += mask_clust
        output += mo_fill * origin_data

    return co_real, mask, output


def Core_Describe(origin_data, xyz, co_real, centInd, centNum):
    dim = len(origin_data.shape)
    NCLUST = centNum.shape[0]
    [xres, yres, zres] = origin_data.shape
    clustSum = np.zeros(NCLUST)
    clustVolume = np.zeros(NCLUST)
    clustPeak = np.zeros(NCLUST)
    clumpCenter = np.zeros([NCLUST, dim])
    clustSize = np.zeros([NCLUST, dim])
    clumpPeakIndex = xyz[centInd]
    describe = {}
    for i in range(NCLUST):
        clust_coords = xyz[co_real == centNum[i]]
        intensity = origin_data[clust_coords[:, 0], clust_coords[:, 1], clust_coords[:, 2]]
        clustSum[i] = intensity.sum()
        clustVolume[i] = clust_coords.shape[0]
        clustPeak[i] = origin_data[clumpPeakIndex[i][0], clumpPeakIndex[i][1], clumpPeakIndex[i][2]]
        x_center = np.sum((clust_coords[:, 0] + 1) * intensity) / clustSum[i] - 1
        y_center = np.sum((clust_coords[:, 1] + 1) * intensity) / clustSum[i] - 1
        z_center = np.sum((clust_coords[:, 2] + 1) * intensity) / clustSum[i] - 1
        clumpCenter[i] = [x_center, y_center, z_center]
        center_delta = clust_coords - clumpCenter[i]
        od = origin_data[clust_coords[:, 0], clust_coords[:, 1], clust_coords[:, 2]]
        center_delta = clust_coords - clumpCenter[i]
        od = origin_data[clust_coords[:, 0], clust_coords[:, 1], clust_coords[:, 2]]
        clustSize[i] = (np.array(np.mat(od) * np.array(np.mat(center_delta)) ** 2 / clustSum[i]) \
                        - np.array(np.mat(od) * np.mat(center_delta) / clustSum[i]) ** 2) ** (1 / 2)
    describe['clustSum'] = clustSum
    describe['clustVolume'] = clustVolume
    describe['clumpPeakIndex'] = clumpPeakIndex
    describe['clustPeak'] = clustPeak
    describe['clumpCenter'] = clumpCenter
    describe['clustSize'] = clustSize
    return describe
