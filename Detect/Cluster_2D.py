from scipy import ndimage
from skimage import filters, morphology, measure
import numpy as np


def densityCluster_2d(data, xx, is_plot, gradmin, rhomin, deltamin, v_min, rms, sigma):
    data_filter = filters.gaussian(data, sigma)
    size_x, size_y = data.shape
    size_z = 1
    rho = data_filter.T.flatten()
    ordrhoInd = np.argsort(-rho)
    rho_sorted = np.sort(-rho) * (-1)
    maxd = size_z + size_y + size_x
    ND = len(rho)
    delta, IndNearNeigh, Gradient = np.zeros(ND), np.zeros(ND), np.zeros(ND)
    delta[ordrhoInd[0]] = -1
    IndNearNeigh[ordrhoInd[0]] = 0
    for ii in range(1, ND):
        ordrho_ii = ordrhoInd[ii]
        rho_ii = rho_sorted[ii]
        if rho_ii >= rms:
            delta[ordrho_ii] = maxd
            delta_ii_xy = xx[ordrho_ii, :]
            bt = kc_coord_2d(delta_ii_xy, size_y, size_x)
            for j_ in range(0, bt.shape[0]):
                rho_jj = data_filter[int(bt[j_, 0] - 1), int(bt[j_, 1] - 1)]
                dist_i_j = dist_xyz(delta_ii_xy, bt[j_, :])
                gradient = (rho_jj - rho_ii) / dist_i_j
                if dist_i_j <= delta[ordrho_ii] and gradient >= Gradient[ordrho_ii]:
                    delta[ordrho_ii] = dist_i_j
                    Gradient[ordrho_ii] = gradient
                    IndNearNeigh[ordrho_ii] = (bt[j_, 1] - 1) * size_x + bt[j_, 0]
            if delta[ordrho_ii] == maxd:
                for jj in range(0, ii):
                    rho_jj = rho_sorted[jj]
                    dist_i_j = distx(ordrho_ii, ordrhoInd[jj], xx)
                    gradient = (rho_jj - rho_ii) / dist_i_j
                    if dist_i_j <= delta[ordrho_ii] and gradient >= Gradient[ordrho_ii]:
                        delta[ordrho_ii] = dist_i_j
                        Gradient[ordrho_ii] = gradient
                        IndNearNeigh[ordrho_ii] = ordrhoInd[jj]
        else:
            IndNearNeigh[ordrho_ii] = ND + 1
    delta[ordrhoInd[0]] = max(delta[:])
    print('delta, rho and Gradient are ok')
    gama = rho * delta
    gama_sorted = np.sort(-gama) * (-1)
    NCLUST = 0
    clustInd = -1 * np.ones(ND + 1)
    clust_index = np.intersect1d(np.where(rho > rhomin), np.where(delta > deltamin))
    clust_num = len(clust_index)
    icl = np.zeros(clust_num, dtype=int)
    for ii in range(0, clust_num):
        i = clust_index[ii]
        NCLUST = NCLUST + 1
        clustInd[i] = NCLUST
        icl[NCLUST - 1] = i
    if is_plot == 1:
        delta = delta / max(delta[:])
        # figure
    for i in range(0, ND):
        ordrho_i = ordrhoInd[i]
        if clustInd[ordrho_i] == -1:
            clustInd[ordrho_i] = clustInd[int(IndNearNeigh[ordrho_i]) - 1]
        else:
            Gradient[ordrho_i] = -1
    clustVolume = np.zeros(NCLUST)
    for i in range(0, NCLUST):
        clustVolume[i] = clustInd.tolist().count(i + 1)
    cluster_info = np.zeros([clustVolume.size, 3])
    cluster_info[:, 0] = delta[icl]
    cluster_info[:, 1] = rho[icl]
    cluster_info[:, 2] = clustVolume
    centInd = []
    for i in range(0, clustVolume.size):
        if clustVolume[i] >= v_min:
            centInd.append(icl[i])
    centInd = np.array(centInd)
    centNum = np.where(clustVolume >= v_min)[0]
    temp = np.zeros([centInd.size, 2], dtype=int)
    temp[:, 0] = centInd
    temp[:, 1] = centNum
    centInd = temp
    clustInd_re = -1 * np.ones(clustInd.size)
    mask = np.zeros([size_x, size_y])
    out = np.zeros([size_x, size_y])
    mask_grad = np.where(Gradient > gradmin)[0]
    for i in range(0, centInd.shape[0]):
        rho_clust_i = np.zeros(rho.size)
        index_clust_i = np.where(clustInd == (centInd[i, 1] + 1))[0]
        index_cc = np.array([val for val in mask_grad if val in index_clust_i], dtype=int)
        rho_clust_i[index_clust_i] = rho[index_clust_i]
        rho_cc_mean = rho[index_cc].mean()
        index_cc_rho = np.where(rho_clust_i > rho_cc_mean)[0]
        index_clust_rho = np.sort(np.array(list(set(index_cc).union(set(index_cc_rho))), dtype=int))
        clustInd_re[index_clust_rho] = centInd[i, 1]
        cl_i_point = xx[index_clust_rho, :]
        mask_out = np.zeros([size_x, size_y])
        for j in range(0, cl_i_point.shape[0]):
            mask_out[cl_i_point[j, 0] - 1, cl_i_point[j, 1] - 1] = 1
        bw = morphology.closing(mask_out)
        BW2 = ndimage.binary_fill_holes(bw).astype(int)
        L = measure.label(BW2)
        STATS = measure.regionprops(BW2)
        Ar = []
        for region in STATS:
            Ar.append(region.area)
        Ar = np.array(Ar)
        ind = np.where(Ar == Ar.max())[0][0] + 1
        BW2[L != ind] = 0
        mask_clust = BW2 * (i + 1)
        mask = mask + mask_clust
        out = out + BW2 * data
    NCLUST_ = centInd.shape[0]
    return NCLUST_, centInd, clustInd, cluster_info, mask, out, Gradient, clustInd_re


def kc_coord_2d(delta_ii_xy, xm, ym):
    jt = delta_ii_xy[0]
    it = delta_ii_xy[1]
    r = 3
    p_i, p_j = np.mgrid[max(1, it - r): min(ym, it + r) + 1, max(1, jt - r): min(xm, jt + r) + 1]
    index_value = np.zeros([p_i.size, 2])
    index_value[:, 0] = p_j.flatten()
    index_value[:, 1] = p_i.flatten()
    return index_value


def dist_xyz(point_a, point_b):
    temp = point_a - point_b
    distance = np.sqrt(sum(temp ** 2))
    return distance


def distx(kc1, kc2, xx):
    distance = np.sqrt((xx[kc1, 0] - xx[kc2, 0]) ** 2 + (xx[kc1, 1] - xx[kc2, 1]) ** 2)
    return distance


def extroclump_parameters(NCLUST, xx, clustInd, centInd, data):
    dim = xx.shape[1]
    clustSum, clustVolume, clustPeak = np.zeros([NCLUST, 1]), np.zeros([NCLUST, 1]), np.zeros([NCLUST, 1])
    clump_Cen, clustSize = np.zeros([NCLUST, dim]), np.zeros([NCLUST, dim])
    size_x, size_y = data.shape
    clump_Peak = xx[centInd[:, 0], :]
    precent = 1
    cl_result = np.zeros_like(data, dtype=int)
    count = 0
    for i in range(0, NCLUST):
        cl_1_index_1 = []
        cl_i = np.zeros([size_x, size_y])
        for ii in range(0, clustInd.size):
            if clustInd[ii] == (centInd[i, 1] + 1):
                cl_1_index_1.append(xx[ii, :])
        cl_1_index_ = np.array(cl_1_index_1)
        clustNum = cl_1_index_.shape[0]
        clump_sum_ = np.zeros(clustNum)
        for j in range(0, clustNum):
            cl_i[cl_1_index_[j, 0], cl_1_index_[j, 1]] = i + 1
            clump_sum_[j] = data[cl_1_index_[j, 0], cl_1_index_[j, 1]]
        clustsum = sum(clump_sum_)
        clump_Cen[i, :] = np.matmul(clump_sum_, cl_1_index_) / clustsum
        clustVolume[i, 0] = clustNum
        clustSum[i, 0] = clustsum
        clustPeak[i, 0] = data[clump_Peak[i, 0] - 1, clump_Peak[i, 1] - 1]
        x_i = cl_1_index_ - clump_Cen[i, :]
        clustSize[i, :] = np.sqrt((np.matmul(clump_sum_, x_i ** 2) / clustsum)
                                    - (np.matmul(clump_sum_, x_i) / clustsum) ** 2)
            # temp = cl_i * data
            # temp_sort = np.sort(-temp.flatten()) * (-1.0)
            # inde_end = round(precent * cl_1_index_.shape[0])
            # temp_mean = temp_sort[inde_end].mean()
            # temp[temp < temp_mean] = 0
            # temp[temp >= temp_mean] = i + 1
        cl_result = cl_result + cl_i
    outcat = np.column_stack((clump_Peak, clump_Cen, clustSize, clustSum, clustPeak, clustVolume))
    return outcat


def localDenCluster(data, is_plot, gradmin, rhomin, deltamin, v_min, rms, sigma):
    print("2d")
    size_x, size_y = data.shape
    p_i, p_j = np.mgrid[1: size_x + 1, 1: size_y + 1]
    xx = np.zeros([p_i.size, 2], dtype=int)
    xx[:, 0] = p_j.flatten()
    xx[:, 1] = p_i.flatten()
    NClust, centInd, clustInd, cluster_info, mask, out, Gradient, clustInd_re = densityCluster_2d(data, xx, is_plot,
                                                                                                  gradmin, rhomin,
                                                                                                  deltamin, v_min, rms,
                                                                                                  sigma)
    outcat = extroclump_parameters(NClust, xx, clustInd, centInd, data)
    return outcat, out, mask, centInd
