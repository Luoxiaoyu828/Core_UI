import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from skimage import filters, measure, morphology
import astropy.io.fits as fits




def Get_Regions(origin_data, n_dilation=0, region_pixels_min=50, thresh='mean'):
    n_find = 1
    kopen_radius = 2
    open_data = np.zeros_like(origin_data)
    core_data = np.zeros_like(origin_data)
    if thresh == 'mean':
        thresh = origin_data.mean()
    elif thresh == 'otsu':
        thresh = filters.threshold_otsu(origin_data)
    else:
        thresh = thresh
    count = 0
    start = time.time()
    while (count < n_find):
        regions = []
        for i in range(open_data.shape[2]):
            open_data[:, :, i] = morphology.opening(origin_data[:, :, i] > thresh, morphology.square(kopen_radius))
        #         co = morphology.opening(origin_data > thresh,morphology.cube(kopen_radius))
        thresh += 0.01
        #         cleared = co.copy()
        #         segmentation.clear_border(cleared)
        label_data = measure.label(open_data)
        #         borders = np.logical_xor(co, cleared) #异或
        #         label_data[borders] = -1
        for i in range(n_dilation):
            label_data = morphology.dilation(label_data, morphology.ball(1))
        old_regions = measure.regionprops(label_data)
        for region in old_regions:
            if region.area > region_pixels_min and region.area:
                regions.append(region)
        count = len(regions)
        if count == 0:
            raise Exception('thresh,kopen_radius,region_pixels_min参数有误或没有云核！')
        end = time.time()
        if end - start > 10:
            raise Exception('调参数n_dilation，kopen_radius，region_pixels_min！')
    for i in range(len(regions)):
        label_x = regions[i].coords[:, 0]
        label_y = regions[i].coords[:, 1]
        label_z = regions[i].coords[:, 2]
        core_data[label_x, label_y, label_z] = origin_data[label_x, label_y, label_z]
    return regions, core_data


def Get_New_Center(core_data, cores_coordinate):
    xres, yres, zres = core_data.shape
    neighbors = []
    x_center = cores_coordinate[0]
    y_center = cores_coordinate[1]
    z_center = cores_coordinate[2]
    x_arange = np.arange(max(0, x_center - 1), min(xres, x_center + 2))
    y_arange = np.arange(max(0, y_center - 1), min(yres, y_center + 2))
    z_arange = np.arange(max(0, z_center - 1), min(zres, z_center + 2))
    [x, y, z] = np.meshgrid(x_arange, y_arange, z_arange);
    xyz = np.column_stack([x.flat, y.flat, z.flat])
    for i in range(xyz.shape[0]):
        if 1 - all(cores_coordinate == xyz[i]):
            neighbors.append([xyz[i][0], xyz[i][1], xyz[i][2]])
    neighbors = np.array(neighbors)
    gradients = core_data[neighbors[:, 0], neighbors[:, 1], neighbors[:, 2]] \
                - core_data[x_center, y_center, z_center]
    #     while gradients.max() > slope:
    #         gradients_max = gradients.max()
    #         gradients = list(gradients)
    #         gradients.remove(gradients_max)
    #         gradients = np.array(gradients)
    if gradients.max() > 0:
        g_step = np.where(gradients == gradients.max())[0][0]
        new_center = neighbors[g_step]
    else:
        new_center = cores_coordinate
    return gradients, new_center


def Get_Peak(core_data, region):
    xres, yres, zres = core_data.shape
    peak = []
    coordinates = region.coords
    for i in range(coordinates.shape[0]):
        gradients, new_center = Get_New_Center(core_data, coordinates[i])
        if gradients.max() < 0:
            peak.append([coordinates[i][0], coordinates[i][1], coordinates[i][2]])
    #     for j in range(1,len(peak)):
    #         if peak[0][0]==peak[j][0] and peak[0][1]==peak[j][1]\
    #                 and peak[0][2]==peak[j][2]:
    #             regions = regions[:j]
    #             peak = peak[:j]
    #             break
    return peak


def Build_CP_Dict(peak, region, core_data):
    prob_core_num = len(peak)
    xres, yres, zres = core_data.shape
    core_dict = {}
    for i in range(prob_core_num):
        core_dict[i] = []
    coordinates = region.coords
    for i in range(coordinates.shape[0]):
        gradients, new_center = Get_New_Center(core_data, coordinates[i])
        while gradients.max() >= 0:
            gradients, new_center = Get_New_Center(core_data, new_center)
        for j in range(prob_core_num):
            if peak[j][0] == new_center[0] and peak[j][1] == new_center[1] and peak[j][2] == new_center[2]:
                core_dict[j].append([coordinates[i][0], coordinates[i][1], coordinates[i][2]])
    peak_dict = {}
    for i in range(len(peak)):
        peak_dict[i] = peak[i]
    return core_dict, peak_dict


def Update_CP_Dict(peak_dict, core_dict, core_data, core_pixels=50, peak_delta=0, nearest_dist=10):
    del_core_number = []
    length = len(peak_dict)
    peak_value = []
    mountain_size = []
    #     for key in peak_dict.keys():
    #         peak_value.append(core_data[peak_dict[key][0],peak_dict[key][1]])
    #     index = np.argsort(peak_value)
    for key in core_dict.keys():
        mountain_size.append(len(core_dict[key]))
    index = np.argsort(mountain_size)
    if len(index) == 1:
        return core_dict, peak_dict
    else:
        for i_num in index:
            distance = []
            for j_num in peak_dict.keys():
                distance.append(((peak_dict[j_num][0] - peak_dict[i_num][0]) ** 2 \
                                 + (peak_dict[j_num][1] - peak_dict[i_num][1]) ** 2 \
                                 + (peak_dict[j_num][2] - peak_dict[i_num][2]) ** 2) ** (1 / 2))
            temp_distance = distance.copy()
            temp_distance.remove(0)
            nearest_dis_0 = np.array(temp_distance).min()
            order = list(peak_dict.keys())[np.where(distance == nearest_dis_0)[0][0]]
            peak_delta_0 = core_data[peak_dict[order][0], peak_dict[order][1], peak_dict[order][2]] \
                           - core_data[peak_dict[i_num][0], peak_dict[i_num][1], peak_dict[i_num][2]]
            logic = (len(core_dict[i_num]) < core_pixels or peak_delta_0 > peak_delta) \
                    and nearest_dis_0 < nearest_dist
            if logic:
                core_dict[order] = core_dict[order] + core_dict[i_num]
                del_core_number.append(i_num)
                del core_dict[i_num]
                del peak_dict[i_num]
            if len(peak_dict.keys()) == 1:
                break
        return core_dict, peak_dict


def Detect_Infor_Dict(peak_dict, core_dict, origin_data):
    peak_value = []
    center = []
    clump_centroid = []
    clump_com = []
    clump_size = []
    clump_sum = []
    clump_volume = []
    clump_regions = []
    detect_infor_dict = {}
    k = 0
    regions_data = np.zeros_like(origin_data)
    for key in peak_dict.keys():
        k += 1
        core_x = np.array(core_dict[key])[:, 0]
        core_y = np.array(core_dict[key])[:, 1]
        core_z = np.array(core_dict[key])[:, 2]
        peak_value.append(origin_data[peak_dict[key][0], peak_dict[key][1], peak_dict[key][2]])
        center.append(peak_dict[key])
        regions_data[core_x, core_y, core_z] = k
        od_mass = origin_data[core_x, core_y, core_z]
        clump_com.append(np.around((np.c_[od_mass, od_mass, od_mass] * core_dict[key]).sum(0) \
                                   / od_mass.sum(), 0).tolist())
        clump_size.append([core_x.max() - core_x.min(), core_y.max() - core_y.min(), core_z.max() - core_z.min()])
        clump_sum.append(origin_data[core_x, core_y, core_z].sum())
        clump_volume.append(len(core_dict[key]))
        clump_regions.append(
            [np.array(core_dict[key])[:, 0], np.array(core_dict[key])[:, 1], np.array(core_dict[key])[:, 2]])
    detect_infor_dict['peak'] = np.around(peak_value, 3).tolist()
    detect_infor_dict['center'] = np.array(center).tolist()
    detect_infor_dict['clump_com'] = clump_com
    detect_infor_dict['clump_size'] = clump_size
    detect_infor_dict['clump_sum'] = np.around(clump_sum, 3).tolist()
    detect_infor_dict['clump_volume'] = clump_volume
    detect_infor_dict['regions_data'] = regions_data
    detect_infor_dict['clump_regions'] = clump_regions
    return detect_infor_dict


def Label_Core_Data(peak_dict, core_dict, core_data):
    label_core_data = core_data.copy()
    for key in core_dict.keys():
        core_x = np.array(core_dict[key])[:, 0]
        core_y = np.array(core_dict[key])[:, 1]
        core_z = np.array(core_dict[key])[:, 2]
        peak_x = peak_dict[key][0]
        peak_y = peak_dict[key][1]
        peak_z = peak_dict[key][2]
        label_core_data[core_x, core_y, core_z] = key + 1
        label_core_data[peak_x, peak_y, peak_z] = key + 10
    return label_core_data


def Detect(origin_data, n_dilation, region_pixels_min, thresh, core_pixels, peak_delta, nearest_dist):
    key = 0
    core_dict_record = {}
    peak_dict_record = {}
    regions, core_data = Get_Regions(origin_data, n_dilation, region_pixels_min, thresh)
    region = regions[2]
    for i in range(len(regions)):
        region = regions[i]
        peak = Get_Peak(core_data, region)
        core_dict, peak_dict = Build_CP_Dict(peak, region, core_data)
        core_dict, peak_dict = Update_CP_Dict(peak_dict, core_dict, core_data, core_pixels, peak_delta, nearest_dist)
        for k in core_dict.keys():
            core_dict_record[key] = core_dict[k]
            peak_dict_record[key] = peak_dict[k]
            key += 1
    detect_infor_dict = Detect_Infor_Dict(peak_dict_record, core_dict_record, origin_data)
    label_core_data = Label_Core_Data(peak_dict_record, core_dict_record, core_data)
    return detect_infor_dict, label_core_data


# origin_data = fits.getdata("detect_3d.fits")
# print(origin_data.ndim)
# n_dilation = 0  # 膨胀次数
# region_pixels_min = 50  # 块的最小像素个数
# thresh = 2  # 阈值
# core_pixels = 20  # 核的大小
# peak_delta = 0  # 可以合并的核的峰值差
# nearest_dist = 10  # 可以合并的最大距离
# detect_infor_dict, label_core_data = Detect(origin_data, n_dilation, region_pixels_min, thresh, core_pixels, peak_delta,
#                                             nearest_dist)
#
# fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(8, 6))
# lable = detect_infor_dict['regions_data']
# center = detect_infor_dict['center']
# for i in range(np.int64(lable.max())):
#     t_data = np.zeros_like(origin_data)
#     core_x = np.array(np.where(lable == i + 1)[0])
#     core_y = np.array(np.where(lable == i + 1)[1])
#     #     t_data[core_x,core_y] = 1
#     minr, minc, maxr, maxc = core_x.min(), core_y.min(), core_x.max(), core_y.max()
#     rect = mpatches.Rectangle((minc - 0.5, minr - 0.5), maxc - minc + 1, maxr - minr + 1,
#                               fill=False, edgecolor='red', linewidth=1)
#     ax1.text(maxc + 0.5, minr - 0.5, '{}'.format(i + 1), color='red')
#     ax1.add_patch(rect)
#
# ax0.imshow(origin_data.sum(2))
# ax1.imshow(lable.sum(2))
# ax0.set_title('Core Image', fontsize=12, color='b')
# ax1.set_title('Label Core Image', fontsize=12, color='r')
# fig.tight_layout()
# plt.xticks([]), plt.yticks([])
# plt.show()
