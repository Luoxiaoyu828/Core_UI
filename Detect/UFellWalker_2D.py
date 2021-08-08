import time
import numpy as np
import matplotlib.pyplot as plt
from skimage import filters, segmentation, measure, morphology
import astropy.io.fits as fits


def Get_Regions(origin_data, n_dilation, region_pixels_min, thresh='mean'):
    n_find = 1
    kopen_radius = 1
    core_data = np.zeros_like(origin_data)
    if thresh == 'mean':
        thresh = origin_data.mean()
    elif thresh == 'otsu':
        thresh = filters.threshold_otsu(origin_data)
    else:
        thresh = thresh
    #     print(thresh)
    count = 0
    start = time.time()
    while (count < n_find):
        regions = []
        #         co=morphology.closing(origin_data > thresh , morphology.square(1)) #闭运算
        open_data = morphology.opening(origin_data > thresh, morphology.square(kopen_radius))

        #         temp_array = np.zeros_like(origin_data)
        #         temp_array[np.where(origin_data>thresh)] = 1
        #         label_data =measure.label(temp_array)
        cleared = open_data.copy()
        segmentation.clear_border(cleared)
        label_data = measure.label(cleared)
        borders = np.logical_xor(open_data, cleared)  # 异或
        label_data[borders] = -1
        for i in range(n_dilation):
            label_data = morphology.dilation(label_data, morphology.disk(1))
        old_regions = measure.regionprops(label_data)

        for region in old_regions:
            #             print(region.area)
            if region.area > region_pixels_min:
                regions.append(region)
        count = len(regions)
        if count == 0:
            raise Exception('thresh,region_pixels_min参数有误或没有云核！')
        end = time.time()
        if end - start > 10:
            raise Exception('调参数n_dilation，region_pixels_min！')
    for i in range(len(regions)):
        label_x = regions[i].coords[:, 0]
        label_y = regions[i].coords[:, 1]
        core_data[label_x, label_y] = origin_data[label_x, label_y]
    return regions, core_data


def Get_New_Center(core_data, cores_coordinate):
    xres, yres = core_data.shape
    neighbors = []
    x_center = cores_coordinate[0]
    y_center = cores_coordinate[1]
    x_arange = np.arange(max(0, x_center - 1), min(xres, x_center + 2))
    y_arange = np.arange(max(0, y_center - 1), min(yres, y_center + 2))
    [x, y] = np.meshgrid(x_arange, y_arange)
    xy = np.column_stack([x.flat, y.flat])
    for i in range(xy.shape[0]):
        if 1 - all(cores_coordinate == xy[i]):
            neighbors.append([xy[i][0], xy[i][1]])
    neighbors = np.array(neighbors)
    gradients = core_data[neighbors[:, 0], neighbors[:, 1]] \
                - core_data[x_center, y_center]
    if gradients.max() > 0:
        g_step = np.where(gradients == gradients.max())[0][0]
        new_center = neighbors[g_step]
    else:
        new_center = cores_coordinate
    return gradients, new_center


def Get_Peak(core_data, region):
    peak = []
    coordinates = region.coords
    for i in range(coordinates.shape[0]):
        gradients, new_center = Get_New_Center(core_data, coordinates[i])
        if gradients.max() <= 0:
            peak.append([coordinates[i][0], coordinates[i][1]])
    sorted_id = sorted(range(len(peak)), key=lambda k: peak[k], reverse=False)
    peak = np.array(peak)[sorted_id].tolist()
    return peak


def Build_CP_Dict(peak, region, core_data):
    core_dict = {}
    prob_core_num = len(peak)
    for i in range(prob_core_num):
        core_dict[i] = []
    coordinates = region.coords
    for i in range(coordinates.shape[0]):
        gradients, new_center = Get_New_Center(core_data, coordinates[i])
        while gradients.max() >= 0:
            gradients, new_center = Get_New_Center(core_data, new_center)
        for j in range(prob_core_num):
            if peak[j][0] == new_center[0] and peak[j][1] == new_center[1]:
                core_dict[j].append([coordinates[i][0], coordinates[i][1]])
    peak_dict = {}
    for i in range(len(peak)):
        peak_dict[i] = peak[i]
    return core_dict, peak_dict


def Update_CP_Dict(peak_dict, core_dict, core_data, core_pixels=20, peak_delta=0, nearest_dist=6):
    del_core_number = []
    length = len(peak_dict)
    #     for i_num in range(length):
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
                                 + (peak_dict[j_num][1] - peak_dict[i_num][1]) ** 2) ** (1 / 2))
            temp_distance = distance.copy()
            temp_distance.remove(0)
            nearest_dis_0 = np.array(temp_distance).min()
            order = list(peak_dict.keys())[np.where(distance == nearest_dis_0)[0][0]]
            peak_delta_0 = core_data[peak_dict[order][0], peak_dict[order][1]] \
                           - core_data[peak_dict[i_num][0], peak_dict[i_num][1]]
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


def DID_FellWalker(peak_dict, core_dict, origin_data):
    peak_value = []
    center = []
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
        peak_value.append(origin_data[peak_dict[key][0], peak_dict[key][1]])
        center.append(peak_dict[key])
        regions_data[core_x, core_y] = k
        od_mass = origin_data[core_x, core_y]
        clump_com.append(np.around((np.c_[od_mass, od_mass] * core_dict[key]).sum(0) \
                                   / od_mass.sum(), 0).tolist())
        #         x_size = ((origin_data[core_x,core_y]*core_x**2).sum()/origin_data[core_x,core_y].sum()\
        #              - ((origin_data[core_x,core_y]*core_x).sum()/origin_data[core_x,core_y].sum())**2)**(1/2)
        #         y_size = ((origin_data[core_x,core_y]*core_y**2).sum()/origin_data[core_x,core_y].sum()\
        #              - ((origin_data[core_x,core_y]*core_y).sum()/origin_data[core_x,core_y].sum())**2)**(1/2)
        #         clump_size.append([x_size,y_size])
        clump_size.append([core_x.max() - core_x.min(), core_y.max() - core_y.min()])
        clump_sum.append(origin_data[core_x, core_y].sum())
        clump_volume.append(len(core_dict[key]))
        clump_regions.append([np.array(core_dict[key])[:, 0], np.array(core_dict[key])[:, 1]])

    detect_infor_dict['peak'] = np.around(peak_value, 3).tolist()
    detect_infor_dict['center'] = np.array(center).tolist()
    detect_infor_dict['clump_com'] = np.around(clump_com, 0).tolist()
    detect_infor_dict['clump_size'] = np.around(clump_size, 3).tolist()
    detect_infor_dict['clump_sum'] = np.around(clump_sum, 3).tolist()
    detect_infor_dict['clump_volume'] = clump_volume
    detect_infor_dict['regions_data'] = regions_data
    detect_infor_dict['clump_regions'] = clump_regions
    return detect_infor_dict


def Detect_FellWalker(origin_data, n_dilation, region_pixels_min, thresh, core_pixels, peak_delta, nearest_dist):
    key = 0
    core_dict_record = {}
    peak_dict_record = {}
    regions, core_data = Get_Regions(origin_data, n_dilation, region_pixels_min, thresh)
    for i in range(len(regions)):
        region = regions[i]
        peak = Get_Peak(core_data, region)
        core_dict, peak_dict = Build_CP_Dict(peak, region, core_data)
        core_dict, peak_dict = Update_CP_Dict(peak_dict, core_dict, core_data, core_pixels, peak_delta, nearest_dist)
        for k in core_dict.keys():
            core_dict_record[key] = core_dict[k]
            peak_dict_record[key] = peak_dict[k]
            key += 1
    detect_infor_dict = DID_FellWalker(peak_dict_record, core_dict_record, origin_data)
    return detect_infor_dict


# origin_data = fits.getdata("detect_2d.fits")
# n_dilation = 0  # 膨胀次数[0,1,2,3]
# region_pixels_min = 10  # 块的最小像素个数[20-]
# thresh = 2  # 阈值['mean','otsu',number]，可输入任意值
# core_pixels = 50  # 核的大小[20-]
# peak_delta = 3  # 可以合并的核的峰值差[-1,10]
# nearest_dist = 7  # 可以合并的最大距离[4,30]

#选择数据
# origin_data = noise_data
# origin_data = filters.gaussian(noise_data,sigma = 0.8)
# origin_data = noise_data
# nd_filters = filters.gaussian(noise_data,sigma = 1)
# origin_data = nd_filters

# did_FellWalker = Detect_FellWalker(origin_data, n_dilation, region_pixels_min, thresh, core_pixels, peak_delta,
#                                     nearest_dist)
# print(did_FellWalker)

# print('滤波数据！')
# print('高斯噪声均值:{}'.format(0))
# print('高斯噪声方差:{}'.format(rms))
# print('信号峰值:[{}]'.format(peak_low))#,peak_high-1
# print('高斯滤波！'.format(peak_low))
# print('使用高斯滤波平滑！')
# print('Up FellWalker')
# print('Dnumber = {}'.format(len(did_FellWalker['center'])))
# fig, (ax1,ax2) = plt.subplots(1,2, figsize=(6, 4))
# lable = did_FellWalker['regions_data']
# center = did_FellWalker['center']
# for i in range(np.int64(lable.max())):
#     t_data = np.zeros_like(origin_data)
#     core_x = np.array(np.where(lable == i)[0])
#     core_y = np.array(np.where(lable == i)[1])
#     t_data[core_x,core_y] = 1
#     contours = measure.find_contours(t_data,0.5)
#     ax2.text(center[i][1],center[i][0],'{}'.format(i+1),color='red')
#     ax2.plot(contours[0][:,1],contours[0][:,0],linewidth = 1)
# ax1.imshow(origin_data)
# ax2.imshow(lable)
# ax1.set_title('Origin Image',fontsize=12,color='b')
# ax2.set_title('Label Core Image',fontsize=12,color='r')
# fig.tight_layout()
# plt.show()
