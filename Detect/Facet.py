import time
import numpy as np
import matplotlib.pyplot as plt
from skimage import transform,filters,segmentation,measure,morphology
from scipy import signal,misc,ndimage
import astropy.io.fits as fits


# Facet Detect
def Get_Regions_First(origin_data, region_pixels_min=10, kopen_radius_first=3, thresh_first='mean'):
    n_find = 1
    if thresh_first == 'mean':
        thresh_first = origin_data.mean()
    elif thresh_first == 'otsu':
        thresh_first = filters.threshold_otsu(origin_data)
    else:
        thresh_first = thresh_first
    #     print(thresh)
    count = 0
    start = time.time()
    while (count < n_find):
        regions = []
        open_data = morphology.opening(origin_data > thresh_first, morphology.square(kopen_radius_first))
        cleared = open_data.copy()
        segmentation.clear_border(cleared)
        label_data = measure.label(cleared)
        borders = np.logical_xor(open_data, cleared)
        label_data[borders] = -1
        old_regions = measure.regionprops(label_data)
        for region in old_regions:
            if region.area > region_pixels_min:
                regions.append(region)
        count = len(regions)
        if count == 0:
            print('参数region_pixels_min有误！')
            break
        thresh_first += 0.1
        end = time.time()
        if end - start > 10:
            raise Exception('调参数kopen_radius,region_pixels_min！')
    core_data = np.zeros_like(origin_data)
    for i in range(len(regions)):
        label_x = regions[i].coords[:, 0]
        label_y = regions[i].coords[:, 1]
        core_data[label_x, label_y] = origin_data[label_x, label_y]
    return regions, core_data


def Convolve(core_data, axis=2):
    s = 2
    t = 17 / 5
    w = np.ones([2 * axis + 1, 2 * axis + 1])
    y = np.linspace(-axis, axis, 2 * axis + 1)
    x = np.expand_dims(y, axis=1)
    k1 = w / np.sum(w ** 2)
    k2 = w * x / np.sum((w * x) ** 2)
    k3 = w * y / np.sum((w * y) ** 2)
    k4 = w * (x ** 2 - s) / np.sum((w * (x ** 2 - s)) ** 2)
    k5 = w * x * y / np.sum((w * x * y) ** 2)
    k6 = w * (y ** 2 - s) / np.sum((y ** 2 - s) ** 2)
    k7 = w * (x ** 3 - t * x) / np.sum((x ** 3 - t * x) ** 2)
    k8 = w * (x ** 2 - s) * y / np.sum(((x ** 2 - s) * y) ** 2)
    k9 = np.array(np.matrix(k8).T)
    k10 = np.array(np.matrix(k7).T)
    gx = k2 - t * k7 - s * k9
    gy = k3 - t * k10 - s * k8
    origin_data = core_data
    # origin_data = np.around(origin_data,1)
    conv4 = signal.convolve(origin_data, k4, mode='same', method='auto')
    conv5 = signal.convolve(origin_data, k5, mode='same', method='auto')
    conv6 = signal.convolve(origin_data, k6, mode='same', method='auto')
    conv_gx = signal.convolve(origin_data, gx, mode='same', method='auto')
    conv_gy = signal.convolve(origin_data, gy, mode='same', method='auto')
    convs = [conv4, conv5, conv6, conv_gx, conv_gy]
    return convs


def Get_Lable(origin_data, convs, region, bins, keig_bins, kgra_bins):
    t_id = 0
    center_of_mass = []
    temp_eigenvalue = []
    conv4 = convs[0]
    conv5 = convs[1]
    conv6 = convs[2]
    conv_gx = convs[3]
    conv_gy = convs[4]
    coords = (region.coords[:, 0], region.coords[:, 1])
    label_data = np.zeros_like(origin_data, dtype='uint16')
    for i, j in zip(coords[0], coords[1]):
        A = np.array([[2 * conv4[i, j], conv5[i, j]], [conv5[i, j], 2 * conv6[i, j]]])
        a, b = np.linalg.eig(A)
        temp_eigenvalue.append(a)
    number_gx, gras_x = np.histogram(conv_gx, bins=bins)
    number_gy, gras_y = np.histogram(conv_gy, bins=bins)
    temp_eigenvalue = np.array(temp_eigenvalue)
    number_ex, eigs_x = np.histogram(temp_eigenvalue[:, 0], bins=bins)
    number_ey, eigs_y = np.histogram(temp_eigenvalue[:, 1], bins=bins)

    number_gx_t = number_gx[int(bins / 10):bins - int(bins / 10)]
    number_gy_t = number_gy[int(bins / 10):bins - int(bins / 10)]
    number_ex_t = number_ex[int(bins / 10):bins - int(bins / 10)]
    number_ey_t = number_ey[int(bins / 10):bins - int(bins / 10)]
    if np.where(number_ex_t == number_ex_t.max())[0][0] + keig_bins > bins - 1:
        raise Exception('减小keig_bins！')
    if np.where(number_ex_t == number_ex_t.max())[0][0] + keig_bins < -int(bins / 10):
        raise Exception('增大keig_bins！')
    if np.where(number_gx == number_gx_t.max())[0][0] + kgra_bins + 1 > bins:
        raise Exception('减小kgra_bins！')
    if np.where(number_gx == number_gx_t.max())[0][0] - kgra_bins < -int(bins / 10):
        raise Exception('增大kgra_bins！')
    eig_x = eigs_x[np.where(number_ex == number_ex_t.max())[0][0] + keig_bins]
    eig_y = eigs_y[np.where(number_ey == number_ey_t.max())[0][0] + keig_bins]
    gra_x_min = gras_x[np.where(number_gx == number_gx_t.max())[0][0] - kgra_bins]
    gra_x_max = gras_x[np.where(number_gx == number_gx_t.max())[0][0] + kgra_bins + 1]
    gra_y_min = gras_y[np.where(number_gy == number_gy_t.max())[0][0] - kgra_bins]
    gra_y_max = gras_y[np.where(number_gy == number_gy_t.max())[0][0] + kgra_bins + 1]
    for i, j in zip(coords[0], coords[1]):
        if (temp_eigenvalue[:, 0][t_id] < eig_x and temp_eigenvalue[:, 1][t_id] < eig_y) \
                and ((conv_gx[i, j] > gra_x_min and conv_gx[i, j] < gra_x_max) \
                     or (conv_gy[i, j] > gra_y_min and conv_gy[i, j] < gra_y_max)):
            label_data[i, j] = 1
        t_id += 1
    return label_data


def Recursion_Lable(origin_data, convs, label_data, final_array, bins, keig_bins, kgra_bins, label_area_max):
    erosion_label = measure.label(label_data)
    ero_regions_l = measure.regionprops(erosion_label)
    for region_l in ero_regions_l:
        temp_ero_array = np.zeros_like(erosion_label)
        if region_l.area <= label_area_max:
            temp_ero_array[np.array(region_l.coords)[:, 0], np.array(region_l.coords)[:, 1]] = 1
            final_array += temp_ero_array
        else:
            region = region_l
            label_data = Get_Lable(origin_data, convs, region, bins, keig_bins, kgra_bins)
            label_data = Recursion_Lable(origin_data, convs, label_data, final_array, bins, keig_bins, kgra_bins,
                                         label_area_max)
    return final_array


def Get_COM(origin_data, final_lable, label_area_min):
    center_of_mass = []
    final_lable = measure.label(final_lable)
    final_regions = measure.regionprops(final_lable)
    for final_region in final_regions:
        if final_region.area > label_area_min:
            x_region = final_region.coords[:, 0]
            y_region = final_region.coords[:, 1]
            od_mass = origin_data[x_region, y_region]
            center_of_mass.append(np.around((np.c_[od_mass, od_mass] * final_region.coords).sum(0) \
                                            / od_mass.sum(), 0).tolist())
    sorted_id_com = sorted(range(len(center_of_mass)), key=lambda k: center_of_mass[k], reverse=False)
    center_of_mass = (np.array(center_of_mass, dtype='uint16')[sorted_id_com]).tolist()
    return center_of_mass


def Get_Total_COM(origin_data, regions_first, convs, bins, keig_bins, kgra_bins, label_area_min, label_area_max):
    com = [[]]
    rc_dict = {}
    label_data_record = np.zeros_like(origin_data)
    erosion_label_record = np.zeros_like(origin_data)
    for i in range(len(regions_first)):
        rc_dict[i] = []
    for i in range(len(regions_first)):
        region = regions_first[i]
        label_data = Get_Lable(origin_data, convs, region, bins, keig_bins, kgra_bins)
        final_array = np.zeros_like(label_data, dtype='int16')
        final_lable = Recursion_Lable(origin_data, convs, label_data, final_array, bins, keig_bins, kgra_bins,
                                      label_area_max)
        center_of_mass = Get_COM(origin_data, final_lable, label_area_min)
        label_data_record += final_lable
        rc_dict[i] = center_of_mass
        com += center_of_mass
    com = np.array(com[1:len(com) + 1])
    return com


def Get_Regions_Second(origin_data, region_pixels_min=10, kopen_radius_second=3, thresh_second='mean'):
    n_find = 1
    if thresh_second == 'mean':
        thresh_second = origin_data.mean()
    elif thresh_second == 'otsu':
        thresh_second = filters.threshold_otsu(origin_data)
    else:
        thresh_second = thresh_second
    #     print(thresh_second)
    count = 0
    while (count < n_find):
        regions = []
        open_data = morphology.opening(origin_data > thresh_second, morphology.square(kopen_radius_second))
        cleared = open_data.copy()
        segmentation.clear_border(cleared)
        label_data = measure.label(cleared)
        #         temp_array = np.zeros_like(origin_data)
        #         temp_array[np.where(origin_data>thresh_second)] = 1
        #         label_data =measure.label(temp_array)
        old_regions = measure.regionprops(label_data)
        for region in old_regions:
            if region.area > region_pixels_min:
                regions.append(region)
        count = len(regions)
        if count == 0:
            print('参数有误！')
            break
        thresh_second += 0.1
        end = time.time()
        core_data = np.zeros_like(origin_data)
    for i in range(len(regions)):
        label_x = regions[i].coords[:, 0]
        label_y = regions[i].coords[:, 1]
        core_data[label_x, label_y] = origin_data[label_x, label_y]
    return regions, core_data


def GNC_Facet(core_data, cores_coordinate):
    xres, yres = core_data.shape
    neighbors = []
    x_center = cores_coordinate[0]
    y_center = cores_coordinate[1]
    x_arange = np.arange(max(0, x_center - 1), min(xres, x_center + 2))
    y_arange = np.arange(max(0, y_center - 1), min(yres, y_center + 2))
    [x, y] = np.meshgrid(x_arange, y_arange);
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


def Build_RC_Dict(center_of_mass, regions):
    k = 0
    i_record = []
    items = []
    rc_dict = {}
    temp_rc_dict = {}
    for i in range(len(regions)):
        rc_dict[i] = []
    for i in range(len(regions)):
        for coord in regions[i].coords:
            for cent in center_of_mass:
                if cent[0] == coord[0] and cent[1] == coord[1]:
                    i_record.append(i)
                    rc_dict[i].append(cent.tolist())
    li_record = list(set(i_record))
    for i in li_record:
        temp_rc_dict[k] = rc_dict[i]
        k += 1
    new_regions = np.array(regions)[li_record].tolist()
    rc_dict = temp_rc_dict
    for key in rc_dict.keys():
        items += rc_dict[key]
    sorted_id = sorted(range(len(items)), key=lambda k: items, reverse=False)
    new_com = np.array(items)[sorted_id]
    return new_regions, rc_dict, new_com


def Get_Peak_Facet(core_data, regions):
    prob_core_num = 0
    peak = []
    for region in regions:
        coordinates = region.coords
        for i in range(coordinates.shape[0]):
            gradients, new_center = GNC_Facet(core_data, coordinates[i])
            if gradients.max() <= 0:
                peak.append([coordinates[i][0], coordinates[i][1]])
    prob_core_num = len(peak)
    sorted_id = sorted(range(len(peak)), key=lambda k: peak[k], reverse=False)
    peak = np.array(peak)[sorted_id].tolist()
    return peak


def Build_MPR_Dict(peak, regions, core_data):
    mountain_dict = {}
    peak_dict = {}
    region_mp_dict = {}
    prob_core_num = len(peak)
    for i in range(prob_core_num):
        mountain_dict[i] = []
        peak_dict[i] = peak[i]
    for i in range(len(regions)):
        region_mp_dict[i] = []
    reg = -1
    for region in regions:
        reg += 1
        coordinates = region.coords
        for i in range(coordinates.shape[0]):
            gradients, new_center = GNC_Facet(core_data, coordinates[i])
            while gradients.max() > 0:
                gradients, new_center = GNC_Facet(core_data, new_center)
            for j in range(prob_core_num):
                if peak[j][0] == new_center[0] and peak[j][1] == new_center[1]:
                    mountain_dict[j].append([coordinates[i][0], coordinates[i][1]])
                    region_mp_dict[reg].append(j)
    for i in range(len(regions)):
        region_mp_dict[i] = list(set(region_mp_dict[i]))
    return mountain_dict, peak_dict, region_mp_dict


def Build_CDM(center_of_mass, peak_dict, rc_dict, region_mp_dict, regions):
    n = 0
    dist_dict = {}
    core_dict_num = {}
    for i in range(len(center_of_mass)):
        core_dict_num[i] = []
    for i in range(len(regions)):
        for j in range(len(rc_dict[i])):
            dist_dict[j] = []
            for key in region_mp_dict[i]:
                dist_dict[j].append(
                    ((rc_dict[i][j][0] - peak_dict[key][0]) ** 2 + (rc_dict[i][j][1] - peak_dict[key][1]) ** 2) ** (
                                1 / 2) + np.random.rand(1) / 100)
        if j == 0:
            core_dict_num[n] = region_mp_dict[i]
            n += 1
        else:
            temp_core_dict = {}
            temp_core_dict[0] = []
            dd_array = dist_dict[0]
            for j in range(1, len(rc_dict[i])):
                dd_array = np.c_[dd_array, dist_dict[j]]
                temp_core_dict[j] = []
            for k in range(len(region_mp_dict[i])):
                row_index = np.where(dd_array[k] == dd_array[k].min())[0][0]
                temp_core_dict[row_index].append(region_mp_dict[i][k])
            for j in range(len(rc_dict[i])):
                core_dict_num[n + j] = temp_core_dict[j]
                if core_dict_num[n + j] == []:
                    del core_dict_num[n + j]
            n += (j + 1)
    return core_dict_num


def Get_Core_Dict(center_of_mass, core_dict_num, mountain_dict):
    m = 0
    sort_index = []
    core_dict = {}
    core_dict_sort = {}
    for i in range(len(center_of_mass)):
        core_dict[i] = []
        core_dict_sort[i] = []
    for key in core_dict_num.keys():
        for num in core_dict_num[key]:
            core_dict[key] += mountain_dict[num]
    for i in range(len(center_of_mass)):
        for j in range(len(center_of_mass)):
            for k in range(len(core_dict[j])):
                if center_of_mass[i][0] == core_dict[j][k][0] and center_of_mass[i][1] == core_dict[j][k][1]:
                    sort_index.append(j)
    for key in sort_index:
        core_dict_sort[m] = core_dict[key]
        m += 1
    core_dict = core_dict_sort
    return core_dict


def LCD_Facet(core_dict, center_of_mass, core_data):
    label_core_data = np.zeros_like(core_data)
    for key in core_dict.keys():
        core_x = np.array(core_dict[key])[:, 0]
        core_y = np.array(core_dict[key])[:, 1]
        center_x = center_of_mass[key][0]
        center_y = center_of_mass[key][1]
        label_core_data[core_x, core_y] = key + 1
        label_core_data[center_x, center_y] = key + 10
    return label_core_data


def DID_Facet(center_of_mass, core_dict, origin_data):
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
    for key in core_dict.keys():
        k += 1
        core_x = np.array(core_dict[key])[:, 0]
        core_y = np.array(core_dict[key])[:, 1]
        peak_value.append(origin_data[center_of_mass[key][0], center_of_mass[key][1]])
        center.append(center_of_mass[key])
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


def Detect_Facet(parameters, thresh_first, thresh_second, region_pixels_min, bins, keig_bins, kgra_bins, label_area_min,
                 origin_data):
    start = time.time()
    kopen_radius_first = parameters[0]
    kopen_radius_second = parameters[1]
    axis = parameters[2]
    label_area_max = parameters[3]
    regions_first, core_data_first = Get_Regions_First(origin_data, region_pixels_min, kopen_radius_first, thresh_first)

    convs = Convolve(core_data_first, axis)
    center_of_mass = Get_Total_COM(origin_data, regions_first, convs, bins, keig_bins, kgra_bins, label_area_min,
                                   label_area_max)

    regions_second, core_data_second = Get_Regions_Second(origin_data, region_pixels_min, kopen_radius_second,
                                                          thresh_second)
    new_regions, rc_dict, new_com = Build_RC_Dict(center_of_mass, regions_second)
    peak = Get_Peak_Facet(core_data_second, new_regions)
    mountain_dict, peak_dict, region_mp_dict = Build_MPR_Dict(peak, new_regions, core_data_second)
    core_dict_num = Build_CDM(new_com, peak_dict, rc_dict, region_mp_dict, new_regions)
    core_dict = Get_Core_Dict(new_com, core_dict_num, mountain_dict)
    label_core_data = LCD_Facet(core_dict, new_com, core_data_second)
    did_Facet = DID_Facet(new_com, core_dict, origin_data)
    end = time.time()
    dtime = end - start
    return did_Facet, label_core_data, core_dict


# noise_data = fits.getdata("detect_2d.fits")
# kopen_radius_first = 2#开运算核的半径[2,3],默认值为3
# kopen_radius_second = 1#开运算核的半径[2,3],默认值为2
# axis = 1#卷积核的半径[2,3,4],默认值为2
# label_area_max = 30 #标签区域的最大面积[20,...],默认值为20 ***
# region_pixels_min = 10#块的最小像素个数
# thresh_first = 1.0#阈值['mean','otsu',number]，可输入任意值，默认'mean'
# thresh_second = 1#阈值['mean','otsu',number]，可输入任意值，默认'mean'
# bins = 11#总区间数[8,10,12,...]
# keig_bins = 1#特征值区间数[-bins/3,bins/3]，默认值为0，增大提高召回率，减小提高精确率
# kgra_bins = 2#一阶导区间数[-bins/3,bins/3]，默认值为0，增大提高召回率，减小提高精确率
# label_area_min = 4#标签区域的最小面积[1,20],默认值为2 ***
# #label_area_min：8,6,4
#
# parameters = [kopen_radius_first,kopen_radius_second,axis,label_area_max]
#
# #选择数据
# origin_data = noise_data
#
# did_Facet,lcd_Facet,cd_Facet = Detect_Facet(parameters,thresh_first,thresh_second,\
#                          region_pixels_min,bins,keig_bins,kgra_bins,label_area_min,origin_data)
#
# # print('高斯噪声均值:{}'.format(0))
# # print('高斯噪声方差:{}'.format(rms))
# # print('信号峰值:[{},{}]'.format(peak_low,peak_high-1))
# fig,(ax1,ax2)= plt.subplots(1,2, figsize=(6, 4))
# k = 0
# for key in cd_Facet.keys():
#     k += 1
#     t_data = np.zeros_like(origin_data)
#     core_x = np.array(cd_Facet[key])[:,0]
#     core_y = np.array(cd_Facet[key])[:,1]
#     t_data[core_x,core_y] = 1
#     contours = measure.find_contours(t_data,0.5)
#     minr, minc, maxr, maxc = core_x.min(),core_y.min(),core_x.max(),core_y.max()
# #     ax2.text(center_of_mass[key][1],center_of_mass[key][0],'{}'.format(k),color='red')
# #     ax2.plot(center_of_mass[key][1],center_of_mass[key][0],'r+')
#     ax2.plot(contours[0][:,1],contours[0][:,0],linewidth = 1)
# # ax0.imshow(noise_data)
# ax1.imshow(origin_data)
# ax2.imshow(lcd_Facet)
# print('Facet')
# print('dnumber = {}'.format(len(did_Facet['center'])))
# ax1.set_title('Core Image',fontsize=12,color='b')
# ax2.set_title('Label Core Image',fontsize=12,color='r')
# # ax0.set_title('Noise Image',fontsize=12,color='r')
# fig.tight_layout()
# plt.show()
