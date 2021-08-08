import matplotlib.pyplot as plt
import numpy as np
from skimage import transform, filters
import time
from scipy.stats import multivariate_normal
import math

# n = 1
# peak_low = 6
# peak_high = 9
# angle = None
# rms = 0.23
# x_min = 0
# y_min = 0
# z_min = 0
# x_max = 100
# y_max = 100
# z_max = 100
# sigma_one = [2, 4]
# sigma_two = [2, 5]
# sigma_three = [4, 6]
# ctime = 3  # math.sqrt(8*math.log(2))
# nearest_dist = 6
# xres = x_max - x_min
# yres = y_max - y_min
# zres = z_max - z_min


def Generate_Sigma(n, sigma_one, sigma_two, sigma_three):
    sigma = [[], [], []]
    for i in range(n):
        sigma[0].append(np.random.uniform(sigma_one[0], sigma_one[1]))
        sigma[1].append(np.random.uniform(sigma_two[0], sigma_two[1]))
        sigma[2].append(np.random.uniform(sigma_three[0], sigma_three[1]))
    return sigma


def Generate_Center(n, xres, yres, zres, sigma, nearest_dist):
    x_center = []
    y_center = []
    z_center = []
    length = 0
    start = time.time()
    while length <= n:
        x_center.append(np.random.uniform(sigma.mean() * sigma[0], xres - sigma.mean() * sigma[0]))
        y_center.append(np.random.uniform(sigma.mean() * sigma[1], yres - sigma.mean() * sigma[1]))
        z_center.append(np.random.uniform(sigma.mean() * sigma[2], zres - sigma.mean() * sigma[2]))
        length = len(x_center)
        distance = []
        for j in range(length - 1):
            distance.append(((x_center[length - 1] - x_center[j]) ** 2 \
                             + (y_center[length - 1] - y_center[j]) ** 2 \
                             + (z_center[length - 1] - z_center[j]) ** 2) ** (1 / 2))
        temp_x = x_center[length - 1]
        temp_y = y_center[length - 1]
        temp_z = z_center[length - 1]
        x_center.reverse()
        y_center.reverse()
        z_center.reverse()
        if distance == []:
            continue
        elif np.array(distance).min() < nearest_dist:
            x_center.remove(temp_x)
            y_center.remove(temp_y)
            z_center.remove(temp_z)
        elif length == n + 1:
            x_center.remove(temp_x)
            y_center.remove(temp_y)
            z_center.remove(temp_z)
        end = time.time()
        if end - start > 10:
            raise Exception('请减小参数nearest_dist！')
    return x_center, y_center, z_center


def Rotate(angle, x_center, y_center, zres, data):
    for i in range(zres):
        rotate_data = transform.rotate(data[:, :, i], angle, center=[x_center, y_center])
        data[:, :, i] = rotate_data
    return data


def Get_Coords(xyz, x_center, y_center, z_center, sigma, ctime):
    cut = ctime * sigma
    logic = (xyz[:, 0] - x_center) ** 2 / (cut[0] ** 2) \
            + (xyz[:, 1] - y_center) ** 2 / (cut[1] ** 2) \
            + (xyz[:, 2] - z_center) ** 2 / (cut[2] ** 2)
    coords = xyz[logic <= 1]
    coords = (coords[:, 0], coords[:, 1], coords[:, 2])
    return coords


def Update_Prob_Density(prob_density, coords, peak):
    # 根据coords更新概率密度prob_density
    pd_data = np.zeros_like(prob_density)
    prob_density = prob_density / prob_density.max()
    pd_data[coords] = prob_density[coords]
    pd_data = pd_data * peak
    return pd_data


def Generate_3D(n, xres, yres, zres, nearest_dist, peak_low, peak_high, sigma_one, sigma_two, sigma_three, ctime, angle=None):
    peak = []
    angle_t = []
    center = []
    clump_size = []
    clump_axial_length = []
    clump_sum = []
    clump_volume = []
    regions = []
    new_regions = []
    Peak = []
    infor_dict = {}
    x, y, z = np.mgrid[0:xres:1, 0:yres:1, 0:zres:1]
    xyz = np.column_stack([x.flat, y.flat, z.flat])
    origin_data = np.zeros([xres, yres, zres])

    sigma = Generate_Sigma(n, sigma_one, sigma_two, sigma_three)
    temp_sigma = np.array([np.array(sigma_one).mean(), np.array(sigma_two).mean(), np.array(sigma_three).mean()],
                          dtype='uint8')
    x_center, y_center, z_center = Generate_Center(n, xres, yres, zres, temp_sigma, nearest_dist)
    for i in range(n):
        peak.append(np.random.uniform(peak_low, peak_high))
        if angle or angle == 0:
            angle_t.append(angle)
        else:
            angle_t.append(np.random.randint(0, 360))

        sigma_t = np.array([sigma[0][i], sigma[1][i], sigma[2][i]])
        covariance = np.diag(sigma_t ** 2)
        center.append([x_center[i], y_center[i], z_center[i]])
        prob_density = multivariate_normal.pdf(xyz, mean=center[i], cov=covariance)
        prob_density = prob_density.reshape(origin_data.shape)
        coords = Get_Coords(xyz, x_center[i], y_center[i], z_center[i], sigma_t, ctime)
        pd_data = Update_Prob_Density(prob_density, coords, peak[i])
        rotate_data = Rotate(angle_t[i], y_center[i], x_center[i], zres, pd_data)
        region = np.where(rotate_data != 0)
        origin_data += rotate_data
        Peak.append(np.where(rotate_data == rotate_data.max()))
        x_size = region[0].max() - region[0].min()
        y_size = region[1].max() - region[1].min()
        z_size = region[2].max() - region[2].min()
        clump_size.append([x_size, y_size, z_size])
        clump_axial_length.append(list(2 * math.sqrt(2 * math.log(2)) * sigma_t))
        clump_sum.append(np.array(rotate_data[region]).sum())
        clump_volume.append(region[0].shape[0])
        regions.append(region)
    sorted_id = sorted(range(len(center)), key=lambda k: center[k], reverse=False)
    infor_dict['peak'] = np.array(peak)[sorted_id].tolist()
    infor_dict['peak1'] = np.around(np.array(Peak)[sorted_id], 3).tolist()
    infor_dict['angle'] = np.array(angle_t)[sorted_id].tolist()
    infor_dict['center'] = np.array(center)[sorted_id].tolist()
    infor_dict['clump_size'] = np.around(np.array(clump_size)[sorted_id], 3).tolist()
    infor_dict['clump_axial_length'] = np.around(np.array(clump_axial_length)[sorted_id], 3).tolist()
    infor_dict['clump_sum'] = np.around(np.array(clump_sum)[sorted_id], 3).tolist()
    infor_dict['clump_volume'] = np.array(clump_volume)[sorted_id].tolist()
    for i in range(len(sorted_id)):
        new_regions.append(regions[sorted_id[i]])
    return origin_data, infor_dict, new_regions


# origin_data, infor_dict, clump_region = Generate_3D(n, xres, yres, zres, nearest_dist, peak_low, peak_high, angle)


def Gauss_Noise_3D(cov, data):
    mean = 0  # 3*cov
    noise = np.random.normal(mean, cov, (data.shape[0], data.shape[1], data.shape[2]))
    #     noise[np.where(data==0)] = 0
    data = data + noise
    return data


# noise_data = Gauss_Noise_3D(rms, origin_data)
# nd_filters = filters.gaussian(noise_data, sigma=1)
# fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(8, 6))
# ax0.imshow(origin_data.sum(2))
# ax1.imshow(noise_data.sum(2))
# ax0.set_title('Origin Image', fontsize=12, color='b')
# ax1.set_title('Gauss Noise Image', fontsize=12, color='r')
# fig.tight_layout()
# plt.xticks([]), plt.yticks([])
# plt.show()

# # 保存数据
#
# # m16_data = fits.getdata('../cloud_spice_R16/hdu0_mosaic_L_3D.fits')
# m16_hdu = fits.open('/home/share/dd/M16_detect/m16_data_with_wcs/hdu0_mosaic_L_3D.fits')
#
# # m16_simu_data = origin_data + m16_data
# data_hdu = fits.PrimaryHDU(origin_data, header=m16_hdu[0].header)
#
# m_s_fits_name = '../simulate_3Ddata_angle_random' + '/simulate_3D_{}.fits'.format(str(i).zfill(3))
# m_s_sdf_name = '../simulate_3Ddata_angle_random' + '/simulate_3D_{}.sdf'.format(str(i).zfill(3))
#
# fits.HDUList([data_hdu]).writeto(m_s_fits_name, overwrite=True)
# #         break


# 保存核表
# peak = infor_dict['peak']
# arr_peak = np.array(peak)
# angle = infor_dict['angle']
# arr_angle = np.array(angle)
# center = infor_dict['center']
# arr_center = np.array(center)
# clump_length = infor_dict['clump_length']
# arr_clump_size = np.array(clump_length)
# clump_axial_length = infor_dict['clump_axial_length']
# arr_axial_length = np.array(clump_axial_length)
# clump_sum = infor_dict['clump_sum']
# arr_clump_sum = np.array(clump_sum)
# clump_volume = infor_dict['clump_volume']
# arr_clump_volume = np.array(clump_volume)
# number = np.linspace(1, len(peak), len(peak))
# header = np.array([['number', 'peak', 'angle', 'x_center', 'y_center', 'z_center', 'x_sigma', 'y_sigma', 'z_sigma',
#                     'x_axial_len', 'y_axial_len', 'z_axial_len', 'clump_sum', 'clump_volume']])
# core_table = np.c_[
#     number, arr_peak, arr_angle, arr_center, arr_axial_length, arr_clump_size, arr_clump_sum, arr_clump_volume]
# core_table = np.r_[header, core_table]
# # np.savetxt('Core_Table.txt', core_table, delimiter=',',fmt='%s')
#
# m_s_fits_outcat = '../simulate_3Doutcat_angle_random' + '/simulate_3D_outcat_{}.txt'.format(str(i).zfill(3))
# # origin_data_outcat.imwrite(m_s_fits_outcat, overwrite=True)
# np.savetxt(m_s_fits_outcat, core_table, fmt='%s')
# #         break