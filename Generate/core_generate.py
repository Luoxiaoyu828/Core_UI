import time
from astropy.io import fits
import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from skimage import transform, filters


def Generate_Center(n, xres, yres, sigma, nearest_dis):
    # 产生n个中心位置，每个中心点间的距离大于nearest_dis
    x_center = []
    y_center = []
    length = 0
    start = time.time()
    while length <= n:
        x_center.append(
            np.random.uniform(sigma.mean() * sigma[0], xres - sigma.mean() * sigma[0]))  # sigma.mean()*sigma[0]
        y_center.append(np.random.uniform(sigma.mean() * sigma[1], yres - sigma.mean() * sigma[1]))
        length = len(x_center)
        distance = []
        for j in range(length - 1):
            distance.append(((x_center[length - 1] - x_center[j]) ** 2 \
                             + (y_center[length - 1] - y_center[j]) ** 2) ** (1 / 2))  # 计算距离
        temp_x = x_center[length - 1]
        temp_y = y_center[length - 1]
        x_center.reverse()  # 降序排列
        y_center.reverse()
        if distance == []:
            continue
        elif np.array(distance).min() < nearest_dis:
            x_center.remove(temp_x)
            y_center.remove(temp_y)
        elif length == n + 1:
            x_center.remove(temp_x)
            y_center.remove(temp_y)
        end = time.time()
        if end - start > 5:
            raise Exception('减小参数nearest_dist！')
    return x_center, y_center


def Get_Coords(xy, x_center, y_center, sigma):  # 获取坐标     xy:
    # 得到[x_center，y_center]点的sigma范围的坐标集
    cut = 3 * sigma  # 8*math.log(2)*sigma[0]**2          #截断
    logic = (xy[:, 0] - x_center) ** 2 / ((cut[0]) ** 2) + (xy[:, 1] - y_center) ** 2 / ((cut[1]) ** 2)  #
    coords = xy[logic <= 1]  # 椭圆内
    coords = (coords[:, 0], coords[:, 1])
    return coords


def Update_Prob_Density(prob_density, coords, peak):
    # 根据coords更新概率密度prob_density
    pd_data = np.zeros_like(prob_density)
    prob_density = prob_density / prob_density.max()
    pd_data[coords] = prob_density[coords]
    pd_data = pd_data * peak
    return pd_data


def Generate(n, xres, yres, nearest_dis, peak_low, peak_high, sigma, angle=None):
    # 主函数，调用前面的函数，随机生成满足条件的n个云核，并得出核表信息
    peak = []
    angle_t = []
    center = []
    clump_size = []
    clump_length = []
    clump_axial_length = []
    clump_sum = []
    clump_volume = []
    regions = []
    new_regions = []
    peak1 = []
    infor_dict = {}
    # region_dict = {}
    x, y = np.mgrid[0:xres:1, 0:yres:1]
    xy = np.column_stack([x.flat, y.flat])  #
    origin_data = np.zeros([xres, yres])

    covariance = np.diag(sigma ** 2)
    x_center, y_center = Generate_Center(n, xres, yres, sigma, nearest_dis)
    for i in range(n):
        peak.append(np.random.randint(peak_low, peak_high))
        if angle or angle == 0:
            angle_t.append(angle)
        else:
            angle_t.append(np.random.randint(0, 360))
        #         S = np.matrix([[math.cos(angle_t[i]/180*math.pi),-math.sin(angle_t[i]/180*math.pi)],\
        #                        [math.sin(angle_t[i]/180*math.pi),math.cos(angle_t[i]/180*math.pi)]])
        #         covariance = (S.T)**(-1)* np.diag(sigma**2)*(S)**(-1)
        #         print(covariance)

        center.append([x_center[i], y_center[i]])  # 质心位置
        prob_density = multivariate_normal.pdf(xy, mean=center[i], cov=covariance)
        prob_density = prob_density.reshape((xres, yres))
        coords = Get_Coords(xy, x_center[i], y_center[i], sigma)
        pd_data = Update_Prob_Density(prob_density, coords, peak[i])
        rotate_data = transform.rotate(pd_data, angle_t[i], center=[y_center[i], x_center[i]])  # 旋转
        region = np.where(rotate_data != 0)
        origin_data += rotate_data  # 旋转后的图像
        peak1.append(np.where(rotate_data == rotate_data.max()))
        #         x_size = ((rotate_data[region]*region[0]**2).sum()/rotate_data[region].sum()\
        #              - ((rotate_data[region]*region[0]).sum()/rotate_data[region].sum())**2)**(1/2)    #size计算公式
        #         y_size = ((rotate_data[region]*region[1]**2).sum()/rotate_data[region].sum()\
        #              - ((rotate_data[region]*region[1]).sum()/rotate_data[region].sum())**2)**(1/2)
        #         clump_size.append([x_size,y_size])
        clump_length.append([region[0].max() - region[0].min(), region[1].max() - region[1].min()])  # 矩形框尺寸
        clump_axial_length.append(list(sigma))  # 2*math.sqrt(2*math.log(2))
        clump_sum.append(np.array(pd_data[coords]).sum())
        clump_volume.append(region[0].shape[0])
        regions.append(region)

    sorted_id = sorted(range(len(center)), key=lambda k: center[k], reverse=False)
    infor_dict['peak'] = np.array(peak)[sorted_id].tolist()
    infor_dict['peak1'] = np.around(np.array(peak1)[sorted_id], 3).tolist()
    infor_dict['angle'] = np.array(angle_t)[sorted_id].tolist()
    infor_dict['center'] = np.array(center)[sorted_id].tolist()
    #     infor_dict['clump_size'] = np.around(np.array(clump_size)[sorted_id],3).tolist()
    infor_dict['clump_length'] = np.around(np.array(clump_length)[sorted_id], 3).tolist()
    infor_dict['clump_sigma'] = np.around(np.array(clump_axial_length)[sorted_id], 3).tolist()
    infor_dict['clump_sum'] = np.around(np.array(clump_sum)[sorted_id], 3).tolist()
    infor_dict['clump_volume'] = np.array(clump_volume)[sorted_id].tolist()
    for i in range(len(sorted_id)):
        new_regions.append(regions[sorted_id[i]])
    infor_dict['clump_regions'] = new_regions
    return origin_data, infor_dict


def Gauss_Noise(cov, data):
    # 添加方差为cov的高斯噪声
    mean = 0  # 3*cov
    noise = np.random.normal(mean, cov, (data.shape[0], data.shape[1]))
    #     noise[np.where(data==0)] = 0
    data = data + noise
    return data
