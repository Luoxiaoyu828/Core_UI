from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtWidgets import QMainWindow
from child_ui.Detect import Ui_Detect
from skimage import measure, filters
import matplotlib.pyplot as plt
from Detect import UFellWalker_2D, Facet, UFellWalker_3D, Cluster_2D, Cluster_3D
import numpy as np
import matplotlib.patches as mpatches
import pandas as pd


class Detect_Window(QMainWindow, Ui_Detect):
    _signal = QtCore.pyqtSignal(dict)

    def __init__(self, origin_data, show_pic: QtWidgets.QLabel, detect_name, detect_count: QtWidgets.QLabel,
                 action: QtWidgets.QAction):
        super().__init__()
        self.setupUi(self)
        self.origin_data = origin_data
        self.show_pic = show_pic
        self.count = detect_count
        self.action = action
        self.item = None
        if detect_name == "uFellWalker":
            self.pushButton.clicked.connect(self.UFellWalker)
        elif detect_name == "Facet":
            self.pushButton.clicked.connect(self.Facet)
        elif detect_name == "Cluster":
            self.pushButton.clicked.connect(self.Cluster)
        self.action.triggered.connect(self.export)

    def UFellWalker(self):

        n_dilation = int(self.lineEdit_ndilation.text() if self.lineEdit_ndilation.text() != ''
                         else self.lineEdit_ndilation.placeholderText())  # 膨胀次数[0,1,2]
        region_pixels_min = int(self.lineEdit_pixelsmin.text() if self.lineEdit_pixelsmin.text() != ''
                                else self.lineEdit_pixelsmin.placeholderText())  # 块的最小像素个数[20-]
        thresh = int(self.lineEdit_thresh.text() if self.lineEdit_thresh.text() != ''
                     else self.lineEdit_thresh.placeholderText())  # 阈值['mean','otsu',number]，可输入任意值
        core_pixels = int(self.lineEdit_corepixels.text() if self.lineEdit_corepixels.text() != ''
                          else self.lineEdit_corepixels.placeholderText())  # 核的大小[20-]
        peak_delta = int(self.lineEdit_peakdelta.text() if self.lineEdit_peakdelta.text() != ''
                         else self.lineEdit_peakdelta.placeholderText())  # 可以合并的核的峰值差[-1,10]
        nearest_dist = int(self.lineEdit_nearestdist.text() if self.lineEdit_nearestdist.text() != ''
                           else self.lineEdit_nearestdist.placeholderText())  # 可以合并的最大距离[8,30]
        if self.origin_data.ndim == 2:
            did_FellWalker = UFellWalker_2D.Detect_FellWalker(self.origin_data, n_dilation, region_pixels_min, thresh,
                                                              core_pixels, peak_delta, nearest_dist)

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 4))
            lable = did_FellWalker['regions_data']
            center = did_FellWalker['center']
            count = len(center)
            arr = [[0] * 13 for i in range(len(center))]
            for i in range(len(center)):
                arr[i][0] = did_FellWalker['clump_com'][i][0]
                arr[i][1] = did_FellWalker['clump_com'][i][1]
                arr[i][2] = -1
                arr[i][3] = did_FellWalker['center'][i][0]
                arr[i][4] = did_FellWalker['center'][i][1]
                arr[i][5] = -1
                arr[i][6] = did_FellWalker['clump_size'][i][0]
                arr[i][7] = did_FellWalker['clump_size'][i][1]
                arr[i][8] = -1
                arr[i][9] = did_FellWalker['clump_sum'][i]
                arr[i][10] = did_FellWalker['peak'][i]
                arr[i][11] = did_FellWalker['clump_volume'][i]
                arr[i][12] = -1
            self.item = pd.DataFrame(arr, columns=["peak1", "peak2", "peak3", "center1", "center2", "center3", "size1",
                                                   "size2", "size3", "sum", "Peak", "Volume", "Shape"])
            self.count.clear()
            self.show_pic.clear()
            for i in range(np.int64(lable.max())):
                t_data = np.zeros_like(self.origin_data)
                core_x = np.array(np.where(lable == i)[0])
                core_y = np.array(np.where(lable == i)[1])
                t_data[core_x, core_y] = 1
                contours = measure.find_contours(t_data, 0.5)
                ax2.text(center[i][1], center[i][0], '{}'.format(i + 1), color='red')
                ax2.plot(contours[0][:, 1], contours[0][:, 0], linewidth=1)
            ax1.imshow(self.origin_data)
            ax2.imshow(lable)
            ax1.set_title('Origin Image', fontsize=12, color='b')
            ax2.set_title('Label Core Image', fontsize=12, color='r')
            fig.tight_layout()
            plt.savefig('./picture/003.jpg')
            self.show_pic.setPixmap(QtGui.QPixmap("./picture/003.jpg"))
            # self.count.setText(count)

            self.count.setNum(count)
            self.close()
        elif self.origin_data.ndim == 3:
            detect_infor_dict, label_core_data = UFellWalker_3D.Detect(self.origin_data, n_dilation, region_pixels_min,
                                                                       thresh, core_pixels, peak_delta, nearest_dist)

            fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(8, 6))
            lable = detect_infor_dict['regions_data']
            center = detect_infor_dict['center']
            count = len(center)
            arr = [[0] * 13 for i in range(len(center))]
            for i in range(len(center)):
                arr[i][0] = detect_infor_dict['clump_com'][i][0]
                arr[i][1] = detect_infor_dict['clump_com'][i][1]
                arr[i][2] = detect_infor_dict['clump_com'][i][2]
                arr[i][3] = detect_infor_dict['center'][i][0]
                arr[i][4] = detect_infor_dict['center'][i][1]
                arr[i][5] = detect_infor_dict['center'][i][2]
                arr[i][6] = detect_infor_dict['clump_size'][i][0]
                arr[i][7] = detect_infor_dict['clump_size'][i][1]
                arr[i][8] = detect_infor_dict['clump_size'][i][2]
                arr[i][9] = detect_infor_dict['clump_sum'][i]
                arr[i][10] = detect_infor_dict['peak'][i]
                arr[i][11] = detect_infor_dict['clump_volume'][i]
                arr[i][12] = -1
            self.item = pd.DataFrame(arr, columns=["peak1", "peak2", "peak3", "center1", "center2", "center3", "size1",
                                                   "size2", "size3", "sum", "Peak", "Volume", "Shape"])
            self.count.clear()
            self.show_pic.clear()
            for i in range(np.int64(lable.max())):
                t_data = np.zeros_like(self.origin_data)
                core_x = np.array(np.where(lable == i + 1)[0])
                core_y = np.array(np.where(lable == i + 1)[1])
                #     t_data[core_x,core_y] = 1
                minr, minc, maxr, maxc = core_x.min(), core_y.min(), core_x.max(), core_y.max()
                rect = mpatches.Rectangle((minc - 0.5, minr - 0.5), maxc - minc + 1, maxr - minr + 1,
                                          fill=False, edgecolor='red', linewidth=1)
                ax1.text(maxc + 0.5, minr - 0.5, '{}'.format(i + 1), color='red')
                ax1.add_patch(rect)

            ax0.imshow(self.origin_data.sum(2))
            ax1.imshow(lable.sum(2))
            ax0.set_title('Core Image', fontsize=12, color='b')
            ax1.set_title('Label Core Image', fontsize=12, color='r')
            fig.tight_layout()
            plt.xticks([]), plt.yticks([])
            plt.savefig('./picture/003.jpg')
            self.show_pic.setPixmap(QtGui.QPixmap("./picture/003.jpg"))
            self.count.setNum(count)
            self.close()

    def Facet(self):
        kopen_radius_first = int(self.lineEdit_kopenfirst.text() if self.lineEdit_kopenfirst.text() != ''
                                 else self.lineEdit_kopenfirst.placeholderText())  # 开运算核的半径[2,3],默认值为3
        kopen_radius_second = int(self.lineEdit_kopensecond.text() if self.lineEdit_kopensecond.text() != ''
                                  else self.lineEdit_kopensecond.placeholderText())  # 开运算核的半径[2,3],默认值为2
        axis = int(self.lineEdit_axis.text() if self.lineEdit_axis.text() != ''
                   else self.lineEdit_axis.placeholderText())  # 卷积核的半径[2,3,4],默认值为2
        label_area_max = int(self.lineEdit_areamax.text() if self.lineEdit_areamax.text() != ''
                             else self.lineEdit_areamax.placeholderText())  # 标签区域的最大面积[20,...],默认值为20 ***
        region_pixels_min = int(self.lineEdit_pixelsminF.text() if self.lineEdit_pixelsminF.text() != ''
                                else self.lineEdit_pixelsminF.placeholderText())  # 块的最小像素个数
        thresh_first = int(self.lineEdit_threshfirst.text() if self.lineEdit_threshfirst.text() != ''
                           else self.lineEdit_threshfirst.placeholderText())  # 阈值['mean','otsu',number]，可输入任意值，默认'mean'
        thresh_second = int(self.lineEdit_threshsecond.text() if self.lineEdit_threshsecond.text() != ''
                            else self.lineEdit_threshsecond.placeholderText())  # 阈值['mean','otsu',number]，可输入任意值，默认'mean'
        bins = int(self.lineEdit_bins.text() if self.lineEdit_bins.text() != ''
                   else self.lineEdit_bins.placeholderText())  # 总区间数[8,10,12,...]
        keig_bins = int(self.lineEdit_keigbins.text() if self.lineEdit_keigbins.text() != ''
                        else self.lineEdit_keigbins.placeholderText())  # 特征值区间数[-bins/3,bins/3]，默认值为0，增大提高召回率，减小提高精确率
        kgra_bins = int(self.lineEdit_kgrabins.text() if self.lineEdit_kgrabins.text() != ''
                        else self.lineEdit_kgrabins.placeholderText())  # 一阶导区间数[-bins/3,bins/3]，默认值为0，增大提高召回率，减小提高精确率
        label_area_min = int(self.lineEdit_areamin.text() if self.lineEdit_areamin.text() != ''
                             else self.lineEdit_areamin.placeholderText())  # 标签区域的最小面积[1,20],默认值为2 ***
        parameters = [kopen_radius_first, kopen_radius_second, axis, label_area_max]
        did_Facet, lcd_Facet, cd_Facet = Facet.Detect_Facet(parameters, thresh_first, thresh_second, region_pixels_min,
                                                            bins, keig_bins, kgra_bins, label_area_min, self.origin_data
                                                            )
        self.count.clear()
        self.show_pic.clear()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 4))
        center = did_Facet["center"]
        count = len(center)
        arr = [[0] * 13 for i in range(len(center))]
        for i in range(len(center)):
            arr[i][0] = did_Facet['clump_com'][i][0]
            arr[i][1] = did_Facet['clump_com'][i][1]
            arr[i][2] = -1
            arr[i][3] = did_Facet['center'][i][0]
            arr[i][4] = did_Facet['center'][i][1]
            arr[i][5] = -1
            arr[i][6] = did_Facet['clump_size'][i][0]
            arr[i][7] = did_Facet['clump_size'][i][1]
            arr[i][8] = -1
            arr[i][9] = did_Facet['clump_sum'][i]
            arr[i][10] = did_Facet['peak'][i]
            arr[i][11] = did_Facet['clump_volume'][i]
            arr[i][12] = -1
        self.item = pd.DataFrame(arr, columns=["peak1", "peak2", "peak3", "center1", "center2", "center3", "size1",
                                               "size2", "size3", "sum", "Peak", "Volume", "Shape"])
        k = 0
        for key in cd_Facet.keys():
            k += 1
            t_data = np.zeros_like(self.origin_data)
            core_x = np.array(cd_Facet[key])[:, 0]
            core_y = np.array(cd_Facet[key])[:, 1]
            t_data[core_x, core_y] = 1
            contours = measure.find_contours(t_data, 0.5)
            minr, minc, maxr, maxc = core_x.min(), core_y.min(), core_x.max(), core_y.max()
            ax2.text(center[key][1], center[key][0], '{}'.format(k), color='red')
            # ax2.plot(center_of_mass[key][1],center_of_mass[key][0],'r+')
            ax2.plot(contours[0][:, 1], contours[0][:, 0], linewidth=1)
        # ax0.imshow(noise_data)
        ax1.imshow(self.origin_data)
        ax2.imshow(lcd_Facet)
        ax1.set_title('Core Image', fontsize=12, color='b')
        ax2.set_title('Label Core Image', fontsize=12, color='r')
        # ax0.set_title('Noise Image',fontsize=12,color='r')
        fig.tight_layout()
        plt.savefig('./picture/003.jpg')
        self.show_pic.setPixmap(QtGui.QPixmap("./picture/003.jpg"))
        self.count.setNum(count)
        self.close()

    def Cluster(self):
        gradmin = float(self.lineEdit_gradmin.text() if self.lineEdit_gradmin.text() != ''
                        else self.lineEdit_gradmin.placeholderText())
        rhomin = float(self.lineEdit_rhomin.text() if self.lineEdit_rhomin.text() != ''
                     else self.lineEdit_rhomin.placeholderText())
        deltamin = int(self.lineEdit_deltamin.text() if self.lineEdit_deltamin.text() != ''
                       else self.lineEdit_deltamin.placeholderText())
        v_min = int(self.lineEdit_volmin.text() if self.lineEdit_volmin.text() != ''
                    else self.lineEdit_volmin.placeholderText())
        rms = float(self.lineEdit_rms.text() if self.lineEdit_rms.text() != ''
                  else self.lineEdit_rms.placeholderText())
        sigma = float(self.lineEdit_sigma.text() if self.lineEdit_sigma.text() != ''
                    else self.lineEdit_sigma.placeholderText())
        if self.origin_data.ndim == 2:
            is_plot = 0
            outcat, out, mask, centInd = Cluster_2D.localDenCluster(self.origin_data, is_plot, gradmin, rhomin, deltamin,
                                                                    v_min, rms, sigma)
            count = outcat.shape[0]
            arr = [[0] * 13 for i in range(10)]
            for i in range(count):
                arr[i][0] = outcat[i][0]
                arr[i][1] = outcat[i][1]
                arr[i][2] = -1
                arr[i][3] = outcat[i][2]
                arr[i][4] = outcat[i][3]
                arr[i][5] = -1
                arr[i][6] = outcat[i][4]
                arr[i][7] = outcat[i][5]
                arr[i][8] = -1
                arr[i][9] = outcat[i][6]
                arr[i][10] = outcat[i][7]
                arr[i][11] = outcat[i][8]
                arr[i][12] = -1
            self.item = pd.DataFrame(arr, columns=["peak1", "peak2", "peak3", "center1", "center2", "center3", "size1",
                                                   "size2", "size3", "sum", "Peak", "Volume", "Shape"])
            self.count.clear()
            self.show_pic.clear()
            fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(8, 6))
            ax0.imshow(self.origin_data, 'gray')
            ax1.imshow(out)
            ax2.imshow(mask)
            ax0.set_title('Origin Image', fontsize=12, color='b')
            ax1.set_title('Core Image', fontsize=12, color='r')
            ax2.set_title('Label Core Image', fontsize=12, color='r')
            fig.tight_layout()
            plt.xticks([]), plt.yticks([])
            plt.savefig('./picture/003.jpg')
            self.show_pic.setPixmap(QtGui.QPixmap("./picture/003.jpg"))
            self.count.setNum(count)
            self.close()
        elif self.origin_data.ndim == 3:
            origin_data = filters.gaussian(self.origin_data, sigma)
            [xres, yres, zres] = origin_data.shape
            x = np.arange(xres)
            y = np.arange(yres)
            z = np.arange(zres)
            yy, xx, zz = np.meshgrid(y, x, z)
            xyz = np.column_stack([xx.flat, yy.flat, zz.flat])
            Distance, Gradient, IndNearNeigh = Cluster_3D.Build_DGI(origin_data, rms, xyz)
            clust_order, centInd, centNum = Cluster_3D.Get_Infor(origin_data, Distance, Gradient, IndNearNeigh, rhomin,
                                                                 v_min, deltamin)
            co_real, mask, output = Cluster_3D.Get_Mask_Out(origin_data, xyz, clust_order, Gradient, gradmin, centNum)
            describe = Cluster_3D.Core_Describe(origin_data, xyz, co_real, centInd, centNum)
            count = centNum.shape[0]
            arr = [[0] * 13 for i in range(count)]
            for i in range(count):
                arr[i][0] = describe['clumpPeakIndex'][i][0]
                arr[i][1] = describe['clumpPeakIndex'][i][1]
                arr[i][2] = describe['clumpPeakIndex'][i][2]
                arr[i][3] = describe['clumpCenter'][i][0]
                arr[i][4] = describe['clumpCenter'][i][1]
                arr[i][5] = describe['clumpCenter'][i][2]
                arr[i][6] = describe['clustSize'][i][0]
                arr[i][7] = describe['clustSize'][i][1]
                arr[i][8] = describe['clustSize'][i][2]
                arr[i][9] = describe['clustSum'][i]
                arr[i][10] = describe['clustPeak'][i]
                arr[i][11] = describe['clustVolume'][i]
                arr[i][12] = -1
            self.item = pd.DataFrame(arr, columns=["peak1", "peak2", "peak3", "center1", "center2", "center3", "size1",
                                                   "size2", "size3", "sum", "Peak", "Volume", "Shape"])
            self.count.clear()
            self.show_pic.clear()
            fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(8, 6))
            for i in range(centNum.shape[0]):
                core_x = np.where(mask == i + 1)[0]
                core_y = np.where(mask == i + 1)[1]
                minr, minc, maxr, maxc = core_x.min(), core_y.min(), core_x.max(), core_y.max()
                rect = mpatches.Rectangle((minc - 0.5, minr - 0.5), maxc - minc + 1, maxr - minr + 1,
                                          fill=False, edgecolor='red', linewidth=1)
                ax2.text(maxc + 0.5, minr - 0.5, '{}'.format(i + 1), color='red')
                ax2.add_patch(rect)

            ax0.imshow(origin_data.sum(2), 'gray')
            ax1.imshow(output.sum(2))
            ax2.imshow(mask.sum(2))
            ax0.set_title('Origin Image', fontsize=12, color='b')
            ax1.set_title('Core Image', fontsize=12, color='r')
            ax2.set_title('Label Core Image', fontsize=12, color='r')
            fig.tight_layout()
            plt.xticks([]), plt.yticks([])
            plt.savefig('./picture/003.jpg')
            self.show_pic.setPixmap(QtGui.QPixmap("./picture/003.jpg"))
            self.count.setNum(count)
            self.close()

    def export(self):
        try:
            filter_ = "EXCEL(*.xlsx)"
            fileName, _suffix = QtWidgets.QFileDialog.getSaveFileName(filter=filter_)
            self.item.to_excel(fileName)
        except Exception as e:
            print(e)
