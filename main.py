import sys
from UI.molecularcloud import Ui_mainwindow
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5 import QtWidgets, QtGui, QtCore
import pandas as pd
from Generate.core_generate import *
from Generate.core_generate_3d import *
from Generate.core_generate_other import *
from setparawindow import childWindow
from second.main import MainWindow_Sec
from detect import Detect_Window
import astropy.io.fits as fits
import os
import pymysql
import pathlib


class MainWindow(QMainWindow, Ui_mainwindow):
    signal = QtCore.pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setupUi(self)
        # self.df_info_table = None
        self.list = None
        self.item = None
        self.origin_data = None
        self.noise_data = None
        self.show_pic.setScaledContents(True)
        self.childDialog = childWindow()  # 子窗口实例化
        self.MainWindow_Sec = None
        self.isOpen = False
        self.count = 0
        self.DetectDialog = None
        self.actionSet_Parameter.triggered.connect(lambda: self.onClicked(para="model_para"))
        self.action2D.triggered.connect(self.generate_gaussian_2d)
        self.action3D.triggered.connect(self.generate_gaussian_3d)
        self.actionGenerate_2.triggered.connect(self.generate_power_law)
        self.actionExport_excel.triggered.connect(lambda: self.export_data(para="excel"))
        self.actionExport_fits.triggered.connect(lambda: self.export_data(para="fits"))
        self.actionUFellWalker.triggered.connect(lambda: self.onClicked(para="uFellWalker"))
        self.actionFacet.triggered.connect(lambda: self.onClicked(para="Facet"))
        self.actionCluster.triggered.connect(lambda: self.onClicked(para="Cluster"))
        self.actionOpen_data.triggered.connect(self.read_file)
        self.Connect_btn.clicked.connect(self.Connect_SecWindow)
        self.connect_database.triggered.connect(self.connectDatabase)
        for i in os.listdir('./temp_fits'):
            c_path = os.path.join('./temp_fits', i)
            os.remove(c_path)

    def Connect_SecWindow(self):
        if not self.isOpen:
            if self.noise_data is not None:
                fits.writeto('./temp_fits/s_out_{}.fits'.format(self.count), self.noise_data)
                self.count += 1
            self.MainWindow_Sec = MainWindow_Sec()
            self.MainWindow_Sec.show()
            self.isOpen = True
        else:
            fits.writeto('./temp_fits/s_out_{}.fits'.format(self.count), self.noise_data)
            self.count += 1

    def onClicked(self, para: str):
        if para == "model_para":
            self.label_count.clear()
            self.childDialog.show()  # 打开子窗口
            self.childDialog._signal.connect(self.getData)  # 接受信号
        elif para == "uFellWalker":
            self.DetectDialog = Detect_Window(origin_data=self.noise_data, show_pic=self.show_pic,
                                              detect_name="uFellWalker", detect_count=self.label_count,
                                              action=self.detect_Export)
            # 若窗口出现闪退问题，则表示这句代码有问题，子窗口的初始化还是要放到__init__中
            self.DetectDialog.show()
        elif para == "Facet":
            self.DetectDialog = Detect_Window(origin_data=self.noise_data, show_pic=self.show_pic,
                                              detect_name="Facet", detect_count=self.label_count,
                                              action=self.detect_Export)
            self.DetectDialog.show()
        elif para == "Cluster":
            self.DetectDialog = Detect_Window(origin_data=self.noise_data, show_pic=self.show_pic,
                                              detect_name="Cluster", detect_count=self.label_count,
                                              action=self.detect_Export)
            self.DetectDialog.show()

    def getData(self, para):
        self.list = para

    def generate_gaussian_2d(self):
        n = int(self.list["n"])  # 分子云核个数
        arr = [[0] * 13 for i in range(n)]
        peak_low = int(self.list["peak_low"])  # 分子云核峰值的最小值
        peak_high = int(self.list["peak_high"])  # 分子云核峰值的最大值
        angle = float(self.list["angle"])  # 分子云核的旋转角度，如果为None，则角度为随机数
        rms = float(self.list["rms"])  # 高斯噪声的方差可能为小数
        x_min = int(self.list["x_min"])  # 二维边界大小
        y_min = int(self.list["y_min"])  # 二维边界大小
        x_max = int(self.list["x_max"])  # 二维边界大小
        y_max = int(self.list["y_max"])  # 二维边界大小
        a = np.random.uniform(int(self.list["a"].split(",")[0]), int(self.list["a"].split(",")[1]))
        b = np.random.uniform(int(self.list["b"].split(",")[0]), int(self.list["b"].split(",")[1]))
        sigma = np.array([a, b])  # 分子云核大小的协方差矩阵
        nearest_dis = int(self.list["nearest_dis"])  # 分子云核质心点的最小欧式距离
        xres = x_max - x_min
        yres = y_max - y_min
        self.origin_data, infor_dict = Generate(n, xres, yres, nearest_dis, peak_low, peak_high, sigma,
                                                angle)
        for i in range(0, n):
            arr[i][0] = infor_dict["peak1"][i][0][0]
            arr[i][1] = infor_dict["peak1"][i][1][0]
            arr[i][2] = -1
            arr[i][3] = infor_dict["center"][i][0]
            arr[i][4] = infor_dict["center"][i][1]
            arr[i][5] = -1
            arr[i][6] = infor_dict["clump_sigma"][i][0]
            arr[i][7] = infor_dict["clump_sigma"][i][1]
            arr[i][8] = -1
            arr[i][9] = infor_dict["clump_sum"][i]
            arr[i][10] = infor_dict["peak"][i]
            arr[i][11] = infor_dict["clump_volume"][i]
            arr[i][12] = -1
        self.item = pd.DataFrame(arr, columns=["peak1", "peak2", "peak3", "center1", "center2", "center3", "size1",
                                               "size2", "size3", "sum", "Peak", "Volume", "Shape"])
        self.noise_data = Gauss_Noise(rms, self.origin_data)
        self.label_count.clear()
        self.show_pic.clear()
        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(8, 6))
        ax0.imshow(self.origin_data)
        ax1.imshow(self.noise_data)
        ax0.set_title('Origin Image', fontsize=12, color='b')
        ax1.set_title('Gauss Noise Image', fontsize=12, color='r')
        fig.tight_layout()
        plt.xticks([]), plt.yticks([])
        plt.savefig("./picture/001.jpg")  # TODO:need to be improved
        self.show_pic.setPixmap(QtGui.QPixmap("./picture/001.jpg"))  # TODO:need to be improved

    def generate_gaussian_3d(self):
        n = int(self.list["n1"])
        arr = [[0] * 13 for j in range(n)]
        peak_low = int(self.list["peak_low1"])
        peak_high = int(self.list["peak_high1"])
        angle = float(self.list["angle1"])
        rms = float(self.list["rms1"])
        x_min = int(self.list["x_min1"])
        y_min = int(self.list["y_min1"])
        z_min = int(self.list["z_min"])
        x_max = int(self.list["x_max1"])
        y_max = int(self.list["y_max1"])
        z_max = int(self.list["z_max"])
        sigma_one = [int(self.list["sigma1"].split(",")[0]), int(self.list["sigma1"].split(",")[1])]
        sigma_two = [int(self.list["sigma2"].split(",")[0]), int(self.list["sigma2"].split(",")[1])]
        sigma_three = [int(self.list["sigma3"].split(",")[0]), int(self.list["sigma3"].split(",")[1])]
        ctime = int(self.list["ctime"])  # math.sqrt(8*math.log(2))
        nearest_dist = int(self.list["nearest_dis1"])
        xres = x_max - x_min
        yres = y_max - y_min
        zres = z_max - z_min
        self.origin_data, infor_dict, clump_region = Generate_3D(n, xres, yres, zres, nearest_dist, peak_low, peak_high,
                                                                 sigma_one, sigma_two, sigma_three, ctime, angle)
        for i in range(0, n):
            arr[i][0] = infor_dict["peak1"][i][0][0]
            arr[i][1] = infor_dict["peak1"][i][1][0]
            arr[i][2] = infor_dict["peak1"][i][2][0]
            arr[i][3] = infor_dict["center"][i][0]
            arr[i][4] = infor_dict["center"][i][1]
            arr[i][5] = infor_dict["center"][i][2]
            arr[i][6] = infor_dict["clump_axial_length"][i][0]
            arr[i][7] = infor_dict["clump_axial_length"][i][1]
            arr[i][8] = infor_dict["clump_axial_length"][i][2]
            arr[i][9] = infor_dict["clump_sum"][i]
            arr[i][10] = infor_dict["peak"][i]
            arr[i][11] = infor_dict["clump_volume"][i]
            arr[i][12] = -1
        self.item = pd.DataFrame(arr, columns=["peak1", "peak2", "peak3", "center1", "center2", "center3", "size1",
                                               "size2", "size3", "sum", "Peak", "Volume", "Shape"])
        self.noise_data = Gauss_Noise_3D(rms, self.origin_data)
        self.label_count.clear()
        self.show_pic.clear()
        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(8, 6))
        ax0.imshow(self.origin_data.sum(2))
        ax1.imshow(self.noise_data.sum(2))
        ax0.set_title('Origin Image', fontsize=12, color='b')
        ax1.set_title('Gauss Noise Image', fontsize=12, color='r')
        fig.tight_layout()
        plt.xticks([]), plt.yticks([])
        plt.savefig("./picture/002.jpg")
        self.show_pic.setPixmap(QtGui.QPixmap("./picture/002.jpg"))

    def generate_power_law(self):
        n_clumps = int(self.list["n2"])
        sigma = [float(self.list["sigma"].split(",")[0]), float(self.list["sigma"].split(",")[1])]
        A = [float(self.list["peak"].split(",")[0]), float(self.list["peak"].split(",")[1])]
        IMG_x = int(self.list["IMG_x"])
        IMG_y = int(self.list["IMG_y"])
        IMG_z = int(self.list["IMG_z"])
        p = int(self.list["p"])
        self.origin_data, info_dict = make_clumps(n_clumps, sigma, A, IMG_x, IMG_y, IMG_z, p)
        self.noise_data = Gauss_Noise(0, self.origin_data)
        self.item = pd.DataFrame(info_dict,
                                 columns=["peak1", "peak2", "peak3", "center1", "center2", "center3", "size1",
                                          "size2", "size3", "sum", "Peak", "Volume", "shape"])
        self.label_count.clear()
        self.show_pic.clear()
        for i in range(3):
            plt.subplot(1, 3, i + 1)
            plt.imshow(self.origin_data.sum(i))
        plt.savefig("./picture/004.jpg")
        self.show_pic.setPixmap(QtGui.QPixmap('./picture/004.jpg'))

    def export_data(self, para: str):
        try:
            if para == "excel":
                filter_ = "EXCEL(*.xlsx)"
                fileName, _suffix = QtWidgets.QFileDialog.getSaveFileName(filter=filter_)
                self.item.to_excel(fileName)
            elif para == "fits":
                filter_ = "FITS(*.fits)"
                fileName, _suffix = QtWidgets.QFileDialog.getSaveFileName(filter=filter_)
                fits.writeto(fileName, self.origin_data)
        except Exception as e:
            print(e)

    def read_file(self):
        fits_files = []
        try:
            dlg = QtWidgets.QFileDialog()
            filter_ = "FITS(*.fits)"
            dlg.setFileMode(QtWidgets.QFileDialog.ExistingFiles)
            filenames, *_ = dlg.getOpenFileNames(filter=filter_)
            self.noise_data = fits.getdata(filenames[0])
        except Exception as e:
            print(e)

    @staticmethod
    def connectDatabase():
        # mysql -hlocalhost -uroot -p
        db = pymysql.connect(host='localhost', user='root', passwd='lyyzgr211314', database='User')
        cursor = db.cursor()
        filePath, *_type = QtWidgets.QFileDialog.getOpenFileName(filter='*.xlsx')
        book = pd.read_excel(filePath, engine="openpyxl")
        query = "insert into Core(peak1, peak2, peak3, center1, center2, center3, size1, size2, size3, sum, Peak, Volume," \
                "Shape) values(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
        for i in range(0, book.shape[0]):
            peak1 = book.loc[i, 'peak1']
            peak2 = book.loc[i, 'peak2']
            peak3 = book.loc[i, 'peak3']
            center1 = book.loc[i, 'center1']
            center2 = book.loc[i, 'center2']
            center3 = book.loc[i, 'center3']
            size1 = book.loc[i, 'size1']
            size2 = book.loc[i, 'size2']
            size3 = book.loc[i, 'size3']
            sum = book.loc[i, 'sum']
            Peak = book.loc[i, 'Peak']
            Volume = book.loc[i, 'Volume']
            Shape = book.loc[i, 'Shape']
            values = (peak1, peak2, peak3, center1, center2, center3, size1, size2, size3, sum, Peak, Volume, Shape)
            cursor.execute(query, values)
        cursor.close()
        db.commit()
        db.close()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.exec_()

# data = fits.getdata('./data_test/s_out_000.fits')
