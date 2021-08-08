from PyQt5 import QtCore
from PyQt5.QtWidgets import QMainWindow
from child_ui.second_window import Ui_ChildWindow


class childWindow(QMainWindow, Ui_ChildWindow):
    _signal = QtCore.pyqtSignal(dict)

    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.para = {
            "n": None, "peak_low": None, "peak_high": None, "angle": None, "rms": None,
            "x_min": None, "y_min": None, "x_max": None, "y_max": None, "a": None, "b": None,
            "nearest_dis": None, "n1": None, "peak_low1": None, "peak_high1": None,
            "angle1": None, "rms1": None, "x_min1": None, "y_min1": None, "z_min": None,
            "x_max1": None, "y_max1": None, "z_max": None, "sigma1": None, "sigma2": None,
            "sigma3": None, "ctime": None, "nearest_dis1": None, "n2": None, "p": None,
            "peak": None, "sigma": None, "IMG_x": None, "IMG_y": None, "IMG_z": None
        }
        self.Submit.clicked.connect(self.getPara)

    def getPara(self):
        # 二维参数设置
        self.para["n"] = self.lineEdit_n.text() if self.lineEdit_n.text() != '' else self.lineEdit_n.placeholderText()
        self.para["peak_low"] = self.lineEdit_peak_low.text() if self.lineEdit_peak_low.text() != '' \
            else self.lineEdit_peak_low.placeholderText()
        self.para["peak_high"] = self.lineEdit_peak_high.text() if self.lineEdit_peak_high.text() != '' \
            else self.lineEdit_peak_high.placeholderText()
        self.para["angle"] = self.lineEdit_angle.text() if self.lineEdit_angle.text() != '' \
            else self.lineEdit_angle.placeholderText()
        self.para["rms"] = self.lineEdit_rms.text() if self.lineEdit_rms.text() != '' \
            else self.lineEdit_rms.placeholderText()
        self.para["x_min"] = self.lineEdit_xmin.text() if self.lineEdit_xmin.text() != '' \
            else self.lineEdit_xmin.placeholderText()
        self.para["y_min"] = self.lineEdit_ymin.text() if self.lineEdit_ymin.text() != '' \
            else self.lineEdit_ymin.placeholderText()
        self.para["x_max"] = self.lineEdit_xmax.text() if self.lineEdit_xmax.text() != '' \
            else self.lineEdit_xmax.placeholderText()
        self.para["y_max"] = self.lineEdit_ymax.text() if self.lineEdit_ymax.text() != '' \
            else self.lineEdit_ymax.placeholderText()
        self.para["nearest_dis"] = self.lineEdit_nearest_dis.text() if self.lineEdit_nearest_dis.text() != '' \
            else self.lineEdit_nearest_dis.placeholderText()
        self.para["a"] = self.lineEdit_a.text() if self.lineEdit_a.text() != '' \
            else self.lineEdit_a.placeholderText()
        self.para["b"] = self.lineEdit_b.text() if self.lineEdit_b.text() != '' \
            else self.lineEdit_b.placeholderText()

        # 三维参数设置
        self.para["n1"] = self.lineEdit_n_3d.text() if self.lineEdit_n_3d.text() != '' \
            else self.lineEdit_n_3d.placeholderText()
        self.para["peak_low1"] = self.lineEdit_peak_low_3d.text() if self.lineEdit_peak_low_3d.text() != '' \
            else self.lineEdit_peak_low_3d.placeholderText()
        self.para["peak_high1"] = self.lineEdit_peak_high_3d.text() if self.lineEdit_peak_high_3d.text() != '' \
            else self.lineEdit_peak_high_3d.placeholderText()
        self.para["angle1"] = self.lineEdit_angle_3d.text() if self.lineEdit_angle_3d.text() != '' \
            else self.lineEdit_angle_3d.placeholderText()
        self.para["rms1"] = self.lineEdit_rms_3d.text() if self.lineEdit_rms_3d.text() != '' \
            else self.lineEdit_rms_3d.placeholderText()
        self.para["x_min1"] = self.lineEdit_xmin_3d.text() if self.lineEdit_xmin_3d.text() != '' \
            else self.lineEdit_xmin_3d.placeholderText()
        self.para["y_min1"] = self.lineEdit_ymin_3d.text() if self.lineEdit_ymin_3d.text() != '' \
            else self.lineEdit_ymin_3d.placeholderText()
        self.para["z_min"] = self.lineEdit_zmin_3d.text() if self.lineEdit_zmin_3d.text() != '' \
            else self.lineEdit_zmin_3d.placeholderText()
        self.para["x_max1"] = self.lineEdit_xmax_3d.text() if self.lineEdit_xmax_3d.text() != '' \
            else self.lineEdit_xmax_3d.placeholderText()
        self.para["y_max1"] = self.lineEdit_ymax_3d.text() if self.lineEdit_ymax_3d.text() != '' \
            else self.lineEdit_ymax_3d.placeholderText()
        self.para["z_max"] = self.lineEdit_zmax_3d.text() if self.lineEdit_zmax_3d.text() != '' \
            else self.lineEdit_zmax_3d.placeholderText()
        self.para["sigma1"] = self.lineEdit_sigma1.text() if self.lineEdit_sigma1.text() != '' \
            else self.lineEdit_sigma1.placeholderText()
        self.para["sigma2"] = self.lineEdit_sigma2.text() if self.lineEdit_sigma2.text() != '' \
            else self.lineEdit_sigma2.placeholderText()
        self.para["sigma3"] = self.lineEdit_sigma3.text() if self.lineEdit_sigma3.text() != '' \
            else self.lineEdit_sigma3.placeholderText()
        self.para["ctime"] = self.lineEdit_ctime_3d.text() if self.lineEdit_ctime_3d.text() != '' \
            else self.lineEdit_ctime_3d.placeholderText()
        self.para["nearest_dis1"] = self.lineEdit_nearest_dis_3d.text() if self.lineEdit_nearest_dis_3d.text() != '' \
            else self.lineEdit_nearest_dis_3d.placeholderText()

        # the second of generate
        self.para["n2"] = self.lineEdit_n2.text() if self.lineEdit_n2.text() != '' \
            else self.lineEdit_n2.placeholderText()
        self.para["p"] = self.lineEdit_p.text() if self.lineEdit_p.text() != '' \
            else self.lineEdit_p.placeholderText()
        self.para["sigma"] = self.lineEdit_sigma.text() if self.lineEdit_sigma.text() != '' \
            else self.lineEdit_sigma.placeholderText()
        self.para["peak"] = self.lineEdit_peak.text() if self.lineEdit_peak.text() != '' \
            else self.lineEdit_peak.placeholderText()
        self.para["IMG_x"] = self.lineEdit_IMGx.text() if self.lineEdit_IMGx.text() != '' \
            else self.lineEdit_IMGx.placeholderText()
        self.para["IMG_y"] = self.lineEdit_IMGy.text() if self.lineEdit_IMGy.text() != '' \
            else self.lineEdit_IMGy.placeholderText()
        self.para["IMG_z"] = self.lineEdit_IMGz.text() if self.lineEdit_IMGz.text() != '' \
            else self.lineEdit_IMGz.placeholderText()
        self._signal.emit(self.para)  # 发送信号
        self.close()
