import sys
from PyQt5.QtWidgets import QMainWindow, QApplication
from Plot.plot import Ui_MainWindow
from PyQt5.QtCore import QObject
import pyqtgraph.opengl as gl
from pyqtgraph.opengl import GLViewWidget, GLAxisItem
import numpy as np
from scipy.ndimage import gaussian_filter
import pyqtgraph as pg
from astropy.io import fits


def Singleton(cls):
    _instance = {}

    def _singleton(*args, **kargs):
        if cls not in _instance:
            _instance[cls] = cls(*args, **kargs)
        return _instance[cls]

    return _singleton


class PlotWindow(QMainWindow, Ui_MainWindow):
    is_load = False

    def __init__(self):
        super().__init__()
        self.setupUi(self)
        if(not self.is_load):
            self.data1 = fits.getdata('./data_test/detect_3d.fits')
            self.data2 = fits.getdata('./data_test/s_out_000.fits')
            self.is_load = True
        self.pushButton_1.clicked.connect(lambda: self.paint(data=self.data1))
        self.pushButton_2.clicked.connect(lambda: self.paint(data=self.data2))

    def paint(self, data):
        self.plot = Plot3D(self.cloud_3d_view, data)
        self.plot.load_data()


class Plot3D(QObject):
    def __init__(self, view: GLViewWidget, data):
        self.view = view
        self.data = data
        self.draw_method_kwargs = {'sigma': 2, 'levels': 7}
        self.draw_color_method_kwargs = {'c': 0., 's': 0.05, 'a': 0.1}
        self.draw_method = self._gaussian_smoothed_isosurface
        self.draw_color_method = self._generate_hsv_color

    def load_data(self):
        self.update_plot()

    def update_plot(self):
        print(1)
        self.data = np.flip(self.data, axis=0)
        mesh_data_list = self.draw_method(self.data, **self.draw_method_kwargs)
        self.view.items.clear()
        mesh_data_list = self.draw_color_method(mesh_data_list, **self.draw_color_method_kwargs)
        meshes = [gl.GLMeshItem(meshdata=md, smooth=True, shader='balloon') for md in mesh_data_list]
        x, y, z = self.data.shape
        for m in meshes:
            m.setGLOptions('additive')
            m.translate(-x / 2, -y / 2, -z / 2)
            self.view.addItem(m)

        xgrid = gl.GLGridItem()
        ygrid = gl.GLGridItem()
        zgrid = gl.GLGridItem()
        xgrid.setSize(y, z, 1)
        ygrid.setSize(x, z, 1)
        zgrid.setSize(x, y, 1)
        xgrid.rotate(90, 0, 1, 0)
        ygrid.rotate(90, 1, 0, 0)
        xgrid.translate(-x / 2, 0, 0)
        ygrid.translate(0, -y / 2, 0)
        zgrid.translate(0, 0, -z / 2)
        self.view.addItem(xgrid)
        self.view.addItem(ygrid)
        self.view.addItem(zgrid)
        axis = GLAxisItem()
        axis.setSize(x, y, z)
        axis.translate(dx=-x / 2, dy=-y / 2, dz=-z / 2)
        self.view.addItem(axis)

    @staticmethod
    def _gaussian_smoothed_isosurface(data, sigma, levels):
        data[np.isnan(data)] = -10000
        smoothed = gaussian_filter(data, sigma)
        hist, bin_edges = np.histogram(smoothed.flatten(), bins=levels)
        level_values = bin_edges[1:]
        mds = []
        for lv in level_values:
            verts, faces = pg.isosurface(smoothed, lv)
            md = gl.MeshData(vertexes=verts, faces=faces)
            mds.append(md)
        return mds

    @staticmethod
    def _generate_hsv_color(mesh_data_list: list, c, s, a):
        for i, md in enumerate(reversed(mesh_data_list)):
            colors = np.ones((md.faceCount(), 4), dtype=float)
            hue = c + s * i
            if i < 2:
                color = pg.glColor(pg.hsvColor(hue if 0 < hue < 1.0 else 0, alpha=1))
            else:
                color = pg.glColor(pg.hsvColor(hue if 0 < hue < 1.0 else 0, alpha=a))
            colors *= np.array(color)
            md.setFaceColors(colors)
        return mesh_data_list


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = PlotWindow()
    window.show()
    app.exec_()
