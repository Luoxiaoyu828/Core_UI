from PyQt5 import QtWidgets
from PyQt5.QtCore import QObject
import pyqtgraph as pg
from pyqtgraph.opengl import GLViewWidget, GLVolumeItem, GLAxisItem, GLGridItem
from disface_py.data_loader import DataLoader
import pyqtgraph.opengl as gl
import numpy as np
from scipy.ndimage import gaussian_filter


class SlicePlot(QObject):
    def __init__(self,
                 axis: int,
                 view: pg.ImageView,
                 spin_box: QtWidgets.QSpinBox,
                 scroll_bar: QtWidgets.QScrollBar):
        self.view = view
        self.axis = axis
        self.spin_box = spin_box
        self.scroll_bar = scroll_bar
        self.idx_range = 100
        self._init_widgets()

        self.data = None

    def load_data(self, data: DataLoader):
        self.data = data.current_data.copy()
        self.idx_range = self.data.shape[self.axis]
        self.spin_box.setRange(1, self.idx_range)
        self.scroll_bar.setRange(1, self.idx_range)
        self.update_plot(self.data.shape[self.axis] // 2)

    def _init_widgets(self):
        self.spin_box.setRange(1, self.idx_range)
        self.scroll_bar.setRange(1, self.idx_range)
        self.spin_box.valueChanged.connect(lambda: self.scroll_bar.setValue(self.spin_box.value()))
        self.scroll_bar.valueChanged.connect(lambda: self.spin_box.setValue(self.scroll_bar.value()))
        self.spin_box.valueChanged.connect(lambda: self.update_plot(self.spin_box.value()))

        # self.view.ui.histogram.hide()
        # self.view.ui.roiBtn.hide()
        # self.view.ui.menuBtn.hide()

    def update_plot(self, idx):
        idx -= 1
        self.current_slice = SlicePlot.get_slice(self.data, self.axis, idx)
        self.current_slice = np.flip(self.current_slice, axis=0).T
        self.view.setImage(self.current_slice)

    @staticmethod
    def get_slice(data_cube, axis, idx):
        # transpose the order of axis, prepare for making slices
        tr = [axis] + list(range(data_cube.ndim))
        del tr[axis + 1]
        slice = np.transpose(data_cube, tr)[idx, :, :]

        return slice

    # def rotate_3D(self):
    #     import time
    #     view = self.graphicsView
    #     count = 0
    #     while True:
    #         app.processEvents()
    #         view.items[1].rotate(5, 0, 1, 0)
    #         time.sleep(0.001)
    #         count += 1

    # def my_print(self):
    #     import pyqtgraph.opengl as gl
    #     view = self.graphicsView
    #     xgrid = gl.GLGridItem()
    #     ygrid = gl.GLGridItem()
    #     zgrid = gl.GLGridItem()
    #     view.addItem(xgrid)
    #     view.addItem(ygrid)
    #     view.addItem(zgrid)

    #     ## rotate x and y grids to face the correct direction
    #     xgrid.rotate(90, 0, 1, 0)
    #     ygrid.rotate(90, 1, 0, 0)

    #     ## scale each grid differently
    #     xgrid.scale(0.2, 0.1, 0.1)
    #     ygrid.scale(0.2, 0.1, 0.1)
    #     zgrid.scale(0.1, 0.2, 0.1)

    # def test(self):
    #     import pyqtgraph as pg
    #     import pyqtgraph.opengl as gl

    #     view = self.graphicsView
    #     # view.setFixedSize(1000, 1000)
    #     xgrid = gl.GLGridItem()
    #     ygrid = gl.GLGridItem()
    #     zgrid = gl.GLGridItem()
    #     view.addItem(xgrid)
    #     view.addItem(ygrid)
    #     view.addItem(zgrid)
    #     xgrid.rotate(90, 0, 1, 0)
    #     ygrid.rotate(90, 1, 0, 0)
    #     xgrid.scale(0.2, 0.1, 0.1)
    #     ygrid.scale(0.2, 0.1, 0.1)
    #     zgrid.scale(0.1, 0.2, 0.1)

    # def slot_print(self):
    #     buttonReply = QtWidgets.QMessageBox.question(self,
    #                                                  'PyQt5 message',
    #                                                  "Do you like PyQt5?",
    #                                                  QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
    #                                                  QtWidgets.QMessageBox.No)
    #     if buttonReply == QtWidgets.QMessageBox.Yes:
    #         print('Yes clicked.')
    #     else:
    #         print('No clicked.')

    # def save(self):
    #     s_reply = QtWidgets.QFileDialog.getOpenFileName()


class IntegralPlot(QObject):
    def __init__(self,
                 axis: int,
                 view: pg.ImageView,
                 spin_box_left: QtWidgets.QSpinBox,
                 spin_box_right: QtWidgets.QSpinBox):

        self.view = view
        self.axis = axis
        self.spin_box_left = spin_box_left
        self.spin_box_right = spin_box_right
        self.left_range = (0, 49)
        self.right_range = (50, 100)
        self.int_range = [0, 100]
        self._init_widgets()

        self.data = None

    def load_data(self, data: DataLoader):
        self.data = data.current_data.copy()
        self.int_range = [1, self.data.shape[self.axis]]
        self.left_range = (1, self.int_range[1] - 1)
        self.right_range = (2, self.int_range[1])
        self.spin_box_left.setRange(*self.left_range)
        self.spin_box_right.setRange(*self.right_range)
        self.update_plot(*self.int_range)

    def _init_widgets(self):
        self.spin_box_left.setRange(*self.left_range)
        self.spin_box_right.setRange(*self.right_range)

        self.spin_box_left.valueChanged.connect(lambda: self.update_plot(left=self.spin_box_left.value()))
        self.spin_box_right.valueChanged.connect(lambda: self.update_plot(right=self.spin_box_right.value()))

        self.spin_box_left.valueChanged.connect(
            lambda: self.spin_box_right.setRange(self.spin_box_left.value() + 1, self.data.shape[self.axis]))
        self.spin_box_right.valueChanged.connect(
            lambda: self.spin_box_left.setRange(1, self.spin_box_right.value() - 1))

        # self.view.ui.histogram.hide()
        # self.view.ui.roiBtn.hide()
        # self.view.ui.menuBtn.hide()

    def update_plot(self, left=None, right=None):
        if left is not None:
            self.int_range[0] = left
        if right is not None:
            self.int_range[1] = right

        self.current_integ = IntegralPlot.get_integerate(self.data, self.axis, [i - 1 for i in self.int_range])
        self.current_integ = np.flip(self.current_integ, axis=0).T
        self.view.setImage(self.current_integ)

    @staticmethod
    def get_integerate(data_cube, axis, int_range):
        # transpose the order of axis, prepare for making slices
        left, right = int_range
        tr = [axis] + list(range(data_cube.ndim))
        del tr[axis + 1]
        int_cube = np.transpose(data_cube, tr)[left: right, :, :]
        if np.any(np.isnan(int_cube)):
            int_cube[np.isnan(int_cube)] = 0
        int_map = np.sum(int_cube, axis=0)
        return int_map


class Plot4D(QObject):
    def __init__(self, view: GLViewWidget):
        self.view = view
        self.draw_method = self._gaussian_smoothed_isosurface
        self.draw_method_kwargs = {'sigma': 2,
                                   'levels': 7}
        self.draw_color_method = self._generate_hsv_color
        self.draw_color_method_kwargs = {'c': 0.,
                                         's': 0.05,
                                         'a': 0.1}

    def load_data(self, data: DataLoader):
        print(f"1: {data}")
        self.data = data.current_data.copy()
        self.update_plot()

    def update_plot(self):
        self.data = np.flip(self.data, axis=0)
        mesh_data_list = self.draw_method(self.data, **self.draw_method_kwargs)
        self.view.items.clear()
        mesh_data_list = self.draw_color_method(mesh_data_list, **self.draw_color_method_kwargs)
        meshes = [gl.GLMeshItem(meshdata=md, smooth=True, shader='balloon') for md in mesh_data_list]
        x, y, z = self.data.shape
        for m in meshes:
            m.setGLOptions('additive')
            m.translate(- x / 2, - y / 2, - z / 2)
            self.view.addItem(m)

        xgrid = gl.GLGridItem()
        ygrid = gl.GLGridItem()
        zgrid = gl.GLGridItem()
        xgrid.setSize(y, z, 1)
        ygrid.setSize(x, z, 1)
        zgrid.setSize(x, y, 1)
        xgrid.rotate(90, 0, 1, 0)
        ygrid.rotate(90, 1, 0, 0)
        xgrid.translate(- x / 2, 0, 0)
        ygrid.translate(0, - y / 2, 0, 0)
        zgrid.translate(0, 0, - z / 2)
        self.view.addItem(xgrid)
        self.view.addItem(ygrid)
        self.view.addItem(zgrid)

        axis = GLAxisItem()
        axis.setSize(x, y, z)
        axis.translate(dx=- x / 2, dy=- y / 2, dz=- z / 2)
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
