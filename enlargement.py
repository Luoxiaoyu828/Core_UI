from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *


# class ImageWithMouseControl(QWidget):
#     def __init__(self, parent=None):
#         super().__init__(parent)
#         self.parent = parent
#         print(self.parent.x(), self.parent.y())
#         self.img = QPixmap("./picture/001.png")
#         self.scaled_img = self.img.scaled(self.size())
#         self.point = QPoint(self.parent.x(), self.parent.y())
#
#     def mouseMoveEvent(self, e):
#         if self.left_click:
#             self._endPos = e.pos() - self._startPos
#             self.point = self.point + self._endPos
#             self._startPos = e.pos()
#             self.repaint()
#
#     def mousePressEvent(self, e):
#         if e.button() == Qt.LeftButton:
#             self.left_click = True
#             self._startPos = e.pos()
#
#     def mouseReleaseEvent(self, e):
#         if e.button() == Qt.LeftButton:
#             self.left_click = False
#         elif e.button() == Qt.RightButton:
#             self.point = QPoint(self.parent.x(), self.parent.y())
#             self.scaled_img = self.img.scaled(self.size())
#             self.repaint()
#
#     def wheelEvent(self, e):
#         if e.angleDelta().y() > 0:
#             # 放大图片
#             self.scaled_img = self.img.scaled(self.scaled_img.width() - 5, self.scaled_img.height() - 5)
#             new_w = e.x() - (self.scaled_img.width() * (e.x() - self.point.x())) / (self.scaled_img.width() + 5)
#             new_h = e.y() - (self.scaled_img.height() * (e.y() - self.point.y())) / (self.scaled_img.height() + 5)
#             self.point = QPoint(new_w, new_h)
#             self.repaint()
#         elif e.angleDelta().y() < 0:
#             # 缩小图片
#             self.scaled_img = self.img.scaled(self.scaled_img.width() + 5, self.scaled_img.height() + 5)
#             new_w = e.x() - (self.scaled_img.width() * (e.x() - self.point.x())) / (self.scaled_img.width() - 5)
#             new_h = e.y() - (self.scaled_img.height() * (e.y() - self.point.y())) / (self.scaled_img.height() - 5)
#             self.point = QPoint(new_w, new_h)
#             self.repaint()
#
#     def resizeEvent(self, e):
#         if self.parent is not None:
#             self.scaled_img = self.img.scaled(self.size())
#             self.point = QPoint(self.parent.x(), self.parent.y())
#             self.update()

class ImageWithMouseControl(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        print(parent)
        self.parent = parent
        self.img = QPixmap('./picture/001.png')
        self.scaled_img = self.img.scaled(self.size())
        self.point = QPoint(self.parent.x(), self.parent.y())

    def paintEvent(self, e):
        """
        绘图
        :param e:
        :return:
        """
        painter = QPainter()
        painter.begin(self)
        self.draw_img(painter)
        painter.end()

    def draw_img(self, painter):
        painter.drawPixmap(self.point, self.scaled_img)

    def mouseMoveEvent(self, e):  # 重写移动事件
        if self.left_click:
            self._endPos = e.pos() - self._startPos
            self.point = self.point + self._endPos
            self._startPos = e.pos()
            self.repaint()

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self.left_click = True
            self._startPos = e.pos()

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.LeftButton:
            self.left_click = False
        elif e.button() == Qt.RightButton:
            self.point = QPoint(self.parent.x(), self.parent.y())
            self.scaled_img = self.img.scaled(self.size())
            self.repaint()

    def wheelEvent(self, e):
        if e.angleDelta().y() > 0:
            # 放大图片
            self.scaled_img = self.img.scaled(self.scaled_img.width()-5, self.scaled_img.height()-5)
            new_w = e.x() - (self.scaled_img.width() * (e.x() - self.point.x())) / (self.scaled_img.width() + 5)
            new_h = e.y() - (self.scaled_img.height() * (e.y() - self.point.y())) / (self.scaled_img.height() + 5)
            self.point = QPoint(new_w, new_h)
            self.repaint()
        elif e.angleDelta().y() < 0:
            # 缩小图片
            self.scaled_img = self.img.scaled(self.scaled_img.width()+5, self.scaled_img.height()+5)
            new_w = e.x() - (self.scaled_img.width() * (e.x() - self.point.x())) / (self.scaled_img.width() - 5)
            new_h = e.y() - (self.scaled_img.height() * (e.y() - self.point.y())) / (self.scaled_img.height() - 5)
            self.point = QPoint(new_w, new_h)
            self.repaint()

    def resizeEvent(self, e):
        if self.parent is not None:
            self.scaled_img = self.img.scaled(self.size())
            self.point = QPoint(self.parent.x(), self.parent.y())
            self.update()