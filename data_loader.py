from PyQt5 import QtWidgets
from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot
from astropy.io import fits
import pandas as pd
import numpy as np


class DataLoader(QObject):
    fitsChanged = pyqtSignal()
    def __init__(self,
                 parent: QtWidgets.QMainWindow,
                 fits_files: list,
                 df_info_table: pd.DataFrame,
                 info_table: QtWidgets.QTableWidget,
                 file_list: QtWidgets.QListWidget,
                 all_in_mem=True):
        """
        This class will be used to manipulate the data we have, include loading, drawing and contact with the ui.
        :param fits_files: list of pathlib.Path objects
        :param info_table: pd.DataFrame, which contains the information about the data
        :param all_in_mem: read all fits into data in the first place.
        :param parent:
        :param fits_files:
        :param df_info_table:
        :param info_table:
        :param all_in_mem:
        """
        super().__init__(parent=parent)  # 初始化QObject
        self.parent = parent
        self.fits_data_cubes = {}
        self.df_info_table = df_info_table
        self.info_table = info_table
        self.file_list = file_list
        self.fits_files = fits_files

        if all_in_mem:
            for fits_file in fits_files:
                current_fits_name = fits_file.resolve().stem
                current_data = fits.getdata(str(fits_file))
                # TODO check fits dimension
                if len(current_data.shape) == 4:
                    current_data = current_data[0, :, :, :]
                # TODO handle unsuccessful read

                self.fits_data_cubes[current_fits_name] = current_data

                # self.file_list.addItem(QtWidgets.QListWidgetItem(str(current_fits_name)))

            self.file_list.currentItemChanged.connect(lambda current, previous: self.change_fits(current.text()))

            self.fits_num = len(self.fits_data_cubes.values())
            self.fits_names = sorted(list(self.fits_data_cubes.keys()))

            self.file_filter = {'YES', 'NO', 'TBD', 'nolabel'}
            self._init_filter_btn()
            self.update_file_list()

            self.current_fits = self.fits_names[0]
            self.current_data = self.fits_data_cubes[self.current_fits]
            parent.statusBar().showMessage("成功导入{} 个FITS文件，当前为{}.fits".format(self.fits_num, self.current_fits))
            self._init_label_btn()
            self._init_comment_edit()

            self.fitsChanged.emit()


        # TODO: read fits in the lazy way.

    def _init_filter_btn(self):
        btns = [self.parent.filter_YES,
                self.parent.filter_NO,
                self.parent.filter_TBD,
                self.parent.filter_nolabel]
        mask = ['YES', 'NO', 'TBD', 'nolabel']

        def update_file_filter():
            print('state changed')
            self.file_filter.clear()
            for i, b in enumerate(btns):
                print(b.text(), b.isChecked())
                if b.isChecked():
                    self.file_filter.add(mask[i])
        for b in btns:
            b.setChecked(True)
            b.stateChanged.connect(update_file_filter)
            b.stateChanged.connect(self.update_file_list)


    def update_file_list(self):
        self.file_list.clear()
        for name in self.fits_names:
            if name in self.df_info_table.index:
                current_label = self.df_info_table.loc[name, 'label']
                current_label = 'nolabel' if pd.isna(current_label) else current_label
                if current_label in self.file_filter:
                    self.file_list.addItem(QtWidgets.QListWidgetItem(str(name)))

    def change_fits(self, fits_name):
        self.current_fits = fits_name
        self.current_data = self.fits_data_cubes[self.current_fits]
        self.update_info_table()
        self.fitsChanged.emit()

    def go_to_next_fits(self):
        idx = self.fits_names.index(self.current_fits)
        if idx + 1 < self.fits_num:
            next_fits_name = self.fits_names[idx + 1]
            self.change_fits(next_fits_name)
            self.parent.statusBar().showMessage("当前FITS文件为：{}.fits".format(self.current_fits))
        else:
            self.parent.statusBar().showMessage("下一个FITS文件不存在")

    def go_to_last_fits(self):
        idx = self.fits_names.index(self.current_fits)
        if idx - 1 >= 0:
            next_fits_name = self.fits_names[idx - 1]
            self.change_fits(next_fits_name)
            self.parent.statusBar().showMessage("当前FITS文件为：{}.fits".format(self.current_fits))
        else:
            self.parent.statusBar().showMessage("上一个FITS文件不存在")

    def update_info_table(self):
        name = self.current_fits.replace(".fits", "")
        if name in self.df_info_table.index:
            row = self.df_info_table.loc[name, :]
            for i in range(self.df_info_table.shape[1]):
                self.info_table.setItem(i, 0, QtWidgets.QTableWidgetItem(str(row[i])))

    def _init_label_btn(self):
        def set_label(btn, label):
            print('label set as {}'.format(label))
            if btn.isChecked():
                self.df_info_table.loc[self.current_fits, "label"] = label
                self.update_info_table()
        self.parent.yes_btn.toggled.connect(lambda : set_label(self.parent.yes_btn, 'YES'))
        self.parent.no_btn.toggled.connect(lambda : set_label(self.parent.no_btn,'NO'))
        self.parent.tbd_btn.toggled.connect(lambda : set_label(self.parent.tbd_btn,'TBD'))

        def update_btn():
            current_label = self.df_info_table.loc[self.current_fits, "label"]
            print(current_label)
            self.parent.label_btn_group.setExclusive(False)
            self.parent.yes_btn.setChecked(False)
            self.parent.no_btn.setChecked(False)
            self.parent.tbd_btn.setChecked(False)
            self.parent.label_btn_group.setExclusive(True)

            if current_label == 'YES':
                self.parent.yes_btn.setChecked(True)
            elif current_label == 'NO':
                self.parent.no_btn.setChecked(True)
            elif current_label == 'TBD':
                self.parent.tbd_btn.setChecked(True)
            else:
                pass

        self.fitsChanged.connect(update_btn)



    def _init_comment_edit(self):
        def set_comment(text):
            print(text)
            self.df_info_table.loc[self.current_fits, "comment"] = text
            self.update_info_table()
        self.parent.save_comment_btn.clicked.connect(lambda : set_comment(self.parent.comment_edit.text()))

        def update_comment_text():
            current_comment = self.df_info_table.loc[self.current_fits, "comment"]
            if pd.isna(current_comment):
                self.parent.comment_edit.clear()
            else:
                self.parent.comment_edit.setText(str(current_comment))
        self.fitsChanged.connect(update_comment_text)
