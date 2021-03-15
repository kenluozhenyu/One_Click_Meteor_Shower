# -*- coding: UTF-8 -*-
import sys
import os
import shutil
import threading

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow, QDialog, QFileDialog, QMessageBox
from PyQt5.QtCore import QStringListModel, QObject, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QPixmap

from main_ui_chn import Ui_MainWindow
# from main_ui_en import Ui_MainWindow

# from numba import cuda
import keras
import tensorflow as tf

import detection
import gen_mask

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def get_folder_list(processFolder):
    process_dir = os.path.join(processFolder, 'process')
    detection_dir = os.path.join(process_dir, '01_detection')
    extracted_dir = os.path.join(process_dir, '02_cropped')
    filtered_dir = os.path.join(process_dir, '03_filtered')
    keep_dir = os.path.join(filtered_dir, 'good')
    removed_dir = os.path.join(filtered_dir, 'removed')
    mosaic_dir = os.path.join(process_dir, '04_mosaic')
    gray_256_dir = os.path.join(process_dir, '05_gray_256')
    mask_256_dir = os.path.join(process_dir, '06_mask_256')
    mask_resize_back_dir = os.path.join(process_dir, '07_mask_resize_back')
    mosaic_merge_back_dir = os.path.join(process_dir, '08_mosaic_merged_back')
    object_extracted_dir = os.path.join(process_dir, '09_object_extracted')
    FINAL_dir = os.path.join(process_dir, '10_FINAL')
    FINAL_w_label_dir = os.path.join(process_dir, '10_FINAL_w_label')
    FINAL_combined_dir = os.path.join(process_dir, '11_FINAL_combined')

    return process_dir, detection_dir, extracted_dir, filtered_dir, keep_dir, removed_dir,\
           mosaic_dir, gray_256_dir, mask_256_dir, mask_resize_back_dir, mosaic_merge_back_dir, \
           object_extracted_dir, FINAL_dir, FINAL_w_label_dir, FINAL_combined_dir


def Step_1_Process_Detection(processFolder, is_equatorial_mount):
    original_dir = processFolder

    # process_dir = os.path.join(original_dir, 'process')
    # extracted_dir = os.path.join(process_dir, '02_cropped')
    # filtered_dir = os.path.join(process_dir, '03_filtered')
    # keep_dir = os.path.join(filtered_dir, 'good')
    # removed_dir = os.path.join(filtered_dir, 'removed')

    process_dir, detection_dir, extracted_dir, filtered_dir, keep_dir, removed_dir, \
    mosaic_dir, gray_256_dir, mask_256_dir, mask_resize_back_dir, mosaic_merge_back_dir, \
    object_extracted_dir, FINAL_dir, FINAL_w_label_dir, FINAL_combined_dir = get_folder_list(processFolder)

    print("\n\n==========================================================")
    print("Starting step 1 processing ...")

    detection.multi_thread_process_detect_n_extract_meteor_from_folder(original_dir,
                                                                       process_dir,
                                                                       subtraction=True,
                                                                       # subtraction=False,
                                                                       equatorial_mount=is_equatorial_mount,
                                                                       verbose=1)

    detection.filter_possible_not_meteor_objects(extracted_dir, keep_dir, removed_dir)

    # device = cuda.get_current_device()
    # device.reset()
    # cuda.select_device(0)
    # cuda.close()

    # Need to clean up the keras session
    # to allow it to be run in the 2nd time
    keras.backend.clear_session()

    print("\n==========================================================")
    print("Possible objects extraction finished.")
    print("You may perform step 2/3.")


# Due to the detection procedure is using multiple-thread, to allow
# the output log be able to display on GUI simultaneously, we cannot
# block the main thread.
# So we need to invoke a new sub-thread from the main thread first.
# And this sub-thread will call the detection procedure.
class Detection_sub_thread_called_by_main(threading.Thread, QObject):

    detection_finish_signal = pyqtSignal()

    def __init__(self, processFolder, is_equatorial_mount):
        threading.Thread.__init__(self)
        QObject.__init__(self)
        self.processFolder = processFolder
        self.is_equatorial_mount = is_equatorial_mount

    def run(self):
        Step_1_Process_Detection(self.processFolder, self.is_equatorial_mount)
        self.detection_finish_signal.emit()


# There's no multi-threading used in this procedure.
# Just run the procedures directly. No need to invoke
# a new thread.
def Step_3_Generate_Mask(processFolder):
    # original_dir = processFolder
    # process_dir = os.path.join(original_dir, 'process')
    # extracted_dir = os.path.join(process_dir, '02_cropped')
    # filtered_dir = os.path.join(process_dir, '03_filtered')
    # keep_dir = os.path.join(filtered_dir, 'good')
    # removed_dir = os.path.join(filtered_dir, 'removed')

    # mosaic_dir = os.path.join(process_dir, '04_mosaic')
    # gray_256_dir = os.path.join(process_dir, '05_gray_256')
    # mask_256_dir = os.path.join(process_dir, '06_mask_256')
    # mask_resize_back_dir = os.path.join(process_dir, '07_mask_resize_back')
    # mosaic_merge_back_dir = os.path.join(process_dir, '08_mosaic_merged_back')

    process_dir, detection_dir, extracted_dir, filtered_dir, keep_dir, removed_dir, \
    mosaic_dir, gray_256_dir, mask_256_dir, mask_resize_back_dir, mosaic_merge_back_dir, \
    object_extracted_dir, FINAL_dir, FINAL_w_label_dir, FINAL_combined_dir = get_folder_list(processFolder)

    print("\n\n==========================================================")
    print("Starting step 3 processing ...")

    my_gen_mask = gen_mask.Gen_mask()

    my_gen_mask.convert_cropped_image_folder_to_mosaic_for_big_files(keep_dir, mosaic_dir)
    my_gen_mask.convert_image_folder_to_gray_256(mosaic_dir, gray_256_dir)
    my_gen_mask.gen_meteor_mask_from_folder(gray_256_dir, mask_256_dir)
    my_gen_mask.resize_mask_to_original_cropped_size(mask_256_dir, mask_resize_back_dir)
    my_gen_mask.mosaic_mask_files_merge_back(mask_resize_back_dir, mosaic_merge_back_dir)

    # del my_gen_mask

    # Need to clean up the keras session
    # to allow it to be run in the 2nd time
    keras.backend.clear_session()

    print("\n==========================================================")
    print("Mask generation finished.")
    print("You may perform step 4/5.")


class Genmask_sub_thread_called_by_main(threading.Thread, QObject):

    genmask_finish_signal = pyqtSignal()

    def __init__(self, processFolder):
        threading.Thread.__init__(self)
        QObject.__init__(self)
        self.processFolder = processFolder

    def run(self):
        Step_3_Generate_Mask(self.processFolder)
        self.genmask_finish_signal.emit()


def Step_5_Generate_Final(processFolder):
    original_dir = processFolder

    # process_dir = os.path.join(original_dir, 'process')
    # extracted_dir = os.path.join(process_dir, '02_cropped')
    # filtered_dir = os.path.join(process_dir, '03_filtered')
    # keep_dir = os.path.join(filtered_dir, 'good')
    # removed_dir = os.path.join(filtered_dir, 'removed')

    # mosaic_dir = os.path.join(process_dir, '04_mosaic')
    # gray_256_dir = os.path.join(process_dir, '05_gray_256')
    # mask_256_dir = os.path.join(process_dir, '06_mask_256')
    # mask_resize_back_dir = os.path.join(process_dir, '07_mask_resize_back')
    # mosaic_merge_back_dir = os.path.join(process_dir, '08_mosaic_merged_back')

    # object_extracted_dir = os.path.join(process_dir, '09_object_extracted')
    # FINAL_dir = os.path.join(process_dir, '10_FINAL')
    # FINAL_w_label_dir = os.path.join(process_dir, '10_FINAL_w_label')
    # FINAL_combined_dir = os.path.join(process_dir, '11_FINAL_combined')

    process_dir, detection_dir, extracted_dir, filtered_dir, keep_dir, removed_dir, \
    mosaic_dir, gray_256_dir, mask_256_dir, mask_resize_back_dir, mosaic_merge_back_dir, \
    object_extracted_dir, FINAL_dir, FINAL_w_label_dir, FINAL_combined_dir = get_folder_list(processFolder)

    print("\n\n==========================================================")
    print("Starting step 5 processing ...")

    my_gen_mask = gen_mask.Gen_mask()

    my_gen_mask.extract_meteor_from_original_folder_with_mask(original_dir,
                                                              mosaic_merge_back_dir,
                                                              object_extracted_dir,
                                                              verbose=1)

    my_gen_mask.extend_extracted_objects_to_original_photo_size_by_multi_threading(object_extracted_dir,
                                                                                   FINAL_dir,
                                                                                   FINAL_w_label_dir)

    my_gen_mask.combine_meteor_images_to_one(FINAL_dir, FINAL_combined_dir, 'final.png', verbose=1)
    my_gen_mask.combine_meteor_images_to_one(FINAL_w_label_dir, FINAL_combined_dir, 'final_w_label.png', verbose=1)

    print("\n==========================================================")
    print("Final output files generation finished!")
    print("You may follow step 6 to get the output files.")


class Final_sub_thread_called_by_main(threading.Thread, QObject):

    final_finish_signal = pyqtSignal()

    def __init__(self, processFolder):
        threading.Thread.__init__(self)
        QObject.__init__(self)
        self.processFolder = processFolder

    def run(self):
        Step_5_Generate_Final(self.processFolder)
        self.final_finish_signal.emit()


# No used now
class Logger(object):
    def __init__(self, textWritten, st=sys.stdout):
        # self.terminal = st
        self.log = textWritten

    def write(self, message):
        # self.terminal.write(message)
        self.log(message)

    def flush(self):
        pass


class EmittingStream(QtCore.QObject):
    textWritten = QtCore.pyqtSignal(str)

    def write(self, text):
        self.textWritten.emit(str(text))

    def flush(self):
        pass


class MyMainForm(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        # self.ui = Ui_Dialog()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        # self.ui.groupBox_3.resize(400, self.ui.groupBox_3.size().height())

        # logoPixMap = QPixmap('images/GZSA-logo-240.jpg')
        logoPixMap = QPixmap('images/GZSA-logo-160.jpg')
        self.ui.logoLabel.setPixmap(logoPixMap)

        meteorPixMap = QPixmap('images/Meteor_shower_190.jpg')
        self.ui.meteorLabel.setPixmap(meteorPixMap)

        self.ui.selectFolderButton.clicked.connect(self.displayFolderDialog)
        self.ui.doDetectionButton.clicked.connect(self.Step_1_DoDetection)
        self.ui.openDetectionFolderButton.clicked.connect(self.Step_2_OpenDetectionFolder)

        self.ui.generateMaskButton.clicked.connect(self.Step_3_DoGenerateMask)
        self.ui.openMaskFolderButton.clicked.connect(self.Step_4_OpenMaskFolder)

        self.ui.generateFinalButton.clicked.connect(self.Step_5_DoFinalGeneration)
        self.ui.openFinalFolderButton.clicked.connect(self.Step_6_OpenFinalFolders)

        self.ui.folderNameText.setText("(No folder selected)")
        self.processFolder = ""
        # self.ui.doDetectionButton.setEnabled(False)
        self.change_GUI_control_status(False)
        self.ui.selectFolderButton.setEnabled(True)
        # self.ui.selectFolderButton.setStyleSheet("background-color: rgb(205, 205, 205)")
        self.ui.selectFolderButton.setStyleSheet("background-color: rgb(205, 205, 205);color: rgb(0, 0, 255)")

        self.stdoutbak = sys.stdout
        self.stderrbak = sys.stderr

        # If we want to use this, need to implement a self.write() function
        # sys.stdout = self
        # sys.stderr = self

        # sys.stdout = Logger(textWritten=self.outputWritten)
        # sys.stderr = Logger(textWritten=self.outputWritten)

        sys.stdout = EmittingStream(textWritten=self.outputWritten)
        sys.stderr = EmittingStream(textWritten=self.outputWritten)

        # self.threads = MyThread(self)
        # self.threads.trigger.connect(self.update_log)
        # self.threads.run_(message='')

    def outputWritten(self, text):
        # cursor = self.ui.processLogsEdit.textCursor()
        # cursor.movePosition(QtGui.QTextCursor.End)
        # cursor.insertText(text)
        # self.ui.processLogsEdit.setTextCursor(cursor)
        self.ui.processLogsEdit.insertPlainText(text)
        self.ui.processLogsEdit.ensureCursorVisible()
        # QtWidgets.qApp.processEvents()
        QApplication.processEvents()
        # self.stdoutbak.write(text)

    # Use only when this is called in the __init__:
    #   sys.stdout = self
    #
    # But seems when terminating the program an error
    # will be reported, though didn't affect the usage
    def write(self, info):
        self.ui.processLogsEdit.insertPlainText(info)
        QtWidgets.qApp.processEvents()

    def restoreStd(self):
        sys.stdout = self.stdoutbak
        sys.stderr = self.stderrbak

    def __del__(self):
        self.restoreStd()

    def resizeEvent(self, e):
        # print("w = {0}; h = {1}".format(e.size().width(), e.size().height()))
        if e.size().width() > 1440:
            # Here the 1024 is the form height. Then 770 is the height of the groupBox_3
            self.ui.groupBox_3.resize(self.ui.groupBox_3.size().width(), e.size().height() - (1024 - 770) + 8)

            self.ui.groupBox.resize(self.ui.groupBox.size().width(), e.size().height() - 35)

            # Here the 1440 is the form width. Then 591 is the width of the groupBox_2
            self.ui.groupBox_2.resize(e.size().width() - 1440 + 591 - 10, e.size().height()-35)
        QtWidgets.QWidget.resizeEvent(self, e)

    def change_GUI_control_status(self, status):
        self.ui.selectFolderButton.setEnabled(status)
        self.ui.isEQmountCheckBox.setEnabled(status)
        self.ui.doDetectionButton.setEnabled(status)
        self.ui.openDetectionFolderButton.setEnabled(status)
        self.ui.generateMaskButton.setEnabled(status)
        self.ui.openMaskFolderButton.setEnabled(status)
        self.ui.generateFinalButton.setEnabled(status)
        self.ui.openFinalFolderButton.setEnabled(status)

        if status:
            self.ui.selectFolderButton.setStyleSheet("background-color: rgb(205, 205, 205);color: rgb(0, 0, 255)")
            # self.ui.isEQmountCheckBox.setStyleSheet("color: rgb(0, 0, 255)")
            self.ui.doDetectionButton.setStyleSheet("background-color: rgb(205, 205, 205);color: rgb(0, 0, 255)")
            self.ui.openDetectionFolderButton.setStyleSheet("background-color: rgb(205, 205, 205);color: rgb(0, 0, 255)")
            self.ui.generateMaskButton.setStyleSheet("background-color: rgb(205, 205, 205);color: rgb(0, 0, 255)")
            self.ui.openMaskFolderButton.setStyleSheet("background-color: rgb(205, 205, 205);color: rgb(0, 0, 255)")
            self.ui.generateFinalButton.setStyleSheet("background-color: rgb(205, 205, 205);color: rgb(0, 0, 255)")
            self.ui.openFinalFolderButton.setStyleSheet("background-color: rgb(205, 205, 205);color: rgb(0, 0, 255)")
        else:
            self.ui.selectFolderButton.setStyleSheet("background-color: rgb(205, 205, 205);color: rgb(120, 120, 120)")
            # self.ui.isEQmountCheckBox.setStyleSheet("color: rgb(244, 244, 244)")
            self.ui.doDetectionButton.setStyleSheet("background-color: rgb(205, 205, 205);color: rgb(120, 120, 120)")
            self.ui.openDetectionFolderButton.setStyleSheet("background-color: rgb(205, 205, 205);color: rgb(120, 120, 120)")
            self.ui.generateMaskButton.setStyleSheet("background-color: rgb(205, 205, 205);color: rgb(120, 120, 120)")
            self.ui.openMaskFolderButton.setStyleSheet("background-color: rgb(205, 205, 205);color: rgb(120, 120, 120)")
            self.ui.generateFinalButton.setStyleSheet("background-color: rgb(205, 205, 205);color: rgb(120, 120, 120)")
            self.ui.openFinalFolderButton.setStyleSheet("background-color: rgb(205, 205, 205);color: rgb(120, 120, 120)")

    def displayFolderDialog(self):
        # self.ui.displayText.setText("Button clicked")

        dialog = QFileDialog(self, 'Image files', directory='')
        dialog.setFileMode(QFileDialog.DirectoryOnly)
        # dialog.setSidebarUrls([QtCore.QUrl.fromLocalFile(place)])

        fileDir = ""
        if dialog.exec_() == QDialog.Accepted:
            fileDir = dialog.selectedFiles()[0]
            self.ui.folderNameText.setText(fileDir)
            self.processFolder = fileDir

            included_extensions = ['jpg', 'JPG', 'jpeg', 'JPEG', 'bmp', 'BMP', 'png', 'PNG', 'tif', 'TIF', 'tiff',
                                   'TIFF']

            photo_list = [fn for fn in os.listdir(self.processFolder)
                         if any(fn.endswith(ext) for ext in included_extensions)]

            file_num = len(photo_list)
            fileList2View = QStringListModel()
            fileList2View.setStringList(photo_list)
            self.ui.listView.setModel(fileList2View)
            self.ui.fileNumberLabel.setText("{} image file(s) in the list".format(file_num))
            if file_num > 0:
                # self.ui.doDetectionButton.setEnabled(True)
                self.change_GUI_control_status(True)
            else:
                self.change_GUI_control_status(False)
                self.ui.selectFolderButton.setEnabled(True)
                self.ui.selectFolderButton.setStyleSheet("background-color: rgb(205, 205, 205);color: rgb(0, 0, 255)")

    def Step_1_DoDetection(self):
        if self.processFolder != "":

            process_dir, detection_dir, extracted_dir, filtered_dir, keep_dir, removed_dir, \
            mosaic_dir, gray_256_dir, mask_256_dir, mask_resize_back_dir, mosaic_merge_back_dir, \
            object_extracted_dir, FINAL_dir, FINAL_w_label_dir, FINAL_combined_dir = get_folder_list(self.processFolder)

            if os.path.exists(keep_dir):
                msg = QMessageBox()
                msg.setWindowTitle("Confirmation needed")
                msg.setText("It seems the object detection for this folder had been done before. \
                                    \n\nDo you want to clean up the previous output data and re-run ?")
                msg.setIcon(QMessageBox.Question)
                msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
                msg.setDefaultButton(QMessageBox.No)
                msg.setStyleSheet("QLabel{ color: white}")
                msg.setStyleSheet("text-color: rgb(0, 0, 0);")

                if msg.exec_() == QMessageBox.Yes:
                    try:
                        shutil.rmtree(detection_dir)
                    except OSError as e:
                        print("Error: %s - %s." % (e.filename, e.strerror))

                    try:
                        shutil.rmtree(extracted_dir)
                    except OSError as e:
                        print("Error: %s - %s." % (e.filename, e.strerror))

                    try:
                        shutil.rmtree(filtered_dir)
                    except OSError as e:
                        print("Error: %s - %s." % (e.filename, e.strerror))
                else:
                    return

            is_equatorial_mount = False
            if self.ui.isEQmountCheckBox.checkState() == QtCore.Qt.Checked:
                is_equatorial_mount = True

            process_thread = Detection_sub_thread_called_by_main(self.processFolder, is_equatorial_mount)
            process_thread.detection_finish_signal.connect(self.Step_1_Detection_Process_Finsihed)

            self.change_GUI_control_status(False)
            process_thread.start()

    def Step_1_Detection_Process_Finsihed(self):
        # print("\nDetection process finished !")
        self.change_GUI_control_status(True)

    def Step_2_OpenDetectionFolder(self):
        if self.processFolder == "":
            QMessageBox.information(self, "Info", "No image folder selected", QMessageBox.Ok)
            return

        # original_dir = self.processFolder
        # process_dir = os.path.join(original_dir, 'process')
        # extracted_dir = os.path.join(process_dir, '02_cropped')
        # filtered_dir = os.path.join(process_dir, '03_filtered')
        # keep_dir = os.path.join(filtered_dir, 'good')
        # removed_dir = os.path.join(filtered_dir, 'removed')

        process_dir, detection_dir, extracted_dir, filtered_dir, keep_dir, removed_dir, \
        mosaic_dir, gray_256_dir, mask_256_dir, mask_resize_back_dir, mosaic_merge_back_dir, \
        object_extracted_dir, FINAL_dir, FINAL_w_label_dir, FINAL_combined_dir = get_folder_list(self.processFolder)

        if not os.path.exists(keep_dir) or not os.path.exists(removed_dir):
            msg = QMessageBox()
            msg.setWindowTitle("Info")
            msg.setText("Seems object detection not performed yet. \n\nPlease try step 1 first.")
            msg.setIcon(QMessageBox.Information)
            msg.setStandardButtons(QMessageBox.Ok)
            msg.setStyleSheet("QLabel{ color: white}")
            msg.setStyleSheet("text-color: rgb(0, 0, 0);")

            # QMessageBox.information(self, "Info",
            #                         "Seems object detection not performed yet. \n\nPlease try step 1 first.",
            #                         QMessageBox.Ok)
            msg.exec_()
            return

        os.startfile(removed_dir)
        os.startfile(keep_dir)

    def Step_3_DoGenerateMask(self):
        # original_dir = self.processFolder
        # process_dir = os.path.join(original_dir, 'process')
        # extracted_dir = os.path.join(process_dir, '02_cropped')
        # filtered_dir = os.path.join(process_dir, '03_filtered')
        # keep_dir = os.path.join(filtered_dir, 'good')
        # removed_dir = os.path.join(filtered_dir, 'removed')

        process_dir, detection_dir, extracted_dir, filtered_dir, keep_dir, removed_dir, \
        mosaic_dir, gray_256_dir, mask_256_dir, mask_resize_back_dir, mosaic_merge_back_dir, \
        object_extracted_dir, FINAL_dir, FINAL_w_label_dir, FINAL_combined_dir = get_folder_list(self.processFolder)

        if not os.path.exists(keep_dir):
            # QMessageBox.information(self, "Info",
            #                         "Seems object detection not performed yet. \n\nPlease try step 1 first.",
            #                         QMessageBox.Ok)

            msg = QMessageBox()
            msg.setWindowTitle("Info")
            msg.setText("Seems object detection not performed yet. \n\nPlease try step 1 first.")
            msg.setIcon(QMessageBox.Information)
            msg.setStandardButtons(QMessageBox.Ok)
            msg.setStyleSheet("QLabel{ color: white}")
            msg.setStyleSheet("text-color: rgb(0, 0, 0);")
            msg.exec_()
            return

        if os.path.exists(mosaic_merge_back_dir):
            msg = QMessageBox()
            msg.setWindowTitle("Confirmation needed")
            msg.setText("It seems the mask generation for this folder had been done before. \
                                    \n\nDo you want to clean up the previous output data and re-run ?")
            msg.setIcon(QMessageBox.Question)
            msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
            msg.setStyleSheet("QLabel{ color: white}")
            msg.setStyleSheet("text-color: rgb(0, 0, 0);")
            msg.setDefaultButton(QMessageBox.No)
            # if QMessageBox.question(self, "Confirmation needed",
            #                         "It seems the mask generation for this folder had been done before. \
            #                         \n\nDo you want to clean up the previous output data and re-run ?",
            #                         QMessageBox.Yes | QMessageBox.No) == QMessageBox.Yes:
            if msg.exec_() == QMessageBox.Yes:
                try:
                    shutil.rmtree(mosaic_dir)
                except OSError as e:
                    print("Error: %s - %s." % (e.filename, e.strerror))

                try:
                    shutil.rmtree(gray_256_dir)
                except OSError as e:
                    print("Error: %s - %s." % (e.filename, e.strerror))

                try:
                    shutil.rmtree(mask_256_dir)
                except OSError as e:
                    print("Error: %s - %s." % (e.filename, e.strerror))

                try:
                    shutil.rmtree(mask_resize_back_dir)
                except OSError as e:
                    print("Error: %s - %s." % (e.filename, e.strerror))

                try:
                    shutil.rmtree(mosaic_merge_back_dir)
                except OSError as e:
                    print("Error: %s - %s." % (e.filename, e.strerror))
            else:
                return

        process_thread = Genmask_sub_thread_called_by_main(self.processFolder)
        process_thread.genmask_finish_signal.connect(self.Step_3_GenMask_Process_Finsihed)

        self.change_GUI_control_status(False)
        process_thread.start()

    def Step_3_GenMask_Process_Finsihed(self):
        # print("\nMask generation process finished !")
        self.change_GUI_control_status(True)

    def Step_4_OpenMaskFolder(self):
        if self.processFolder == "":
            # QMessageBox.information(self, "Info", "No image folder selected", QMessageBox.Ok)

            msg = QMessageBox()
            msg.setWindowTitle("Info")
            msg.setText("No image folder selected")
            msg.setIcon(QMessageBox.Information)
            msg.setStandardButtons(QMessageBox.Ok)
            msg.setStyleSheet("QLabel{ color: white}")
            msg.setStyleSheet("text-color: rgb(0, 0, 0);")
            msg.exec_()
            return

        # original_dir = self.processFolder
        # process_dir = os.path.join(original_dir, 'process')
        # extracted_dir = os.path.join(process_dir, '02_cropped')
        # filtered_dir = os.path.join(process_dir, '03_filtered')
        # keep_dir = os.path.join(filtered_dir, 'good')
        # removed_dir = os.path.join(filtered_dir, 'removed')
        # mosaic_dir = os.path.join(process_dir, '04_mosaic')
        # gray_256_dir = os.path.join(process_dir, '05_gray_256')
        # mask_256_dir = os.path.join(process_dir, '06_mask_256')
        # mask_resize_back_dir = os.path.join(process_dir, '07_mask_resize_back')
        # mosaic_merge_back_dir = os.path.join(process_dir, '08_mosaic_merged_back')

        process_dir, detection_dir, extracted_dir, filtered_dir, keep_dir, removed_dir, \
        mosaic_dir, gray_256_dir, mask_256_dir, mask_resize_back_dir, mosaic_merge_back_dir, \
        object_extracted_dir, FINAL_dir, FINAL_w_label_dir, FINAL_combined_dir = get_folder_list(self.processFolder)

        if not os.path.exists(mosaic_merge_back_dir):
            # QMessageBox.information(self, "Info",
            #                         "Seems mask file not generated yet. \n\nPlease try step 3 first.",
            #                         QMessageBox.Ok)
            msg = QMessageBox()
            msg.setWindowTitle("Info")
            msg.setText("Seems mask file not generated yet. \n\nPlease try step 3 first.")
            msg.setIcon(QMessageBox.Information)
            msg.setStandardButtons(QMessageBox.Ok)
            msg.setStyleSheet("QLabel{ color: white}")
            msg.setStyleSheet("text-color: rgb(0, 0, 0);")
            msg.exec_()
            return

        os.startfile(mosaic_merge_back_dir)

    def Step_5_DoFinalGeneration(self):
        if self.processFolder == "":
            # QMessageBox.information(self, "Info", "No image folder selected", QMessageBox.Ok)

            msg = QMessageBox()
            msg.setWindowTitle("Info")
            msg.setText("No image folder selected")
            msg.setIcon(QMessageBox.Information)
            msg.setStandardButtons(QMessageBox.Ok)
            msg.setStyleSheet("QLabel{ color: white}")
            msg.setStyleSheet("text-color: rgb(0, 0, 0);")
            msg.exec_()
            return

        # original_dir = self.processFolder
        # process_dir = os.path.join(original_dir, 'process')
        # extracted_dir = os.path.join(process_dir, '02_cropped')
        # filtered_dir = os.path.join(process_dir, '03_filtered')
        # keep_dir = os.path.join(filtered_dir, 'good')
        # removed_dir = os.path.join(filtered_dir, 'removed')
        # mosaic_dir = os.path.join(process_dir, '04_mosaic')
        # gray_256_dir = os.path.join(process_dir, '05_gray_256')
        # mask_256_dir = os.path.join(process_dir, '06_mask_256')
        # mask_resize_back_dir = os.path.join(process_dir, '07_mask_resize_back')
        # mosaic_merge_back_dir = os.path.join(process_dir, '08_mosaic_merged_back')

        process_dir, detection_dir, extracted_dir, filtered_dir, keep_dir, removed_dir, \
        mosaic_dir, gray_256_dir, mask_256_dir, mask_resize_back_dir, mosaic_merge_back_dir, \
        object_extracted_dir, FINAL_dir, FINAL_w_label_dir, FINAL_combined_dir = get_folder_list(self.processFolder)

        if not os.path.exists(mosaic_merge_back_dir):
            # QMessageBox.information(self, "Info",
            #                         "Seems mask file not generated yet. \n\nPlease try step 3 first.",
            #                         QMessageBox.Ok)

            msg = QMessageBox()
            msg.setWindowTitle("Info")
            msg.setText("Seems mask file not generated yet. \n\nPlease try step 3 first.")
            msg.setIcon(QMessageBox.Information)
            msg.setStandardButtons(QMessageBox.Ok)
            msg.setStyleSheet("QLabel{ color: white}")
            msg.setStyleSheet("text-color: rgb(0, 0, 0);")
            msg.exec_()
            return

        if os.path.exists(FINAL_combined_dir):
            msg = QMessageBox()
            msg.setWindowTitle("Confirmation needed")
            msg.setText("It seems the final output generation for this folder had been done before. \
                                    \n\nDo you want to clean up the previous output data and re-run ?")
            msg.setIcon(QMessageBox.Question)
            msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
            msg.setStyleSheet("QLabel{ color: white}")
            msg.setStyleSheet("text-color: rgb(0, 0, 0);")
            msg.setDefaultButton(QMessageBox.No)

            # if QMessageBox.question(self, "Confirmation needed",
            #                         "It seems the final output generation for this folder had been done before. \
            #                         \n\nDo you want to clean up the previous output data and re-run ?",
            #                         QMessageBox.Yes | QMessageBox.No) == QMessageBox.Yes:
            if msg.exec_() == QMessageBox.Yes:
                try:
                    shutil.rmtree(object_extracted_dir)
                except OSError as e:
                    print("Error: %s - %s." % (e.filename, e.strerror))

                try:
                    shutil.rmtree(FINAL_dir)
                except OSError as e:
                    print("Error: %s - %s." % (e.filename, e.strerror))

                try:
                    shutil.rmtree(FINAL_w_label_dir)
                except OSError as e:
                    print("Error: %s - %s." % (e.filename, e.strerror))

                try:
                    shutil.rmtree(FINAL_combined_dir)
                except OSError as e:
                    print("Error: %s - %s." % (e.filename, e.strerror))
            else:
                return

        process_thread = Final_sub_thread_called_by_main(self.processFolder)
        process_thread.final_finish_signal.connect(self.Step_5_Final_Process_Finsihed)

        self.change_GUI_control_status(False)
        process_thread.start()

    def Step_5_Final_Process_Finsihed(self):
        # print("\nFinal output files generation finished !")
        self.change_GUI_control_status(True)

    def Step_6_OpenFinalFolders(self):
        if self.processFolder == "":
            # QMessageBox.information(self, "Info", "No image folder selected", QMessageBox.Ok)

            msg = QMessageBox()
            msg.setWindowTitle("Info")
            msg.setText("No image folder selected")
            msg.setIcon(QMessageBox.Information)
            msg.setStandardButtons(QMessageBox.Ok)
            msg.setStyleSheet("QLabel{ color: white}")
            msg.setStyleSheet("text-color: rgb(0, 0, 0);")
            msg.exec_()
            return

        # original_dir = self.processFolder
        # process_dir = os.path.join(original_dir, 'process')
        # extracted_dir = os.path.join(process_dir, '02_cropped')
        # filtered_dir = os.path.join(process_dir, '03_filtered')
        # keep_dir = os.path.join(filtered_dir, 'good')
        # removed_dir = os.path.join(filtered_dir, 'removed')
        # mosaic_dir = os.path.join(process_dir, '04_mosaic')
        # gray_256_dir = os.path.join(process_dir, '05_gray_256')
        # mask_256_dir = os.path.join(process_dir, '06_mask_256')
        # mask_resize_back_dir = os.path.join(process_dir, '07_mask_resize_back')
        # mosaic_merge_back_dir = os.path.join(process_dir, '08_mosaic_merged_back')
        # object_extracted_dir = os.path.join(process_dir, '09_object_extracted')
        # FINAL_dir = os.path.join(process_dir, '10_FINAL')
        # FINAL_w_label_dir = os.path.join(process_dir, '10_FINAL_w_label')
        # FINAL_combined_dir = os.path.join(process_dir, '11_FINAL_combined')

        process_dir, detection_dir, extracted_dir, filtered_dir, keep_dir, removed_dir, \
        mosaic_dir, gray_256_dir, mask_256_dir, mask_resize_back_dir, mosaic_merge_back_dir, \
        object_extracted_dir, FINAL_dir, FINAL_w_label_dir, FINAL_combined_dir = get_folder_list(self.processFolder)

        if not os.path.exists(FINAL_dir) \
                or not os.path.exists(FINAL_w_label_dir) \
                or not os.path.exists(FINAL_combined_dir):
            # QMessageBox.information(self, "Info",
            #                         "Seems the final output files are not generated yet. \n\nPlease try step 5 first.",
            #                         QMessageBox.Ok)

            msg = QMessageBox()
            msg.setWindowTitle("Info")
            msg.setText("Seems the final output files are not generated yet. \n\nPlease try step 5 first.")
            msg.setIcon(QMessageBox.Information)
            msg.setStandardButtons(QMessageBox.Ok)
            msg.setStyleSheet("QLabel{ color: white}")
            msg.setStyleSheet("text-color: rgb(0, 0, 0);")
            msg.exec_()

            return

        os.startfile(FINAL_dir)
        os.startfile(FINAL_w_label_dir)
        os.startfile(FINAL_combined_dir)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)

    myWin = MyMainForm()
    myWin.show()

    sys.exit(app.exec_())
