# -*- coding: utf-8 -*-
"""
3/11/25

@author: andreas.peckhaus - altered by Mitchell Sueker
"""

# ------------------------------------------------------
# ---------------------- Main_GUI.py -------------------
# ------------------------------------------------------
from PyQt5.QtWidgets import*
from PyQt5.uic import loadUi
from PyQt5 import QtCore, QtGui, QtWidgets
from matplotlib.backends.backend_qt5agg import (NavigationToolbar2QT as NavigationToolbar)

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QApplication,QLabel,QMainWindow,QPushButton,QVBoxLayout,QWidget, QMessageBox, QInputDialog, QComboBox)
from PyQt5.QtCore import QObject, QThread, pyqtSignal
from PyQt5.QtCore import (Qt, QObject, pyqtSignal, pyqtSlot, QRunnable, QThreadPool)

import numpy as np
import pandas as pd
import TScan_Kombi
import time
import matplotlib.pyplot as plt
import os.path
import csv
import datetime
import sys
from time import sleep
from pathlib import Path
import math
import scipy.optimize as opt
from scipy.optimize import curve_fit 
from scipy.signal import savgol_filter


if hasattr(QtCore.Qt, 'AA_EnableHighDpiScaling'):
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)

if hasattr(QtCore.Qt, 'AA_UseHighDpiPixmaps'):
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)

# QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True) #enable highdpi scaling
# QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True) #use highdpi icons

import os
#os.environ["QT_SCALE_FACTOR"] = "1.25"
os.environ["QT_ENABLE_HIGHDPI_SCALING"]   = "1"
os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
os.environ["QT_SCALE_FACTOR"]             = "1"

class WorkerSignals(QObject):
    progress = pyqtSignal(int)
    finished = pyqtSignal()
    result = pyqtSignal()

class Worker(QRunnable):
    signals = WorkerSignals()
    
    def __init__(self,VIS,NIR,VISNIR,cycleAll,writeToExcel,number,
                 radioButton_1, radioButton_2, radioButton_3, radioButton_5, buttonALL):
        super().__init__()
        self.signals = WorkerSignals()
        self.VIS=VIS
        self.NIR=NIR
        self.VISNIR=VISNIR
        self.cycleAll=cycleAll
        self.writeToExcel=writeToExcel
        self.number= number
        self.radioButton_1=radioButton_1
        self.radioButton_2=radioButton_2
        self.radioButton_3=radioButton_3
        self.radioButton_5=radioButton_5
        self.buttonALL = buttonALL
        self.is_paused = False
        self.is_killed = False
        
    @pyqtSlot()
    def run(self):
        for i in range(self.number):
            self.signals.progress.emit(i + 1)
            if self.radioButton_1.isChecked():
                res=self.VIS()
                if self.radioButton_5.isChecked():
                    res2=self.writeToExcel()
            if self.radioButton_2.isChecked():
                res=self.NIR()
                if self.radioButton_5.isChecked():
                    res2=self.writeToExcel()
            if self.radioButton_3.isChecked():
                res=self.VISNIR()
                if self.radioButton_5.isChecked():
                    res2=self.writeToExcel()
            if self.buttonALL.isChecked():
                res=self.cycleAll()
                if self.radioButton_5.isChecked():
                    res2=self.writeToExcel()
            while self.is_paused==True:
                time.sleep(0)
            if self.is_killed==True:
                break
        self.signals.finished.emit()
            
    def pause(self):
        self.is_paused = True
    def resume(self):
        self.is_paused = False
    def kill(self):
        self.is_killed = True

class Worker2(QRunnable):
    signals = WorkerSignals()
    
    def __init__(self,DarkSpectrum_VIS,DarkSpectrum_NIR,writeToExcel,number,
                 radioButton_1, radioButton_2, radioButton_3, radioButton_5, textBrowser):
        super().__init__()
        self.signals = WorkerSignals()
        self.DarkSpectrum_VIS=DarkSpectrum_VIS
        self.DarkSpectrum_NIR=DarkSpectrum_NIR
        self.writeToExcel=writeToExcel
        self.number= number
        self.radioButton_1=radioButton_1
        self.radioButton_2=radioButton_2
        self.radioButton_3=radioButton_3
        self.radioButton_5=radioButton_5
        self.textBrowser=textBrowser
        self.is_paused = False
        self.is_killed = False
        
    @pyqtSlot()
    def run(self):
        results = []
        results2 = []
        for i in range(self.number):
            self.signals.progress.emit(i + 1)
            if self.radioButton_1.isChecked():
                res=self.DarkSpectrum_VIS()
                res= list(res)
                results.append(res)
                if self.radioButton_5.isChecked():
                    res2=self.writeToExcel()
            if self.radioButton_2.isChecked():
                res=self.DarkSpectrum_NIR()
                res= list(res)
                results.append(res)
                if self.radioButton_5.isChecked():
                    res2=self.writeToExcel()
            if self.radioButton_3.isChecked():
                res10=self.DarkSpectrum_VIS()
                res11=self.DarkSpectrum_NIR()
                res10= list(res10)
                res11= list(res11)
                results.append(res10)
                results2.append(res11)
                if self.radioButton_5.isChecked():
                    res2=self.writeToExcel()
            while self.is_paused==True:
                time.sleep(0)
            if self.is_killed==True:
                break
        # self.first_elements = list(map(lambda x: x[0], results)) 
        # self.second_elements = list(map(lambda x: x[1], results))
        # self.MeanOfMean = sum(self.first_elements) / len(self.first_elements)
        # self.MeanOfSD = sum(self.second_elements) / len(self.second_elements)
        #self.textBrowser.append(str(self.MeanOfMean))
        #self.textBrowser.append('Mean: '+ str(round(self.MeanOfMean,2))+' counts')
        #self.textBrowser.append(str(self.MeanOfSD))
        #self.textBrowser.append('Standard deviation: '+ str(round(self.MeanOfSD,2))+' counts')
        self.signals.finished.emit()
            
    def pause(self):
        self.is_paused = True
    def resume(self):
        self.is_paused = False
    def kill(self):
        self.is_killed = True

class MatplotlibWidget(QMainWindow):
    
    def __init__(self, parent=None):
        super().__init__(parent)
        QMainWindow.__init__(self)
        loadUi("MS_UI_File.ui",self)
        
        # Counter to track the number of activations
        self.click_count = 0
        #ts = TScan_Kombi.TScan()
        self.ts = TScan_Kombi.TScan()
        self.setWindowTitle("Mobile multimodal spectroscopy system")
        self.setWindowIcon(QtGui.QIcon("Image/prism_3"))
        
        # Set default spectrometer parameters
        self.lineEdit_1.setText("10")
        self.lineEdit_1.returnPressed.connect(self.error_handling)
        self.spinBox_1.setValue(5)
        self.spinBox_1.valueChanged.connect(self.error_handling) 
        self.lineEdit_2.setText("1.1")
        self.lineEdit_2.returnPressed.connect(self.set_parameter_Gain_VIS)
        
        self.lineEdit_3.setText("100")
        self.lineEdit_3.returnPressed.connect(self.error_handling)
        self.spinBox_2.setValue(5)
        self.spinBox_2.valueChanged.connect(self.error_handling) 
        self.lineEdit_4.setText("1.1")
        self.lineEdit_4.returnPressed.connect(self.set_parameter_Gain_NIR)
        
        # Set default light source parameters
        self.spinBox_3.setValue(1) # Light bulb
        self.spinBox_3.valueChanged.connect(self.error_handling2) 
        self.lineEdit_5.setText("200")
        self.lineEdit_5.returnPressed.connect(self.error_handling2)
        self.lineEdit_6.setText("0")
        self.lineEdit_6.returnPressed.connect(self.set_parameter_sdelay_VIS)
        
        self.spinBox_4.setValue(1) # Light bulb
        self.spinBox_4.valueChanged.connect(self.error_handling2) 
        self.lineEdit_7.setText("200")
        self.lineEdit_7.returnPressed.connect(self.error_handling2)
        self.lineEdit_8.setText("0")
        self.lineEdit_8.returnPressed.connect(self.set_parameter_sdelay_NIR)
        self.lineEdit_9.setText("0")

        # Integration times for CycleAll function
        self.lineEdit_11.setText("10")
        self.lineEdit_11.returnPressed.connect(self.error_handling)
        self.lineEdit_12.setText("10")
        self.lineEdit_12.returnPressed.connect(self.error_handling)
        self.lineEdit_13.setText("10")
        self.lineEdit_13.returnPressed.connect(self.error_handling)
        self.lineEdit_14.setText("10")
        self.lineEdit_14.returnPressed.connect(self.error_handling)

        # Integration Time ranges
        self.lineEdit_19.setText("10:100:10")
        self.lineEdit_19.returnPressed.connect(self.error_handling_range)
        self.lineEdit_20.setText("10:100:10")
        self.lineEdit_20.returnPressed.connect(self.error_handling_range)

        # Current for CycleAll function
        self.lineEdit_15.setText("200")
        self.lineEdit_15.returnPressed.connect(self.error_handling2)
        self.lineEdit_16.setText("200")
        self.lineEdit_16.returnPressed.connect(self.error_handling2)
        self.lineEdit_17.setText("200")
        self.lineEdit_17.returnPressed.connect(self.error_handling2)
        self.lineEdit_18.setText("200")
        self.lineEdit_18.returnPressed.connect(self.error_handling2)

        self.spinBox_5.setValue(0)
        self.spinBox_5.valueChanged.connect(self.error_handling3) 

        # Save settings
        self.checkBox_1.setChecked(True)
        self.checkBox_2.setChecked(True)
        self.checkBox_3.setChecked(True)
        self.checkBox_4.setChecked(True)
        self.checkBox_5.setChecked(True)
        self.checkBox_6.setChecked(False)
        self.checkBox_7.setChecked(True)
        self.checkBox_8.setChecked(True)
        self.checkBox_9.setChecked(True)
        self.checkBox_10.setChecked(True)
        self.checkBox_11.setChecked(False)
        self.checkBox_12.setChecked(False)
        self.checkBox_13.setChecked(False)
        
        # Second Tab Widget
        self.lineEdit_21.setText("0")
        self.lineEdit_22.setText("0")
        self.lineEdit.setText("1")
        self.lineEdit_10.setText("3")
        self.spinBox_7.setValue(2)
        self.spinBox_8.setValue(0)
        # Set ComboBoxes
        Mode_MA = ['reflect', 'constant', 'zero', 'none']
        self.comboBox_1.addItems(Mode_MA)
        Mode_SB = ['interp', 'constant', 'nearest', 'wrap', 'mirror']
        self.comboBox_2.addItems(Mode_SB)
        PixelResolution = ['512', '1024']
        self.comboBox_3.addItems(PixelResolution)
        
        # Definition of Signal/Slot functions
        self.actionRun.triggered.connect(self.select)
        #self.actionSave.triggered.connect(self.save_data)
        self.actionSave.triggered.connect(self.writeToExcel)
        self.actionRefreshGraph.triggered.connect(self.refresh)
        self.actionOpen.triggered.connect(self.get_Directory)
        self.actionSave_settings.triggered.connect(self.save_settings)
        self.actionExit.triggered.connect(self.exit_program)
        self.actionAbout.triggered.connect(self.about_program)
        self.actionHelp.triggered.connect(self.help_program)
        self.CurveFitting.clicked.connect(self.GaussianFit)
        self.actionMaximize.triggered.connect(self.Maximize)
        self.actionMinimize.triggered.connect(self.Minimize)
        self.actionSize.triggered.connect(self.Resize)
        self.MA.clicked.connect(self.moving_average)
        self.SG.clicked.connect(self.apply_savitzky_golay)

        self.actionRuns.triggered.connect(self.Treading)
        self.actionConnect.triggered.connect(self.connection)
        self.actionDark_Spectrum.triggered.connect(self.Treading2)

        # Setup navigationtoolbar
        self.nav1 = NavigationToolbar(self.MplWidget_1.canvas, self, coordinates=False)
        self.nav1.setMinimumHeight(330)
        self.nav1.setMaximumWidth(30)
        #self.addToolBar(QtCore.Qt.LeftToolBarArea, self.nav1)
        self.nav1.setStyleSheet("QToolBar {spacing: 20px; border: 20px }")
        self.nav1.setOrientation(QtCore.Qt.Vertical)
        self.nav1.move(995,110)
        self.nav1.setIconSize(QtCore.QSize(24, 24))
        self.nav1.setMovable(False)
        
        self.nav2 = NavigationToolbar(self.MplWidget_2.canvas, self, coordinates=False)
        self.nav2.setMinimumHeight(330)
        self.nav2.setMaximumWidth(30)
        self.nav2.setStyleSheet("QToolBar {spacing: 0px; border: 0px }")
        self.nav2.setOrientation(QtCore.Qt.Vertical)
        self.nav2.move(990,410)
        self.nav2.setMovable(False)
        self.nav2.setIconSize(QtCore.QSize(24, 24))
        self.directory=""
        self.RadioGroup = QButtonGroup()
        self.RadioGroup.addButton(self.radioButton_5)
        
    def Maximize(self):
        self.showMaximized()
    def Minimize(self):
        self.showMinimized()
    def Resize(self):
        self.setFixedWidth(1708)
        self.setFixedHeight(958)

    def Treading(self):
        self.threadpool = QThreadPool()
        self.thread = QThread()
        self.runner = Worker(self.VIS, self.NIR, self.VISNIR, self.cycleAll, self.writeToExcel, self.spinBox_5.value(),
                             self.radioButton_1, self.radioButton_2, self.radioButton_3, self.radioButton_5, self.buttonALL)
        self.runner.signals.progress.connect(self.reportProgress)
        self.threadpool.start(self.runner)
        self.actionStop.triggered.connect(self.runner.kill)
        self.actionPause.triggered.connect(self.runner.pause)
        self.actionContinue.triggered.connect(self.runner.resume)
        self.actionRuns.setEnabled(False)
        self.actionStop.triggered.connect(lambda: self.actionRuns.setEnabled(True))
        self.runner.signals.finished.connect(lambda: self.actionRuns.setEnabled(True))
        self.runner.signals.finished.connect(self.deselect)
        self.show()
        
    def Treading2(self):
        self.threadpool = QThreadPool()
        self.thread = QThread()
        self.runner2 = Worker2(self.DarkSpectrum_VIS, self.DarkSpectrum_NIR, self.writeToExcel, self.spinBox_5.value(),
                             self.radioButton_1, self.radioButton_2, self.radioButton_3, self.radioButton_5, self.textBrowser)
        self.runner2.signals.progress.connect(self.reportProgress)
        self.threadpool.start(self.runner2)
        self.actionStop.triggered.connect(self.runner2.kill)
        self.actionPause.triggered.connect(self.runner2.pause)
        self.actionContinue.triggered.connect(self.runner2.resume)
        self.actionRuns.setEnabled(False)
        self.actionStop.triggered.connect(lambda: self.actionRuns.setEnabled(True))
        self.runner2.signals.finished.connect(lambda: self.actionRuns.setEnabled(True))
        self.runner2.signals.finished.connect(self.deselect)
        self.show()
        
    def deselect(self):
        self.RadioGroup.setExclusive(False)
        self.radioButton_5.setChecked(False)
        self.RadioGroup.setExclusive(True) 
        
    def finished(self):
        self.actionRuns.setEnabled(True)
        self.threadpool.quit()
        self.threadpool.wait()

    def reportProgress(self, n):
        self.textBrowser.setText(f"Measurement run: {n}")
        
    def connection(self):
        # Zähler inkrementieren
        self.click_count += 1

        # Abwechselnde Aktionen ausführen
        if self.click_count % 2 == 1:
            self.connect_on()
        else:
            self.connect_off()

    def connect_on(self):
        try:
            self.textBrowser.setText("Connected: "+ts.find_device())
            self.textBrowser.append("Press button again to disconnect the device")
            ts.connect()
        except:
            self.textBrowser.setText("Device is not connected")

    def connect_off(self):
        try: 
            ts.disconnect()
            self.textBrowser.setText("Device is disconnected")
        except:
            self.textBrowser.setText("Device is connected")
    
    def GaussianFit(self):
        if self.radioButton_4.isChecked():
            df=pd.DataFrame(list(zip(self.wavelengths1, self.sample1)))
            Start=(float(self.lineEdit_21.text()))
            End=(float(self.lineEdit_22.text()))
            
            ts.set_msm('VIS')
            VIS_params = ts.get_wavelengths_params()
            #Coeffiencts of VIS Spectrometer 
            a=VIS_params[0]
            b=VIS_params[1]
            c=VIS_params[2]
            #print(a,b,c)
            ROI_Start = (-b+math.sqrt(b**2-4*a*(c-Start)))/(2*a)
            ROI_End = (-b+math.sqrt(b**2-4*a*(c-End)))/(2*a)
            #print (ROI_Start)
            #print (ROI_End)

            newdf = df.truncate(before=ROI_Start, after=ROI_End)
            X = newdf.iloc[:, 0]
            Y = newdf.iloc[:, 1]
            
            X_min=X.min(axis=0)
            X_max=X.max(axis=0)
            X_range=np.arange(X_min, X_max, 0.1).tolist()
    
            def gauss(x, H, A, x0, sigma):
                return H + A * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))
    
            def gauss_fit(x, y):
                mean = sum(x * y) / sum(y)
                sigma = np.sqrt(sum(y * (x - mean) ** 2) / sum(y))
                popt, pcov = curve_fit(gauss, x, y, p0=[min(y), max(y), mean, sigma])
                return popt
            
            H, A, x0, sigma = gauss_fit(X, Y)
            FWHM = 2.35482 * sigma
        
            self.WL=X_range
            self.INTEN=gauss(X_range, *gauss_fit(X, Y))            
            self.get_plot1(self.WL, self.INTEN)
            self.textBrowser.setText('Central wavelength: '+ str(round(x0,2))+' nm'
                                      +"\n"+'Maximum intensity: '+ str(round(H+A,2))
                                      +"\n"+'Full width at half maximum (FWHM): '+ str(round(FWHM,2))+' nm')
    
        if self.radioButton_6.isChecked(): 
            df=pd.DataFrame(list(zip(self.wavelengths2, self.sample2)))
            Start=(float(self.lineEdit_21.text()))
            End=(float(self.lineEdit_22.text())) 
            
            ts.set_msm('NIR')
            NIR_params = ts.get_wavelengths_params()
            #Coeffiencts of VIS Spectrometer 
            a=NIR_params[0]
            b=NIR_params[1]
            c=NIR_params[2]
            #print(a,b,c)
            ROI_Start = (-b+math.sqrt(b**2-4*a*(c-Start)))/(2*a)
            ROI_End = (-b+math.sqrt(b**2-4*a*(c-End)))/(2*a)
            #print (ROI_Start)
            #print (ROI_End)
            
            newdf = df.truncate(before=ROI_Start, after=ROI_End)
            X = newdf.iloc[:, 0]
            Y = newdf.iloc[:, 1]
            
            X_min=X.min(axis=0)
            X_max=X.max(axis=0)
            X_range=np.arange(X_min, X_max, 0.1).tolist()
    
            def gauss(x, H, A, x0, sigma):
                return H + A * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))
    
            def gauss_fit(x, y):
                mean = sum(x * y) / sum(y)
                sigma = np.sqrt(sum(y * (x - mean) ** 2) / sum(y))
                popt, pcov = curve_fit(gauss, x, y, p0=[min(y), max(y), mean, sigma])
                return popt
            
            H, A, x0, sigma = gauss_fit(X, Y)
            FWHM = 2.35482 * sigma
            
            self.WL=X_range
            self.INTEN=gauss(X_range, *gauss_fit(X, Y))
            self.get_plot2(self.WL, self.INTEN)
            self.textBrowser.setText('Central wavelength: '+ str(round(x0,2))+'nm'
                                      +"\n"+'Maximum intensity: '+ str(round(H+A,2))
                                      +"\n"+'Full width at half maximum (FWHM): '+ str(round(FWHM,2))+'nm')
          
    def moving_average(self):
       """
       Calculates the moving average for 1D data.
       
       Parameter:
           data (array-like): Input data (e.g. time series or spectrum).
           window_size (int): Size of the window (must be > 1).
           boundary (str): Boundary treatment method, one of the following:
                           - 'reflect': Reflects the data at the boundary (default).
                           - 'constant': Fills with constant values.
                           - 'zero': Fills with zeros.
                           - 'none': Truncates the filter, no values at the border.
       
       Return:
           np.ndarray: Filtered data.
       """
       if self.radioButton.isChecked():
           WL= self.wavelengths1
           data = self.sample1
           window_size =(int(self.lineEdit.text()))
           boundary = str(self.comboBox_1.currentText())
           
           if window_size < 1:
               #raise ValueError("window_size muss mindestens 1 sein.")
               self.textBrowser.setText("window size must be at least 1.")
           if window_size % 2 == 0:
               #raise ValueError("window_size muss ungerade sein.")
               self.textBrowser.setText("window size must be odd.")  
           
           data = np.asarray(data)
           half_window = window_size // 2

           # Boundary condition
           if boundary == 'reflect':
               padded_data = np.pad(data, half_window, mode='reflect')
           elif boundary == 'constant':
               padded_data = np.pad(data, half_window, mode='edge')
           elif boundary == 'zero':
               padded_data = np.pad(data, half_window, mode='constant', constant_values=0)
           elif boundary == 'none':
               # No padding: Truncate the output
               convolved = np.convolve(data, np.ones(window_size) / window_size, mode='valid')
               return convolved
           else:
               #raise ValueError("Unbekannte Randbedingung. Wähle 'reflect', 'constant', 'zero', oder 'none'.")
               self.textBrowser.setText("Unknown boundary condition. Select 'reflect', 'constant', 'zero', 'none'.")
               
           # Calculate moving average
           convolved = np.convolve(padded_data, np.ones(window_size) / window_size, mode='valid')
           return self.get_plot1(WL, convolved)
           
       if self.radioButton_7.isChecked():
           WL= self.wavelengths2
           data = self.sample2
           window_size =(int(self.lineEdit.text()))
           boundary = str(self.comboBox_1.currentText())
           
           if window_size < 1:
               #raise ValueError("window_size muss mindestens 1 sein.")
               self.textBrowser.setText("window size must be at least 1.")
           if window_size % 2 == 0:
               #raise ValueError("window_size muss ungerade sein.")
               self.textBrowser.setText("window size must be odd.")  
           
           data = np.asarray(data)
           half_window = window_size // 2

           # Boundary condition
           if boundary == 'reflect':
               padded_data = np.pad(data, half_window, mode='reflect')
           elif boundary == 'constant':
               padded_data = np.pad(data, half_window, mode='edge')
           elif boundary == 'zero':
               padded_data = np.pad(data, half_window, mode='constant', constant_values=0)
           elif boundary == 'none':
               # No padding: Truncate the output
               convolved = np.convolve(data, np.ones(window_size) / window_size, mode='valid')
               return convolved
           else:
               #raise ValueError("Unbekannte Randbedingung. Wähle 'reflect', 'constant', 'zero', oder 'none'.")
               self.textBrowser.setText("Unknown boundary condition. Select 'reflect', 'constant', 'zero', 'none'.")
               
           # Calculate moving average
           convolved = np.convolve(padded_data, np.ones(window_size) / window_size, mode='valid')
           return self.get_plot2(WL, convolved)
       
    def apply_savitzky_golay(self):
        """
        Applies the Savitzky-Golay filter to 1D data.
        
        Parameters:
            data (array-like): Input data (e.g. spectrum or time series).
            window_size (int): Size of the window (must be odd).
            poly_order (int): Order of the polynomial (must be smaller than window_size).
            deriv (int): Order of the derivative (0 = smoothing, 1 = first derivative, ...).
            mode (str): Edge handling method:
                        - 'interp': Interpolation (default, recommended).
                        - 'mirror': Symmetrical mirroring.
                        - 'nearest': Uses the nearest edge values.
                        - 'constant': Constant boundary values.
                        - 'wrap': Cyclic continuation.
        Return:
            np.ndarray: Filtered data.
        """
        window_size = int(self.lineEdit_10.text())
        poly_order = int(self.spinBox_7.text())
        if window_size % 2 == 0 or window_size <= poly_order:
            self.textBrowser.setText("Window size must be odd and larger than polynom order.")
            #raise ValueError("window_size muss ungerade und größer als poly_order sein.")
            
        if self.radioButton_8.isChecked():
            data = self.sample1
            WL= self.wavelengths1
            window_size = int(self.lineEdit_10.text())
            poly_order = int(self.spinBox_7.text())
            deriv = int(self.spinBox_8.text())
            mode = str(self.comboBox_2.currentText())
            convolved=savgol_filter(data, window_length=window_size, polyorder=poly_order, deriv=deriv, mode=mode)    
            return self.get_plot1(WL, convolved)    

        if self.radioButton_9.isChecked():
            data = self.sample2
            WL= self.wavelengths2
            window_size = int(self.lineEdit_10.text())
            poly_order = int(self.spinBox_7.text())
            deriv = int(self.spinBox_8.text())
            mode = str(self.comboBox_2.currentText())
            convolved=savgol_filter(data, window_length=window_size, polyorder=poly_order, deriv=deriv, mode=mode)    
            return self.get_plot2(WL, convolved)                
            
    def set_parameter_Gain_VIS(self):
        if ((float(self.lineEdit_2.text()) <=5) and (float(self.lineEdit_2.text())>=1)):
            ts.set_msm('VIS')
            command = '*PARA:GAIN '+ str(self.lineEdit_2.text())
            ts.send_command("'"+ command +"'")
            self.textBrowser.setText("Set analog gain for VIS:"+ str(self.lineEdit_2.text()))
        else: self.textBrowser.setText("Analog gain can be set in the range from 1-5")
        
    def set_parameter_Gain_NIR(self):
        if (float(self.lineEdit_4.text())==1.1):
            ts.set_msm('NIR')
            #print('*PARA:GAIN '+ str(self.lineEdit_4.text()))
            command = '*PARA:GAIN '+ str(self.lineEdit_4.text())
            #print("'"+command+"'")
            self.textBrowser.setText("Set analog gain for NIR:"+ str(self.lineEdit_4.text()))
            ts.send_command("'"+ command +"'") 
        else: self.textBrowser.setText("Analog gain must be set to 1.1")
        
    def set_parameter_sdelay_VIS(self):
        ts.set_msm('VIS')
        command = '*PARA:SDEL '+ str(self.lineEdit_6.text())
        ts.send_command("'"+ command +"'")
        
    def set_parameter_sdelay_NIR(self):
        ts.set_msm('NIR')
        command = '*PARA:SDEL '+ str(self.lineEdit_8.text())
        ts.send_command("'"+ command +"'")

    def get_parameters(self):
            print("Return pressed!")
            print (float(self.lineEdit_1.text()))
           
    def error_handling(self):
        Message= "Parameter is out of range."
        # error_dialog = QtWidgets.QErrorMessage()
        # error_dialog.showMessage('Error')
        #Integration time VIS  
        if (((float(self.lineEdit_1.text()))>= 0.01) and ((float(self.lineEdit_1.text()))<= 10000)):
            pass
        elif ((float(self.lineEdit_1.text()))<0.01):
            self.textBrowser.append(Message)
            self.lineEdit_1.setText("0.02")
        elif ((float(self.lineEdit_1.text()))>10000):
            self.textBrowser.append(Message)
            self.lineEdit_1.setText("10000")
        else: 
            print("Invalid input")  
            #Number of scans VIS
        if ((int(self.spinBox_1.text()))>= 1):
            pass
        elif ((int(self.spinBox_1.text()))<= 0):
            self.textBrowser.append(Message)
            self.spinBox_1.setValue(1)
  
          #Integration time NIR  
        if (((float(self.lineEdit_3.text()))>= 0.01) and ((float(self.lineEdit_3.text()))<= 10000)):
            pass
        elif ((float(self.lineEdit_3.text()))<0.01):
            self.textBrowser.append(Message)
            self.lineEdit_3.setText("0.01")
        elif ((float(self.lineEdit_3.text()))>10000):
            self.textBrowser.append(Message)
            self.lineEdit_3.setText("10000")
        else: 
            print("Invalid input")  
            #Number of scans NIR
        if ((int(self.spinBox_2.text()))>= 1):
            pass
        elif ((int(self.spinBox_2.text()))<= 0):
            self.textBrowser.append(Message)
            self.spinBox_2.setValue(1)

        # Integration time VIS for CycleAll
        if (((float(self.lineEdit_11.text())) >= 0.01) and ((float(self.lineEdit_11.text())) <= 10000)):
            pass
        elif ((float(self.lineEdit_11.text())) < 0.01):
            self.textBrowser.append(Message)
            self.lineEdit_11.setText("0.02")
        elif ((float(self.lineEdit_11.text())) > 10000):
            self.textBrowser.append(Message)
            self.lineEdit_11.setText("10000")
        else:
            print("Invalid input")

        # Integration time NIR for CycleAll
        if (((float(self.lineEdit_12.text())) >= 0.01) and ((float(self.lineEdit_12.text())) <= 10000)):
            pass
        elif ((float(self.lineEdit_12.text())) < 0.01):
            self.textBrowser.append(Message)
            self.lineEdit_12.setText("0.02")
        elif ((float(self.lineEdit_12.text())) > 10000):
            self.textBrowser.append(Message)
            self.lineEdit_12.setText("10000")
        else:
            print("Invalid input")

        # Integration time FL365 for CycleAll
        if (((float(self.lineEdit_13.text())) >= 0.01) and ((float(self.lineEdit_13.text())) <= 10000)):
            pass
        elif ((float(self.lineEdit_13.text())) < 0.01):
            self.textBrowser.append(Message)
            self.lineEdit_13.setText("0.02")
        elif ((float(self.lineEdit_13.text())) > 10000):
            self.textBrowser.append(Message)
            self.lineEdit_13.setText("10000")
        else:
            print("Invalid input")

        # Integration time FL405 for CycleAll
        if (((float(self.lineEdit_14.text())) >= 0.01) and ((float(self.lineEdit_14.text())) <= 10000)):
            pass
        elif ((float(self.lineEdit_14.text())) < 0.01):
            self.textBrowser.append(Message)
            self.lineEdit_14.setText("0.02")
        elif ((float(self.lineEdit_14.text())) > 10000):
            self.textBrowser.append(Message)
            self.lineEdit_14.setText("10000")
        else:
            print("Invalid input")

    def error_handling2(self):
        Message= "Parameter is out of range."
        # VIS
        if (int(self.spinBox_3.text())) == 0:
            self.textBrowser.append(Message)
            self.spinBox_3.setValue(1)
        if (int(self.spinBox_3.text())) == 1:
            if ((float(self.lineEdit_5.text())) <= 700): 
                pass
            elif((float(self.lineEdit_5.text())) > 700):
                self.textBrowser.append(Message)
                self.lineEdit_5.setText("700")
        if (int(self.spinBox_3.text())) == 2:
            if ((float(self.lineEdit_5.text())) <= 700): 
                pass
            elif((float(self.lineEdit_5.text())) > 700):
                self.textBrowser.append(Message)
                self.lineEdit_5.setText("700") 
        if (int(self.spinBox_3.text())) == 3:
            if ((float(self.lineEdit_5.text())) <= 700): 
                pass
            elif((float(self.lineEdit_5.text())) > 700):
                self.textBrowser.append(Message)
                self.lineEdit_5.setText("700")
        if (int(self.spinBox_3.text())) == 4:
            if ((float(self.lineEdit_5.text())) <= 700): 
                pass
            elif((float(self.lineEdit_5.text())) > 700):
                self.textBrowser.append(Message)
                self.lineEdit_5.setText("700")
        if (int(self.spinBox_3.text())) >= 5:
            self.textBrowser.append(Message)
            self.spinBox_3.setValue(4)
            
        # NIR
        if (int(self.spinBox_4.text())) == 0:
            self.textBrowser.append(Message)
            self.spinBox_4.setValue(1)
        if (int(self.spinBox_4.text())) == 1:
            if ((float(self.lineEdit_7.text())) <= 700): 
                pass
            elif((float(self.lineEdit_7.text())) > 700):
                self.textBrowser.append(Message)
                self.lineEdit_7.setText("700")
        if (int(self.spinBox_4.text())) == 2:
            if ((float(self.lineEdit_7.text())) <= 700): 
                pass
            elif((float(self.lineEdit_7.text())) > 700):
                self.textBrowser.append(Message)
                self.lineEdit_7.setText("700") 
        if (int(self.spinBox_4.text())) == 3:
            if ((float(self.lineEdit_7.text())) <= 700): 
                pass
            elif((float(self.lineEdit_7.text())) > 700):
                self.textBrowser.append(Message)
                self.lineEdit_7.setText("700")
        if (int(self.spinBox_4.text())) == 4:
            if ((float(self.lineEdit_7.text())) <= 700): 
                pass
            elif((float(self.lineEdit_7.text())) > 700):
                self.textBrowser.append(Message)
                self.lineEdit_7.setText("700")
        if (int(self.spinBox_4.text())) >= 5:
            self.textBrowser.append(Message)
            self.spinBox_4.setValue(4)

        # Cycle All
        if ((float(self.lineEdit_15.text())) <= 700):
            pass
        elif ((float(self.lineEdit_15.text())) > 700):
            self.textBrowser.append(Message)
            self.lineEdit_15.setText("700")
        if ((float(self.lineEdit_16.text())) <= 700):
            pass
        elif ((float(self.lineEdit_16.text())) > 700):
            self.textBrowser.append(Message)
            self.lineEdit_16.setText("700")
        if ((float(self.lineEdit_17.text())) <= 700):
            pass
        elif ((float(self.lineEdit_17.text())) > 700):
            self.textBrowser.append(Message)
            self.lineEdit_17.setText("700")
        if ((float(self.lineEdit_18.text())) <= 700):
            pass
        elif ((float(self.lineEdit_18.text())) > 700):
            self.textBrowser.append(Message)
            self.lineEdit_18.setText("700")
            
    def error_handling3(self):
        if (int(self.spinBox_5.text())) == 0:
            self.textBrowser.append("Select a number higher than zero")
        elif (int(self.spinBox_5.text())) > 0:
            pass

    def error_handling_range(self):
        acquisition_range = self.lineEdit_19
        start_time, end_time, step_size = map(int, acquisition_range.split(":"))

        pattern = r"^\d+:\d+:\d+$"  # Ensure correct format
        if not re.match(pattern, acquisition_range):
            self.textBrowser.append("Invalid format! Use Start:End:Step (e.g., 10:100:10) with valid numbers.")
            return None

        parts = user_input.split(":")  # split the inputs into (start, end, stepsize)
        # Ensure there are exactly three parts
        if not all(part.isdigit() for part in parts):
            self.textBrowser.append("Input only numbers between each ':'")
            return None
        if start_time >= end_time or step_size <= 0:
            self.textBrowser.append("Ensure Start < End and Step > 0.")
            return None
        else:
            pass

    def stop (self):
        print(self.actionStop.isChecked())
        if self.actionStop.isChecked() == False:
            print("pass")
            
    def exit_program(self):
        self.textBrowser.append('Exiting the program')
        #print('Exiting the program')
        ts.disconnect()
        self.close()
        
    def about_program(self):
            self.window = Window2()
            self.window.show()  
            
    def help_program(self):
            self.window = Window3()
            self.window.show()        

    def DarkSpectrum_VIS(self):
        #setup VIS spectrometer
        ts.set_msm('VIS')
        t_aquisition_VIS = float(self.lineEdit_1.text())
        NumberofScans_VIS = int(self.spinBox_1.text())
        Channel_VIS = int(self.spinBox_3.text())
        Current_VIS = float(self.lineEdit_5.text())
        
        #Warm-up time for lamp
        command = '*PARA:SDEL '+ str(self.lineEdit_6.text())
        ts.send_command("'"+ command +"'")
        
        spec1 = ts.spec_measure(t_aquisition_VIS ,NumberofScans_VIS) # Dark measurement
        spec1=np.array(spec1)
        self.Dark_VIS=spec1
        self.sample1=self.Dark_VIS
        #Spectrometer detector
        Pixnr_VIS = int(self.comboBox_3.currentText()) #total number of pixel
        Pixelnumber=Pixnr_VIS
        Pixscale = np.array(range(0, Pixelnumber))
        
        VIS_params = ts.get_wavelengths_params()
        
        #Coeffiencts of VIS Spectrometer 
        a=VIS_params[0]
        b=VIS_params[1]
        c=VIS_params[2]
        
        self.wavelengths1 = [(a * Pixscale*Pixscale + b * Pixscale + c) for Pixscale in range(0, Pixelnumber)]
        self.wavelengths1_formatted= [ '%.2f' % elem for elem in self.wavelengths1]
        self.wavelengths1_formatted =list(np.float_(self.wavelengths1_formatted))
        self.sample1=self.sample1[10:]
        self.wavelengths1=self.wavelengths1[10:]
        self.Mean_VIS=np.mean(self.sample1)
        self.SD_VIS=np.std(self.sample1)
        # self.textBrowser.append(str(self.Mean_VIS))
        self.textBrowser.append('Mean: '+ str(round(self.Mean_VIS,2))+' counts')
        # self.textBrowser.append(str(self.SD_VIS))
        self.textBrowser.append('Standard deviation: '+ str(round(self.SD_VIS,2))+' counts')
        self.get_plot_VIS()
        self.get_temperature()
        return {self.Mean_VIS,self.SD_VIS}
        
    def DarkSpectrum_NIR(self):
        #setup NIR spectrometer
        ts.set_msm('NIR')
        cur_msm = ts.get_msm()
        t_aquisition_NIR = float(self.lineEdit_3.text())
        NumberofScans_NIR = int(self.spinBox_2.text())
        Channel_NIR = int(self.spinBox_4.text())
        Current_NIR = float(self.lineEdit_7.text())
        
        #Warm-up time for lamp
        command = '*PARA:SDEL '+ str(self.lineEdit_8.text())
        ts.send_command("'"+ command +"'")
        
        spec1 = ts.spec_measure(t_aquisition_NIR ,NumberofScans_NIR) # Dark measurement
        spec1=np.array(spec1)
        self.Dark_NIR=spec1
        self.sample2=self.Dark_NIR
        #Spectrometer detector
        Pixnr_NIR = 256#total number of pixel
        Pixelnumber=Pixnr_NIR
        Pixscale = np.array(range(0, Pixelnumber))
        
        NIR_params = ts.get_wavelengths_params()
        
        #Coeffiencts of VIS Spectrometer 
        a=NIR_params[0]
        b=NIR_params[1]
        c=NIR_params[2]
        
        self.wavelengths2 = [(a * Pixscale*Pixscale + b * Pixscale + c) for Pixscale in range(0, Pixelnumber)]
        self.wavelengths2_formatted= [ '%.2f' % elem for elem in self.wavelengths2]
        self.wavelengths2_formatted =list(np.float_(self.wavelengths2_formatted))
        self.Mean_NIR=np.mean(self.sample2)
        self.SD_NIR=np.std(self.sample2)
        # self.textBrowser.append(str(Mean_NIR))
        self.textBrowser.append('Mean: '+ str(round(self.Mean_NIR,2))+' counts')
        # self.textBrowser.append(str(SD_NIR))
        self.textBrowser.append('Standard deviation: '+ str(round(self.SD_NIR,2))+' counts')
        self.get_plot_NIR()
        self.get_temperature()
        return {self.Mean_NIR,self.SD_NIR}
        

    def VIS(self):
        #setup VIS spectrometer
        ts.set_msm('VIS')
        cur_msm = ts.get_msm()

        t_aquisition_VIS = float(self.lineEdit_1.text())
        NumberofScans_VIS = int(self.spinBox_1.text())
        Channel_VIS = int(self.spinBox_3.text())
        Current_VIS = float(self.lineEdit_5.text())
        acquisition_range = self.lineEdit_19.text()
        start_time, end_time, step_size = map(int, acquisition_range.split(":"))

        # Spectrometer detector
        Pixnr_VIS = int(self.comboBox_3.currentText())  # total number of pixel
        Pixelnumber = Pixnr_VIS
        Pixscale = np.array(range(0, Pixelnumber))

        VIS_params = ts.get_wavelengths_params()

        # Coeffiencts of VIS Spectrometer
        a = VIS_params[0]
        b = VIS_params[1]
        c = VIS_params[2]

        self.wavelengths1 = [(a * Pixscale * Pixscale + b * Pixscale + c) for Pixscale in range(0, Pixelnumber)]
        self.wavelengths1_formatted = ['%.2f' % elem for elem in self.wavelengths1]
        self.wavelengths1_formatted = list(np.float_(self.wavelengths1_formatted))

        if not self.checkBox_12.isChecked(): # No range of acquisition times

            # Warm-up time for lamp
            command = '*PARA:SDEL ' + str(self.lineEdit_6.text())
            ts.send_command("'" + command + "'")

            spec1 = ts.spec_measure(t_aquisition_VIS, NumberofScans_VIS)  # Dark measurement
            spec1 = np.array(spec1)
            self.Dark_VIS = spec1
            # switch on light measure data
            spec2 = ts.measure(t_aquisition_VIS, NumberofScans_VIS, Channel_VIS,
                               Current_VIS)  # tint integrationszeit ms , averages, channel, power
            spec2 = np.array(spec2)
            self.Raw_VIS = spec2
            self.sample1 = spec2 - spec1

            self.get_temperature()
            self.get_plot_VIS()

        else:
        # Loop through the range of acquisition times
            for acquisition_time in range(start_time, end_time + 1, step_size):

                #Warm-up time for lamp
                command = '*PARA:SDEL '+ str(self.lineEdit_6.text())
                ts.send_command("'"+ command +"'")

                spec1 = ts.spec_measure(acquisition_time ,NumberofScans_VIS) # Dark measurement
                spec1=np.array(spec1)
                self.Dark_VIS=spec1
                # switch on light measure data
                spec2 = ts.measure(acquisition_time, NumberofScans_VIS, Channel_VIS, Current_VIS)  # tint integrationszeit ms , averages, channel, power
                spec2 = np.array(spec2)
                self.Raw_VIS=spec2
                self.sample1=spec2-spec1

                self.get_temperature()
                self.get_plot_VIS()

        
    def NIR (self): 
        #setup NIR spectrometer
        ts.set_msm('NIR')
        cur_msm = ts.get_msm()
        t_aquisition_NIR = float(self.lineEdit_3.text())
        NumberofScans_NIR = int(self.spinBox_2.text())
        Channel_NIR = int(self.spinBox_4.text())
        Current_NIR = float(self.lineEdit_7.text())
        
        #Warm-up time for lamp
        command = '*PARA:SDEL '+ str(self.lineEdit_8.text())
        ts.send_command("'"+ command +"'")
        
        spec1 = ts.spec_measure(t_aquisition_NIR ,NumberofScans_NIR) # Dark measurement
        spec1=np.array(spec1)
        self.Dark_NIR=spec1
        # switch on light measure data
        spec2 = ts.measure(t_aquisition_NIR, NumberofScans_NIR, Channel_NIR, Current_NIR)  # tint integrationszeit ms , averages, channel, power
        spec2=np.array(spec2)
        self.Raw_NIR=spec2
        self.sample2=spec2-spec1
        #sample1=spec2
        
        #Spectrometer detector
        Pixnr_NIR = 256#total number of pixel
        Pixelnumber=Pixnr_NIR
        Pixscale = np.array(range(0, Pixelnumber))
        
        NIR_params = ts.get_wavelengths_params()
        
        #Coeffiencts of VIS Spectrometer 
        a=NIR_params[0]
        b=NIR_params[1]
        c=NIR_params[2]
        
        self.wavelengths2 = [(a * Pixscale*Pixscale + b * Pixscale + c) for Pixscale in range(0, Pixelnumber)]
        self.wavelengths2_formatted= [ '%.2f' % elem for elem in self.wavelengths2]
        self.wavelengths2_formatted =list(np.float_(self.wavelengths2_formatted))
        self.get_temperature()
        self.get_plot_NIR()
        
        
    def VISNIR(self):
        #setup VIS spectrometer
        ts.set_msm('VIS')
        cur_msm = ts.get_msm()
        t_aquisition_VIS = float(self.lineEdit_1.text())
        NumberofScans_VIS = int(self.spinBox_1.text())
        Channel_VIS = int(self.spinBox_3.text())
        Current_VIS = float(self.lineEdit_5.text())
        
        #Warm-up time for lamp
        command = '*PARA:SDEL '+ str(self.lineEdit_6.text())
        ts.send_command("'"+ command +"'")
        
        spec1 = ts.spec_measure(t_aquisition_VIS ,NumberofScans_VIS) # Dark measurement
        spec1=np.array(spec1)
        self.Dark_VIS=spec1
        # switch on light measure data
        spec2 = ts.measure(t_aquisition_VIS, NumberofScans_VIS, Channel_VIS, Current_VIS)  # tint integrationszeit ms , averages, channel, power
        spec2=np.array(spec2)
        self.Raw_VIS=spec2
        self.sample1=spec2-spec1
        #sample1=spec2
        
        #Spectrometer detector
        Pixnr_VIS = int(self.comboBox_3.currentText()) #total number of pixel
        Pixelnumber=Pixnr_VIS
        Pixscale = np.array(range(0, Pixelnumber))
        
        VIS_params = ts.get_wavelengths_params()
        
        #Coeffiencts of VIS Spectrometer 
        a=VIS_params[0]
        b=VIS_params[1]
        c=VIS_params[2]
        
        self.wavelengths1 = [(a * Pixscale*Pixscale + b * Pixscale + c) for Pixscale in range(0, Pixelnumber)]
        self.wavelengths1_formatted= [ '%.2f' % elem for elem in self.wavelengths1]
        self.wavelengths1_formatted =list(np.float_(self.wavelengths1_formatted))
        
        #setup NIR spectrometer
        ts.set_msm('NIR')
        cur_msm = ts.get_msm()
        t_aquisition_NIR = float(self.lineEdit_3.text())
        NumberofScans_NIR = int(self.spinBox_2.text())
        Channel_NIR = int(self.spinBox_4.text())
        Current_NIR = float(self.lineEdit_7.text())
        
        #Warm-up time for lamp
        command = '*PARA:SDEL '+ str(self.lineEdit_8.text())
        ts.send_command("'"+ command +"'")
        
        spec1 = ts.spec_measure(t_aquisition_NIR ,NumberofScans_NIR) # Dark measurement
        spec1=np.array(spec1)
        self.Dark_NIR=spec1
        # switch on light measure data
        spec2 = ts.measure(t_aquisition_NIR, NumberofScans_NIR, Channel_NIR, Current_NIR)  # tint integrationszeit ms , averages, channel, power
        spec2=np.array(spec2)
        self.Raw_NIR=spec2
        self.sample2=spec2-spec1
        #sample1=spec2
        
        #Spectrometer detector
        Pixnr_NIR = 256#total number of pixel
        Pixelnumber=Pixnr_NIR
        Pixscale = np.array(range(0, Pixelnumber))

        NIR_params = ts.get_wavelengths_params()
        
        #Coeffiencts of VIS Spectrometer 
        a=NIR_params[0]
        b=NIR_params[1]
        c=NIR_params[2]

        self.wavelengths2 = [(a * Pixscale*Pixscale + b * Pixscale + c) for Pixscale in range(0, Pixelnumber)]
        self.wavelengths2_formatted= [ '%.2f' % elem for elem in self.wavelengths2]
        self.wavelengths2_formatted =list(np.float_(self.wavelengths2_formatted))
        self.get_temperature()
        self.get_plot_VIS()
        self.get_plot_NIR()

    # Cycles through all possible modes
    def cycleAll(self):
        # lists to hold the specta obtained from the measurements
        self.FL365_lst = []  # first column dark, rest of columns are samples
        self.FL405_lst = []  # first column dark, rest of columns are samples
        self.vis_lst = []  # first column white, then dark, then rest are samples
        self.nir_lst = []  # first column white, then dark, then rest are samples

        # setup integration times
        self.t_aquisition_VIS = float(self.lineEdit_11.text())
        self.t_aquisition_NIR = float(self.lineEdit_12.text())
        self.t_aquisition_FL365 = float(self.lineEdit_13.text())
        self.t_aquisition_FL405 = float(self.lineEdit_14.text())

        # setup current
        self.Current_VIS = float(self.lineEdit_15.text())
        self.Current_NIR = float(self.lineEdit_16.text())
        self.Current_FL365 = float(self.lineEdit_17.text())
        self.Current_FL405 = float(self.lineEdit_18.text())

        # setup source channels
        reflect_source = 1
        FL365_source = 2
        FL405_source = 4

        # setup VIS spectrometer parameters
        self.NumberofScans_VIS = int(self.spinBox_1.text())

        # setup NIR spectrometer parameters
        self.NumberofScans_NIR = int(self.spinBox_2.text())

        # SETUP PLOTTING PARAMETERS
        # Spectrometer detector
        ts.set_msm('VIS')
        Pixnr_VIS = int(self.comboBox_3.currentText())  # total number of pixel
        Pixelnumber = Pixnr_VIS
        Pixscale = np.array(range(0, Pixelnumber))

        VIS_params = ts.get_wavelengths_params()

        # Coeffiencts of VIS Spectrometer
        a = VIS_params[0]
        b = VIS_params[1]
        c = VIS_params[2]

        self.wavelengths1 = [(a * Pixscale * Pixscale + b * Pixscale + c) for Pixscale in range(0, Pixelnumber)]
        self.wavelengths1_formatted = ['%.2f' % elem for elem in self.wavelengths1]
        self.wavelengths1_formatted = list(np.float_(self.wavelengths1_formatted))


        ts.set_msm("NIR")
        # Spectrometer detector
        Pixnr_NIR = 256  # total number of pixel
        Pixelnumber = Pixnr_NIR
        Pixscale = np.array(range(0, Pixelnumber))

        NIR_params = ts.get_wavelengths_params()

        # Coeffiencts of NIR Spectrometer
        a = NIR_params[0]
        b = NIR_params[1]
        c = NIR_params[2]

        self.wavelengths2 = [(a * Pixscale * Pixscale + b * Pixscale + c) for Pixscale in range(0, Pixelnumber)]
        self.wavelengths2_formatted = ['%.2f' % elem for elem in self.wavelengths2]
        self.wavelengths2_formatted = list(np.float_(self.wavelengths2_formatted))

        # START DARK MEASUREMENTS
        # Warm-up time command
        self.warmup_time = str(self.lineEdit_6.text())
        warmup_command = '*PARA:SDEL ' + f'{self.warmup_time}'

        QMessageBox.warning(self, "Dark Measurement Adjustments", "Place the dark cap on the device for dark measurements")

        #VIS DARK
        ts.set_msm('VIS')
        dark_vis = ts.spec_measure(self.t_aquisition_VIS, self.NumberofScans_VIS)  # Dark measurement
        self.dark_vis = np.array(dark_vis)
        self.vis_lst.append(pd.Series(self.wavelengths1_formatted, name='Wavelength [nm]'))
        self.vis_lst.append(pd.Series(self.dark_vis, name='Dark'))

        #FL365 Dark
        dark_FL365 = ts.spec_measure(self.t_aquisition_FL365, self.NumberofScans_VIS)  # Dark measurement
        self.dark_FL365 = np.array(dark_FL365)
        self.FL365_lst.append(pd.Series(self.wavelengths1_formatted, name='Wavelength [nm]'))
        self.FL365_lst.append(pd.Series(self.dark_FL365, name='Dark'))

        #FL405 Dark
        dark_FL405 = ts.spec_measure(self.t_aquisition_FL405, self.NumberofScans_VIS)  # Dark measurement
        self.dark_FL405 = np.array(dark_FL405)
        self.FL405_lst.append(pd.Series(self.wavelengths1_formatted, name='Wavelength [nm]'))
        self.FL405_lst.append(pd.Series(self.dark_FL405, name='Dark'))

        #NIR Dark
        ts.set_msm('NIR')
        dark_nir = ts.spec_measure(self.t_aquisition_NIR, self.NumberofScans_NIR)  # Dark measurement
        self.dark_nir = np.array(dark_nir)
        self.nir_lst.append(pd.Series(self.wavelengths2_formatted, name='Wavelength [nm]'))
        self.nir_lst.append(pd.Series(self.dark_nir, name='Dark'))

        self.get_darkplot_cycleAll()

        # START WHITE MEASUREMENTS
        # switch on light measure white reference
        QMessageBox.warning(self, "White Measurement Adjustments",
                            "Place the white reference on the device for white measurements")

        # VIS White Reference
        ts.set_msm('VIS')
        ts.send_command("'" + warmup_command + "'") # warmup
        white_vis = ts.measure(self.t_aquisition_VIS, self.NumberofScans_VIS, reflect_source, self.Current_VIS)
        self.white_vis = np.array(white_vis)
        self.vis_lst.append(pd.Series(self.white_vis, name='White'))

        # NIR White Reference
        ts.set_msm('NIR')
        ts.send_command("'" + warmup_command + "'") # warmup
        white_nir = ts.measure(self.t_aquisition_NIR, self.NumberofScans_NIR, reflect_source, self.Current_NIR)
        self.white_nir = np.array(white_nir)
        self.nir_lst.append(pd.Series(self.white_nir, name='White'))

        self.get_whiteplot_cycleAll()

        # START SAMPLE MEASUREMENTS
        # VIS Sample
        QMessageBox.warning(self, "Sample Measurement Adjustments",
                            "Remove the cap from the device for sample measurements")

        # Refresh the plots so that you can see the calibrated values
        self.refresh()

        i = 1
        cont_meas = True # Take the first sample measurement. The user will be asked to take more at the end
        while(cont_meas):
            ts.set_msm('VIS')
            ts.send_command("'" + warmup_command + "'") # warmup
            raw_vis = ts.measure(self.t_aquisition_VIS, self.NumberofScans_VIS, reflect_source, self.Current_VIS)  # tint integrationszeit ms , averages, channel, power
            self.raw_vis = np.array(raw_vis)
            self.sample_vis = (self.raw_vis - self.dark_vis) / (self.white_vis - self.dark_vis)
            self.vis_lst.append(pd.Series(self.raw_vis, name='Raw ' + f'{i}'))
            self.vis_lst.append(pd.Series(self.sample_vis, name='Calibrated ' + f'{i}'))

            # FL365 Sample
            ts.send_command("'" + warmup_command + "'") # warmup
            raw_FL365 = ts.measure(self.t_aquisition_FL365, self.NumberofScans_VIS, FL365_source,
                                 self.Current_FL365)  # tint integrationszeit ms , averages, channel, power
            self.raw_FL365 = np.array(raw_FL365)
            self.sample_FL365 = self.raw_FL365 - self.dark_FL365
            self.FL365_lst.append(pd.Series(self.raw_FL365, name='Raw ' + f'{i}'))
            self.FL365_lst.append(pd.Series(self.sample_FL365, name='Calibrated ' + f'{i}'))

            # FL405 Sample
            ts.send_command("'" + warmup_command + "'") # warmup
            raw_FL405 = ts.measure(self.t_aquisition_FL405, self.NumberofScans_VIS, FL405_source,
                                   self.Current_FL405)  # tint integrationszeit ms , averages, channel, power
            self.raw_FL405 = np.array(raw_FL405)
            self.sample_FL405 = self.raw_FL405 - self.dark_FL405
            self.FL405_lst.append(pd.Series(self.raw_FL405, name='Raw ' + f'{i}'))
            self.FL405_lst.append(pd.Series(self.sample_FL405, name='Calibrated ' + f'{i}'))

            #NIR
            ts.set_msm('NIR')
            ts.send_command("'" + warmup_command + "'") # warmup
            raw_nir = ts.measure(self.t_aquisition_NIR, self.NumberofScans_NIR, reflect_source, self.Current_NIR)  # tint integrationszeit ms , averages, channel, power
            self.raw_nir = np.array(raw_nir)
            self.sample_nir = (self.raw_nir - self.dark_nir) / (self.white_nir - self.dark_nir)
            self.nir_lst.append(pd.Series(self.raw_nir, name='Raw ' + f'{i}'))
            self.nir_lst.append(pd.Series(self.sample_nir, name='Calibrated ' + f'{i}'))

            self.get_temperature()
            self.get_plot_cycleAll()

            # Ask if the user wants to tke more measurements
            cont_meas = self.ask_more_measurements()
            i += 1

        # Write to excel once finished with measurements
        self.cycleALLwriteToExcel()

    def get_plot_cycleAll(self):
        #refl_legend_labels = ["Dark", "White", "1 run", "2 run", "3 run", "4 run", "5 run", "6 run", "7 run", "8 run"]
        #fl_legend_labels = ["Dark", "1 run", "2 run", "3 run", "4 run", "5 run", "6 run", "7 run", "8 run"]
        legend_labels = ["1 run", "2 run", "3 run", "4 run", "5 run", "6 run", "7 run", "8 run", "9 run", "10 run"]

        #Visible
        self.MplWidget_1.canvas.axes.plot(self.wavelengths1, self.sample_vis)
        self.MplWidget_1.canvas.axes.legend(labels=legend_labels, loc='upper right')
        self.MplWidget_1.canvas.axes.set_xlim(450, 950)  # Set x-axis range
        self.MplWidget_1.canvas.axes.set_xlabel("Wavelength [nm]")
        self.MplWidget_1.canvas.axes.set_ylabel('Intensity (counts)')
        self.MplWidget_1.canvas.draw()

        #NIR
        self.MplWidget_2.canvas.axes.plot(self.wavelengths2, self.sample_nir)
        self.MplWidget_2.canvas.axes.legend(labels=legend_labels, loc='upper right')
        self.MplWidget_2.canvas.axes.set_xlim(950, 1600)  # Set x-axis range
        self.MplWidget_2.canvas.axes.set_xlabel("Wavelength [nm]")
        self.MplWidget_2.canvas.axes.set_ylabel('Intensity (counts)')
        self.MplWidget_2.canvas.draw()

        # FL 365
        self.MplWidget_3.canvas.axes.plot(self.wavelengths1, self.sample_FL365)
        self.MplWidget_3.canvas.axes.legend(labels=legend_labels, loc='upper right')
        self.MplWidget_3.canvas.axes.set_xlim(450, 950)  # Set x-axis range
        self.MplWidget_3.canvas.axes.set_xlabel("Wavelength [nm]")
        self.MplWidget_3.canvas.axes.set_ylabel('Intensity (counts)')
        self.MplWidget_3.canvas.draw()

        # FL 405
        self.MplWidget_4.canvas.axes.plot(self.wavelengths1, self.sample_FL405)
        self.MplWidget_4.canvas.axes.legend(labels=legend_labels, loc='upper right')
        self.MplWidget_4.canvas.axes.set_xlim(450, 950)  # Set x-axis range
        self.MplWidget_4.canvas.axes.set_xlabel("Wavelength [nm]")
        self.MplWidget_4.canvas.axes.set_ylabel('Intensity (counts)')
        self.MplWidget_4.canvas.draw()

    def get_darkplot_cycleAll(self):
        legend_labels = ["Dark"]
        # Visible
        self.MplWidget_1.canvas.axes.plot(self.wavelengths1, self.dark_vis)
        self.MplWidget_1.canvas.axes.legend(labels=legend_labels, loc='upper right')
        self.MplWidget_1.canvas.axes.set_xlim(450, 950)  # Set x-axis range
        self.MplWidget_1.canvas.axes.set_xlabel("Wavelength [nm]")
        self.MplWidget_1.canvas.axes.set_ylabel('Intensity (counts)')
        self.MplWidget_1.canvas.draw()

        # NIR
        self.MplWidget_2.canvas.axes.plot(self.wavelengths2, self.dark_nir)
        self.MplWidget_2.canvas.axes.legend(labels=legend_labels, loc='upper right')
        self.MplWidget_2.canvas.axes.set_xlim(950, 1600)  # Set x-axis range
        self.MplWidget_2.canvas.axes.set_xlabel("Wavelength [nm]")
        self.MplWidget_2.canvas.axes.set_ylabel('Intensity (counts)')
        self.MplWidget_2.canvas.draw()

        # FL 365
        self.MplWidget_3.canvas.axes.plot(self.wavelengths1, self.dark_FL365)
        self.MplWidget_3.canvas.axes.legend(labels=legend_labels, loc='upper right')
        self.MplWidget_3.canvas.axes.set_xlim(450, 950)  # Set x-axis range
        self.MplWidget_3.canvas.axes.set_xlabel("Wavelength [nm]")
        self.MplWidget_3.canvas.axes.set_ylabel('Intensity (counts)')
        self.MplWidget_3.canvas.draw()

        # FL 405
        self.MplWidget_4.canvas.axes.plot(self.wavelengths1, self.dark_FL405)
        self.MplWidget_4.canvas.axes.legend(labels=legend_labels, loc='upper right')
        self.MplWidget_4.canvas.axes.set_xlim(450, 950)  # Set x-axis range
        self.MplWidget_4.canvas.axes.set_xlabel("Wavelength [nm]")
        self.MplWidget_4.canvas.axes.set_ylabel('Intensity (counts)')
        self.MplWidget_4.canvas.draw()

    def get_whiteplot_cycleAll(self):
        legend_labels = ["Dark", "White"]
        #Visible
        self.MplWidget_1.canvas.axes.plot(self.wavelengths1, self.white_vis)
        self.MplWidget_1.canvas.axes.legend(labels=legend_labels, loc='upper right')
        self.MplWidget_1.canvas.axes.set_xlim(450, 950)  # Set x-axis range
        self.MplWidget_1.canvas.axes.set_xlabel("Wavelength [nm]")
        self.MplWidget_1.canvas.axes.set_ylabel('Intensity (counts)')
        self.MplWidget_1.canvas.draw()

        #NIR
        self.MplWidget_2.canvas.axes.plot(self.wavelengths2, self.white_nir)
        self.MplWidget_2.canvas.axes.legend(labels=legend_labels, loc='upper right')
        self.MplWidget_2.canvas.axes.set_xlim(950, 1600)  # Set x-axis range
        self.MplWidget_2.canvas.axes.set_xlabel("Wavelength [nm]")
        self.MplWidget_2.canvas.axes.set_ylabel('Intensity (counts)')
        self.MplWidget_2.canvas.draw()


    
    def get_plot_VIS(self):
        legend_labels=["1 run","2 run","3 run", "4 run", "5 run", "6 run", "7 run", "8 run", "9 run", "10 run"]
        self.MplWidget_1.canvas.axes.plot(self.wavelengths1,self.sample1)
        #self.MplWidget_1.canvas.axes.legend(('Data'),loc='upper right')
        self.MplWidget_1.canvas.axes.legend(labels = legend_labels,loc='upper right')
        #self.MplWidget_1.canvas.axes.text(200,100,'VIS Spectrum',fontsize=12)
        self.MplWidget_1.canvas.axes.set_xlabel("Wavelength [nm]")
        self.MplWidget_1.canvas.axes.set_ylabel('Intensity (counts)')
        #self.MplWidget_1.canvas.axes.set_xlim(300, 1100)
        self.MplWidget_1.canvas.draw()
        
    def get_plot_NIR(self): 
        legend_labels=["1 run","2 run","3 run", "4 run", "5 run", "6 run", "7 run", "8 run", "9 run", "10 run"]
        self.MplWidget_2.canvas.axes.plot(self.wavelengths2,self.sample2)
        #self.MplWidget_2.canvas.axes.legend(('Data'),loc='upper right')
        self.MplWidget_2.canvas.axes.legend(labels = legend_labels,loc='upper right')
        #self.MplWidget_2.canvas.axes.set_title('NIR Spectrum')
        self.MplWidget_2.canvas.axes.set_xlabel("Wavelength [nm]")
        self.MplWidget_2.canvas.axes.set_ylabel('Intensity (counts)')
        #self.MplWidget_2.canvas.axes.set_ylim(-300, 6000)
        self.MplWidget_2.canvas.draw()
    
    def get_plot1(self, xdata, ydata):
        #self.MplWidget_1.canvas.axes.plot(self.WL,self.INTEN)
        self.MplWidget_1.canvas.axes.plot(xdata,ydata)
        self.MplWidget_1.canvas.draw()
    def get_plot2(self, xdata, ydata):
         #self.MplWidget_2.canvas.axes.plot(self.WL,self.INTEN)
         self.MplWidget_2.canvas.axes.plot(xdata,ydata)
         self.MplWidget_2.canvas.draw()

    def ask_more_measurements(self):
        msg = QMessageBox()
        msg.setWindowTitle("More Measurements")
        msg.setText("Would you like to take more measurements?")
        msg.setIcon(QMessageBox.Question)
        msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)

        # Directly execute the message box and return the boolean result
        return msg.exec_() == QMessageBox.Yes
        
    def get_temperature(self):   
        #Temperature reading
        T_head, T_board = ts.get_temperatures()
        T_head=T_head
        T_board=T_board
        self.lcdNumber_1.display('{:.02f}'.format(T_head))
        self.lcdNumber_2.display('{:.02f}'.format(T_board))
        
        if self.radioButton_1.isChecked():
            ts.set_msm('VIS')
            #Set Temp correction to Zero for VIS
            ts.send_command('*PARA:TEMPC 0')
            T_dectector_VIS=ts.send_command2('*MEAS:TEMPE')
            T_dectector_VIS=(float(T_dectector_VIS))
            self.lcdNumber_3.display(T_dectector_VIS)
        if self.radioButton_2.isChecked():
            ts.set_msm('NIR')
            #Set Temp correction to Zero for NIR
            ts.send_command('*PARA:TEMPC 0')
            T_dectector_NIR=ts.send_command2('*MEAS:TEMPE')
            T_dectector_NIR=(float(T_dectector_NIR))
            self.lcdNumber_4.display(T_dectector_NIR)
            #print(T_head, T_board, T_dectector_VIS, T_dectector_NIR)
            #self.lcdNumber_2.display('{:.02f}'.format(value))
        if self.radioButton_3.isChecked():
            ts.set_msm('VIS')
            #Set Temp correction to Zero for VIS
            ts.send_command('*PARA:TEMPC 0')
            T_dectector_VIS=ts.send_command2('*MEAS:TEMPE')
            T_dectector_VIS=float((T_dectector_VIS))
            self.lcdNumber_3.display(T_dectector_VIS)
            ts.set_msm('NIR')
            #Set Temp correction to Zero for NIR
            ts.send_command('*PARA:TEMPC 0')
            T_dectector_NIR=ts.send_command2('*MEAS:TEMPE')
            T_dectector_NIR=(float(T_dectector_NIR))
            self.lcdNumber_4.display(T_dectector_NIR)
        
    def refresh(self):
        self.MplWidget_1.canvas.axes.clear()
        self.MplWidget_1.canvas.draw()
        self.MplWidget_2.canvas.axes.clear()
        self.MplWidget_2.canvas.draw()
        self.MplWidget_3.canvas.axes.clear()
        self.MplWidget_3.canvas.draw()
        self.MplWidget_4.canvas.axes.clear()
        self.MplWidget_4.canvas.draw()
    
    def select(self):
        if self.radioButton_1.isChecked():
            self.textBrowser.setText('VIS spectrometer selected')
            self.VIS()
            
        if self.radioButton_2.isChecked():
            self.textBrowser.setText('NIR spectrometer selected')
            self.NIR()
            
        if self.radioButton_3.isChecked():
            self.textBrowser.setText('VIS & NIR spectrometer selected')
            self.VISNIR()

        if self.buttonALL.isChecked():
            self.textBrowser.setText('Cycle through all sources and spectrometers (References and sample measurements)')
            self.cycleAll()
                    
    def get_Directory(self):
        response = QFileDialog.getExistingDirectory(self,caption='Select a folder')
        #self.lineEdit.setText(str(response)) 
        self.directory=response.replace('/','\\')
        #print(self.directory)

    def set_pause(self):
        self.Tpause = float(self.lineEdit_9.text())
        self.textBrowser.setText('Delay time:'+ self.lineEdit_9.text()+'s')
        time.sleep(self.Tpause)
               
    def save_data(self):
        time=datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
        header1= ['Sample Measurement']
        header2= ['Wavelengh_VIS (nm)', 'Intensity_VIS', 'Wavelengh_NIR (nm)', 'Intensity_NIR']
        filename = str(time)
        file_path = os.path.join(self.directory, filename)
        new_list = zip(self.wavelengths1_formatted, self.sample1, self.wavelengths2_formatted, self.sample2)
        #new_list.replace(".", ",")
        with open(file_path + ".csv", 'w') as csvfile:
            filewriter = csv.writer(csvfile, delimiter=',')
            filewriter.writerow(header1)
            filewriter.writerow(header2)
            filewriter.writerows(new_list)
            
    def save_settings(self):
        time=datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
        if self.radioButton_1.isChecked():
            MessParam = {'Date & Time': time}
            if self.checkBox_1.isChecked():
                MessParam['Sample name'] = self.textEdit_1.toPlainText()
            if self.checkBox_2.isChecked():
                MessParam['Channel VIS'] = int(self.spinBox_3.text())
            if self.checkBox_3.isChecked():
                MessParam['Lightsource current VIS [mA]'] = float(self.lineEdit_5.text())
            if self.checkBox_4.isChecked():
                MessParam['Number of scans VIS'] = int(self.spinBox_1.text())
            if self.checkBox_5.isChecked():
                MessParam['Acquisition time [ms] VIS'] = float(self.lineEdit_1.text())
                     
            PAR_Data = pd.DataFrame([MessParam]).transpose()
            PAR_Data = PAR_Data.reset_index()
            filename = str(time)
            file_path = os.path.join(self.directory, filename)
            #print(file_path)
            save_path = file_path + ".xlsx"
            #print(save_path)
            with pd.ExcelWriter(save_path, engine='xlsxwriter') as writer:
                PAR_Data.to_excel(writer, sheet_name='Info', index=True)
                writer.sheets['Info'].set_column(1,1, 40)
                writer.sheets['Info'].set_column(2,2, 20)
                
        if self.radioButton_2.isChecked():
            MessParam = {'Date & Time': time}
            if self.checkBox_1.isChecked():
                MessParam['Sample name'] = self.textEdit_1.toPlainText()
            if self.checkBox_6.isChecked():
                MessParam['Channel NIR'] = int(self.spinBox_4.text())
            if self.checkBox_7.isChecked():
                MessParam['Lightsource current NIR [mA]'] = float(self.lineEdit_7.text())
            if self.checkBox_8.isChecked():
                MessParam['Number of scans NIR'] = int(self.spinBox_2.text())
            if self.checkBox_9.isChecked():
                MessParam['Acquisition time [ms] NIR'] = float(self.lineEdit_3.text())
                
            PAR_Data = pd.DataFrame([MessParam]).transpose()
            PAR_Data = PAR_Data.reset_index()
            filename = str(time)
            file_path = os.path.join(self.directory, filename)
            #print(file_path)
            save_path = file_path + ".xlsx"
            #print(save_path)
            with pd.ExcelWriter(save_path, engine='xlsxwriter') as writer:
                PAR_Data.to_excel(writer, sheet_name='Info', index=True)
                writer.sheets['Info'].set_column(1,1, 40)
                writer.sheets['Info'].set_column(2,2, 20)
                
        if self.radioButton_3.isChecked():
            MessParam = {'Date & Time': time}
            if self.checkBox_1.isChecked():
                MessParam['Sample name'] = self.textEdit_1.toPlainText()
            if self.checkBox_2.isChecked():
                MessParam['Channel VIS'] = int(self.spinBox_3.text())
            if self.checkBox_3.isChecked():
                MessParam['Lightsource current VIS [mA]'] = float(self.lineEdit_5.text())
            if self.checkBox_4.isChecked():
                MessParam['Number of scans VIS'] = int(self.spinBox_1.text())
            if self.checkBox_5.isChecked():
                MessParam['Acquisition time [ms] VIS'] = float(self.lineEdit_1.text())
            if self.checkBox_6.isChecked():
                MessParam['Channel NIR'] = int(self.spinBox_4.text())
            if self.checkBox_7.isChecked():
                MessParam['Lightsource current NIR [mA]'] = float(self.lineEdit_7.text())
            if self.checkBox_8.isChecked():
                MessParam['Number of scans NIR'] = int(self.spinBox_2.text())
            if self.checkBox_9.isChecked():
                MessParam['Acquisition time [ms] NIR'] = float(self.lineEdit_3.text())

            if self.buttonALL.isChecked():
                MessParam = {'Date & Time': time}
                if self.checkBox_1.isChecked():
                    MessParam['Sample name'] = self.textEdit_1.toPlainText()
                if self.checkBox_2.isChecked():
                    MessParam['Channel VIS'] = int(self.spinBox_3.text())
                if self.checkBox_3.isChecked():
                    MessParam['Lightsource current VIS [mA]'] = float(self.lineEdit_5.text())
                if self.checkBox_4.isChecked():
                    MessParam['Number of scans VIS'] = int(self.spinBox_1.text())
                if self.checkBox_5.isChecked():
                    MessParam['Acquisition time [ms] VIS'] = float(self.lineEdit_1.text())
                if self.checkBox_6.isChecked():
                    MessParam['Channel NIR'] = int(self.spinBox_4.text())
                if self.checkBox_7.isChecked():
                    MessParam['Lightsource current NIR [mA]'] = float(self.lineEdit_7.text())
                if self.checkBox_8.isChecked():
                    MessParam['Number of scans NIR'] = int(self.spinBox_2.text())
                if self.checkBox_9.isChecked():
                    MessParam['Acquisition time [ms] NIR'] = float(self.lineEdit_3.text())

            PAR_Data = pd.DataFrame([MessParam]).transpose()
            PAR_Data = PAR_Data.reset_index()
            filename = str(time)
            file_path = os.path.join(self.directory, filename)
            #print(file_path)
            save_path = file_path + ".xlsx"
            #print(save_path)
            with pd.ExcelWriter(save_path, engine='xlsxwriter') as writer:
                PAR_Data.to_excel(writer, sheet_name='Info', index=True)
                writer.sheets['Info'].set_column(1,1, 40)
                writer.sheets['Info'].set_column(2,2, 20)


    def cycleALLwriteToExcel(self):
        path = str(Path.home() / 'Documents')
        output_path = path + '\Handheld_PIPA_Results'

        print(output_path)

        if not os.path.exists(output_path):
            os.makedirs(output_path)

        timestr = time.strftime("%Y%m%d-%H%M%S")
        SampleName = self.textEdit_1.toPlainText()
        savepath = output_path + "/" + timestr + f'_{SampleName}.xlsx'


        # Save to xlsx-file
        dfxlstVIS = pd.DataFrame(self.vis_lst).transpose()
        dfxlstNIR = pd.DataFrame(self.nir_lst).transpose()
        dfxlstFL365 = pd.DataFrame(self.FL365_lst).transpose()
        dfxlstFL405 = pd.DataFrame(self.FL405_lst).transpose()

        MessParam = {'File Path ': savepath,
                     'Date & Time': timestr,
                     'Sample': SampleName,
                     'Warmup time for each Cycle [ms]': self.warmup_time,
                     'vis IT [ms]': self.t_aquisition_VIS,
                     'nir IT [ms]': self.t_aquisition_NIR,
                     'FL 365 IT [ms]': self.t_aquisition_FL365,
                     'FL 405 IT [ms]': self.t_aquisition_FL405,
                     #'Reference Reflectance Value': refl_val,
                     #'vismsm SN': vismsmSN,
                     'VIS Averaging': self.NumberofScans_VIS,
                     #'vismsm Gain': Gain_vis,
                     #'vismsm PDAGain': PDAGain_vis,
                     #'nirmsm SN': nirmsmSN,
                     'nirmsm Averaging': self.NumberofScans_NIR}
                     #'nirmsm Gain': Gain_nir,
                     #'nirmsm PDAGain': PDAGain_nir,
                     #'nir Offset': nir_Offset,
                     #'vis Offset': vis_Offset}

        # Wrap dictionary into the list to avoid error
        dfxlstPAR = pd.DataFrame([MessParam]).transpose()
        dfxlstPAR = dfxlstPAR.reset_index()

        # Context manager to create several sheets in the excel-file
        with pd.ExcelWriter(savepath) as writer:
            dfxlstPAR.to_excel(writer, sheet_name='Info', index=False)
            dfxlstVIS.to_excel(writer, sheet_name='VIS', index=False)
            dfxlstNIR.to_excel(writer, sheet_name='NIR', index=False)
            dfxlstFL365.to_excel(writer, sheet_name='FL365', index=False)
            dfxlstFL405.to_excel(writer, sheet_name='FL405', index=False)

            writer.sheets['VIS'].set_column(1, 511, 20)
            writer.sheets['NIR'].set_column(1, 255, 20)
            writer.sheets['FL365'].set_column(1, 255, 20)
            writer.sheets['FL405'].set_column(1, 255, 20)

            writer.sheets['Info'].set_column(1, 1, 40)
            writer.sheets['Info'].set_column(2, 2, 20)

        print("exported data")
    
            
    def writeToExcel(self):
        #VIS
        if self.radioButton_1.isChecked():
            list_VIS = zip(self.wavelengths1_formatted, self.Dark_VIS, self.Raw_VIS, self.sample1)
            VIS_Data = pd.DataFrame(list_VIS)
            time=datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
            
            MessParam = {'Date & Time': time}
            if self.checkBox_1.isChecked():
                MessParam['Sample name'] = self.textEdit_1.toPlainText()
            if self.checkBox_2.isChecked():
                MessParam['Channel VIS'] = int(self.spinBox_3.text())
            if self.checkBox_3.isChecked():
                MessParam['Lightsource current VIS [mA]'] = float(self.lineEdit_5.text())
            if self.checkBox_4.isChecked():
                MessParam['Number of scans VIS'] = int(self.spinBox_1.text())
            if self.checkBox_5.isChecked():
                MessParam['Acquisition time [ms] VIS'] = float(self.lineEdit_1.text())
            if self.checkBox_6.isChecked():
                MessParam['Scan delay time VIS [ms]'] = float(self.lineEdit_6.text())
            
            
            PAR_Data = pd.DataFrame([MessParam]).transpose()
            PAR_Data = PAR_Data.reset_index()
            
            col_names1= ['Wavelengh_VIS (nm)', 'Dark measurement_VIS', 'Raw_VIS','Intensity_VIS']
            VIS_Data.columns = col_names1
            if os.path.isdir(self.directory) == True:
                pass
            elif os.path.isdir(self.directory) == False:
                self.directory=os.getcwd().replace('/','\\')
            #print(self.directory)
            filename = str(time)
            #print(filename)
            file_path = os.path.join(self.directory, filename)
            #print(file_path)
            save_path = file_path + ".xlsx"
            print(save_path)
        
            with pd.ExcelWriter(save_path, engine='xlsxwriter') as writer:
                PAR_Data.to_excel(writer, sheet_name='Info', index=True)
                VIS_Data.to_excel(writer, sheet_name='VIS', index=True)
                
                writer.sheets['VIS'].set_column(1, 511, 20)
                writer.sheets['Info'].set_column(1,1, 40)
                writer.sheets['Info'].set_column(2,2, 20)
            
        #NIR    
        if self.radioButton_2.isChecked():
            list_NIR = zip(self.wavelengths2_formatted, self.Dark_NIR, self.Raw_NIR, self.sample2)
            NIR_Data = pd.DataFrame(list_NIR)
            time=datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
            
            MessParam = {'Date & Time': time}
            if self.checkBox_1.isChecked():
                MessParam['Sample name'] = self.textEdit_1.toPlainText()
            if self.checkBox_7.isChecked():
                MessParam['Channel NIR'] = int(self.spinBox_4.text())
            if self.checkBox_8.isChecked():
                MessParam['Lightsource current NIR [mA]'] = float(self.lineEdit_7.text())
            if self.checkBox_9.isChecked():
                MessParam['Number of scans NIR'] = int(self.spinBox_2.text())
            if self.checkBox_10.isChecked():
                MessParam['Acquisition time [ms] NIR'] = float(self.lineEdit_3.text())
            if self.checkBox_11.isChecked():
                MessParam['Scan delay time NIR [ms]'] = float(self.lineEdit_8.text())
            
            PAR_Data = pd.DataFrame([MessParam]).transpose()
            PAR_Data = PAR_Data.reset_index()
            
            col_names2= ['Wavelengh_NIR (nm)', 'Dark measurement_NIR', 'Raw_NIR','Intensity_NIR']
            NIR_Data.columns = col_names2
            if os.path.isdir(self.directory) == True:
                pass
            elif os.path.isdir(self.directory) == False:
                self.directory=os.getcwd().replace('/','\\')
            #print(self.directory)
            filename = str(time)
            #print(filename)
            file_path = os.path.join(self.directory, filename)
            #print(file_path)
            save_path = file_path + ".xlsx"
            print(save_path)
        
            with pd.ExcelWriter(save_path, engine='xlsxwriter') as writer:
                PAR_Data.to_excel(writer, sheet_name='Info', index=True)
                NIR_Data.to_excel(writer, sheet_name='NIR', index=True)
                
                writer.sheets['NIR'].set_column(1, 255, 20)
                writer.sheets['Info'].set_column(1,1, 40)
                writer.sheets['Info'].set_column(2,2, 20)
        
        #VIS+NIR
        if self.radioButton_3.isChecked():
            list_VIS = zip(self.wavelengths1_formatted, self.Dark_VIS, self.Raw_VIS, self.sample1)
            list_NIR = zip(self.wavelengths2_formatted, self.Dark_NIR, self.Raw_NIR, self.sample2)
         
            VIS_Data = pd.DataFrame(list_VIS)
            NIR_Data = pd.DataFrame(list_NIR)
            
            # VIS_Data.insert(0,'Wavelength [nm]')
            # NIR_Data.insert(0,'Wavelength [nm]')
            time=datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
            
            MessParam = {'Date & Time': time}
            if self.checkBox_1.isChecked():
                MessParam['Sample name'] = self.textEdit_1.toPlainText()
            if self.checkBox_2.isChecked():
                MessParam['Channel VIS'] = int(self.spinBox_3.text())
            if self.checkBox_3.isChecked():
                MessParam['Lightsource current VIS [mA]'] = float(self.lineEdit_5.text())
            if self.checkBox_4.isChecked():
                MessParam['Number of scans VIS'] = int(self.spinBox_1.text())
            if self.checkBox_5.isChecked():
                MessParam['Acquisition time VIS [ms]'] = float(self.lineEdit_1.text())
            if self.checkBox_6.isChecked():
                MessParam['Scan delay time VIS [ms]'] = float(self.lineEdit_6.text())
            if self.checkBox_7.isChecked():
                MessParam['Channel NIR'] = int(self.spinBox_4.text())
            if self.checkBox_8.isChecked():
                MessParam['Lightsource current NIR [mA]'] = float(self.lineEdit_7.text())
            if self.checkBox_9.isChecked():
                MessParam['Number of scans NIR'] = int(self.spinBox_2.text())
            if self.checkBox_10.isChecked():
                MessParam['Acquisition time NIR [ms]'] = float(self.lineEdit_3.text()) 
            if self.checkBox_11.isChecked():
                MessParam['Scan delay time NIR [ms]'] = float(self.lineEdit_8.text())
            
            #Wrap dictionary into the list to avoid error
            PAR_Data = pd.DataFrame([MessParam]).transpose()
            PAR_Data = PAR_Data.reset_index()
            
            col_names1= ['Wavelengh_VIS (nm)', 'Dark measurement_VIS', 'Raw_VIS','Intensity_VIS']
            col_names2= ['Wavelengh_NIR (nm)', 'Dark measurement_NIR', 'Raw_NIR','Intensity_NIR']
            VIS_Data.columns = col_names1
            NIR_Data.columns = col_names2
            if os.path.isdir(self.directory) == True:
                pass
            elif os.path.isdir(self.directory) == False:
                self.directory=os.getcwd().replace('/','\\')
            #print(self.directory)
            filename = str(time)
            #print(filename)
            file_path = os.path.join(self.directory, filename)
            #print(file_path)
            save_path = file_path + ".xlsx"
            print(save_path)
        
            with pd.ExcelWriter(save_path, engine='xlsxwriter') as writer:
                PAR_Data.to_excel(writer, sheet_name='Info', index=True)
                VIS_Data.to_excel(writer, sheet_name='VIS', index=True)
                NIR_Data.to_excel(writer, sheet_name='NIR', index=True)
                
                writer.sheets['VIS'].set_column(1, 511, 20)
                writer.sheets['NIR'].set_column(1, 255, 20)
                writer.sheets['Info'].set_column(1,1, 40)
                writer.sheets['Info'].set_column(2,2, 20)
                
class Window2(QWidget):
    """
    Open second window
    """
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()
        self.setWindowTitle("Info")
        self.label1 = QLabel("GUI for mobile multimodal spectroscopy system")
        self.label2 = QLabel("Version 1 (2024)")
        self.label3 = QLabel("created by Andreas Peckhaus")
        self.label4 = QLabel("email: andreas.peckhaus@insion.de")
        layout.addWidget(self.label1)
        layout.addWidget(self.label2)
        layout.addWidget(self.label3)
        layout.addWidget(self.label4)
        self.setLayout(layout)
        
class Window3(QWidget):
    """
    Open third window
    """
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()
        self.setWindowTitle("Help")
        self.label1 = QLabel("Technical support available via the resource:")
        self.label2 = QLabel("Documentation for the integrated UV/VIS-NIR handheld spectrometer system  ")
        layout.addWidget(self.label1)
        layout.addWidget(self.label2)
        self.setLayout(layout)
        
if __name__ == "__main__":
    app = QApplication([])
    ts = TScan_Kombi.TScan()
    window = MatplotlibWidget()
    window.show()
    app.exec_()

