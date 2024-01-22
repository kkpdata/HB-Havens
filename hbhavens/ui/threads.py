# -*- coding: utf-8 -*-
"""
Created on  : Thu Oct 03 14:26:34 2017
Author      : Guus Rongen, HKV Lijn in Water
Project     : PR3594.10.00
Description : HB Havens threads
"""

import sys
import traceback

from PyQt5 import QtCore
from hbhavens.ui import dialogs


def show_thread_error(err_info):
    """
    Function to show errors in thread
    """

    exctype, value, tback = err_info

    sys.__excepthook__(exctype, value, tback)

    dialogs.show_dialog(
        text='\n'.join(traceback.format_exception_only(exctype, value)),
        severity='critical',
        details='\n'.join(traceback.format_tb(tback))
    )


class SimpleCalculationThread(QtCore.QThread):
    """
    Thread for simple calculation
    """

    update_progress_signal = QtCore.pyqtSignal(int, str)
    error_signal = QtCore.pyqtSignal(object)

    def __init__(self, parent=None):
        """
        Construct
        """
        super(SimpleCalculationThread, self).__init__(parent)

        self.calculation = parent.calculation
        self.update_progress = parent.update_progress
        self.parent = parent

        # Connect signal
        self.update_progress_signal.connect(self.update_progress)
        # Link error handling to error signal
        self.error_signal.connect(show_thread_error)

    def run(self):
        """
        Run simple calculation. Pass processes and progress function
        to calculation run_all function, to select certain processes
        and keep track of the progress.

        Parameters
        ----------
        processes : list
            List of processes to be executed
        progress_function : function
            Function with which the progress can be updated
        """
        try:
            self.calculation.run_all(
                processes=self.parent.settings['simple']['processes'],
                progress_function=self.update_progress_signal.emit
            )

        except:
            self.update_progress_signal.emit(0, 'Berekening mislukt.')
            err_info = sys.exc_info()
            self.error_signal.emit(err_info)

class ImportFilesThread(QtCore.QThread):
    """
    Thread for importing output files
    """

    update_progress_signal = QtCore.pyqtSignal(int)
    update_tableview_signal = QtCore.pyqtSignal()
    error_signal = QtCore.pyqtSignal(object)
    
    def __init__(self, parent=None):
        """
        Construct
        """
        super(ImportFilesThread, self).__init__(parent)

        self.parent = parent

        # Connect update progress bar signal
        self.update_progress_signal.connect(self.update_progress)
        # Connect update tableview signal
        self.update_tableview_signal.connect(self.parent.tableview.model.layoutChanged.emit)

        if self.parent.step.startswith('I'):
            self.import_func = self.parent.swan.iteration_results.read_results
        else:
            self.import_func = self.parent.swan.calculation_results.read_results

        self.progress_bar = self.parent.import_progress

        # Link error handling to error signal
        self.error_signal.connect(show_thread_error)
        self.finished.connect(self.parent.add_results)

    def update_progress(self, add_value):
        """Update progress bar"""
        if isinstance(add_value, tuple):
            self.progress_bar.setRange(0, add_value[0])
        elif add_value == 0:
            self.progress_bar.setValue(0)
        else:    
            self.progress_bar.setValue(self.progress_bar.value() + add_value)
            if self.progress_bar.value() == self.progress_bar.maximum():
                self.parent.set_finished(True)

    def run(self):
        """
        Method to:
        1. Run import
        2. Re-sort the tableview, since the index is sorted before import
        3. Update the tableview
        """
        try:
            # import the swan results
            self.import_func(
                step=self.parent.step,
                progress_function=self.update_progress_signal.emit
            )
            
            # sort the tableview again
            self.parent.sort_tableview()
            # Emit signal to update tableview
            self.update_tableview_signal.emit()
            
        except:
            self.update_progress_signal.emit(0)
            err_info = sys.exc_info()
            self.error_signal.emit(err_info)

class ImportPharosFilesThread(QtCore.QThread):
    """
    Thread for importing output files
    """

    update_progress_signal = QtCore.pyqtSignal(int)
    update_tableview_signal = QtCore.pyqtSignal()
    error_signal = QtCore.pyqtSignal(object)
    
    def __init__(self, parent=None):
        """
        Construct
        """
        super(ImportPharosFilesThread, self).__init__(parent)

        self.parent = parent

        # Connect update progress bar signal
        self.update_progress_signal.connect(self.update_progress)
        # Connect update tableview signal
        self.update_tableview_signal.connect(
            self.parent.tableview.model.layoutChanged.emit)

        # Define progress bar
        self.progress_bar = self.parent.import_progress

        # Link error handling to error signal
        self.error_signal.connect(show_thread_error)
        self.finished.connect(self.parent.add_results)

    def update_progress(self, add_value):
        """Update progress bar"""
        if add_value == 0:
            self.progress_bar.setValue(0)
        else:    
            self.progress_bar.setValue(self.progress_bar.value() + add_value)
            if self.progress_bar.value() == self.progress_bar.maximum():
                self.parent.set_finished(True)        

    def run(self):
        """
        Method to:
        1. Run import
        2. Re-sort the tableview, since the index is sorted before import
        3. Update the tableview
        """
        try:
            # import the swan results
            self.parent.pharos.read_calculation_results(progress_function=self.update_progress_signal.emit)
            self.parent.pharos.assign_energies(progress_function=self.update_progress_signal.emit)
            # sort the tableview again
            self.parent.sort_tableview()
            # Emit signal to update tableview
            self.update_tableview_signal.emit()
            
        except:
            self.update_progress_signal.emit(0)
            err_info = sys.exc_info()
            self.error_signal.emit(err_info)

# Toegevoegd Svasek 04/10/18 - Importeren van Hares output
class ImportHaresFilesThread(QtCore.QThread):
    """
    Thread for importing Hares output files
    """

    update_progress_signal = QtCore.pyqtSignal(int)
    update_tableview_signal = QtCore.pyqtSignal()
    error_signal = QtCore.pyqtSignal(object)
    
    def __init__(self, parent=None):
        """
        Construct
        """
        super(ImportHaresFilesThread, self).__init__(parent)

        self.parent = parent

        # Connect update progress bar signal
        self.update_progress_signal.connect(self.update_progress)
        # Connect update tableview signal
        self.update_tableview_signal.connect(
            self.parent.tableview.model.layoutChanged.emit)

        # Define progress bar
        self.progress_bar = self.parent.import_progress

        # Link error handling to error signal
        self.error_signal.connect(show_thread_error)
        self.finished.connect(self.parent.add_results)

    def update_progress(self, add_value):
        """Update progress bar"""
        if add_value == 0:
            self.progress_bar.setValue(0)
        else:    
            self.progress_bar.setValue(self.progress_bar.value() + add_value)
            if self.progress_bar.value() == self.progress_bar.maximum():
                self.parent.set_finished(True)

    def run(self):
        """
        Method to:
        1. Run import
        2. Re-sort the tableview, since the index is sorted before import
        3. Update the tableview
        """
        try:
            # import the hares results
            self.parent.hares.read_calculation_results(progress_function=self.update_progress_signal.emit)

            # sort the tableview again
            self.parent.sort_tableview()

            # Emit signal to update tableview
            self.update_tableview_signal.emit()
            
        except:
            self.update_progress_signal.emit(0)
            err_info = sys.exc_info()
            self.error_signal.emit(err_info)


class GenerateSwanFilesThread(QtCore.QThread):
    """
    Thread for generating swan input files. The run method executes the
    generate function in the core swan class.
    """

    update_progress_signal = QtCore.pyqtSignal(int)
    update_tableview_signal = QtCore.pyqtSignal()
    error_signal = QtCore.pyqtSignal(object)

    def __init__(self, parent=None):
        """
        Construct
        """
        super(GenerateSwanFilesThread, self).__init__(parent)

        self.parent = parent

        # Connect signals
        self.update_progress_signal.connect(self.update_progress)
        # Connect update tableview signal
        self.update_tableview_signal.connect(self.parent.tableview.model.layoutChanged.emit)

        # Pick the right progress bar
        self.progress_bar = self.parent.generate_progress
        # Link error handling to error signal
        self.error_signal.connect(show_thread_error)

    def update_progress(self, add_value):
        """Update progress bar"""
        if add_value == 0:
            self.progress_bar.setValue(0)
        else:    
            self.progress_bar.setValue(self.progress_bar.value() + add_value)

    def run(self):
        """
        Run import
        """

        try:
            self.parent.swan.generate(
                step=self.parent.step,
                progress_function=self.update_progress_signal.emit
            )
            # Emit signal to update tableview
            self.update_tableview_signal.emit()
        except:
            self.update_progress_signal.emit(0)
            err_info = sys.exc_info()
            self.error_signal.emit(err_info)

class GeneratePharosFilesThread(QtCore.QThread):
    """
    Thread for generating swan input files. The run method executes the
    generate function in the core swan or pharos class.
    """

    update_progress_signal = QtCore.pyqtSignal(int)
    error_signal = QtCore.pyqtSignal(object)

    def __init__(self, parent=None):
        """
        Construct
        """
        super(GeneratePharosFilesThread, self).__init__(parent)

        self.parent = parent

        # Connect signals
        self.update_progress_signal.connect(self.update_progress)

        # Pick the right progress bar
        self.progress_bar = self.parent.generate_progress
        # Link error handling to error signal
        self.error_signal.connect(show_thread_error)

    def update_progress(self, add_value):
        """Update progress bar"""
        if add_value == 0:
            self.progress_bar.setValue(0)
        else:    
            self.progress_bar.setValue(self.progress_bar.value() + add_value)

    def run(self):
        """
        Run import
        """

        try:
            # Run swan or pharos generate, depending on the parent
            self.parent.pharos.generate(
                progress_function=self.update_progress_signal.emit
            )
        except:
            self.update_progress_signal.emit(0)
            err_info = sys.exc_info()
            self.error_signal.emit(err_info)

class UpdateTilesThread(QtCore.QThread):
    """
    Thread for generating output files
    """

    error_signal = QtCore.pyqtSignal(object)

    def __init__(self, parent=None):
        """
        Construct
        """
        super(UpdateTilesThread, self).__init__(parent)
        self.parent = parent
        # Link error handling to error signal
        self.error_signal.connect(show_thread_error)

    def run(self):
        """
        Run import
        """
        try:
            # Update the tiles (download new ones)
            mpp = (self.parent.bbox[2] - self.parent.bbox[0]) / self.parent.geometry().width()
            # Downloading tiles
            self.parent.WMTS.update_tiles(self.parent.bbox, mpp=mpp)
            # Plotting tiles
            self.parent.WMTS.plot_and_clean(ax=self.parent.ax, clip=True)
            # Update canvas
            self.parent.canvas.draw()

        except:
            err_info = sys.exc_info()
            self.error_signal.emit(err_info)
