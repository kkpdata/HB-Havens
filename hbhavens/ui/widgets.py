# -*- coding: utf-8 -*-
"""
Created on  : Thu Aug 24 16:46:40 2017
Author      : Guus Rongen, HKV Lijn in Water
Project     : PR0000.00
Description :

"""

from math import isclose
from shapely.plotting import patch_from_polygon as PolygonPatch
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT

from PyQt5 import QtCore, QtGui, QtWidgets

from hbhavens.ui import models as HBHModels

# Adjust rc Parameters
plt.rcParams['axes.linewidth'] = 0.5
plt.rcParams['axes.labelsize'] = 'medium'
plt.rcParams['grid.alpha'] = 0.25
plt.rcParams['legend.handletextpad'] = 0.4
plt.rcParams['legend.fontsize'] = 8
plt.rcParams['legend.labelspacing'] = 0.2
plt.rcParams['font.size'] = 8

class CustomTabWidget(QtWidgets.QTabWidget):
    def __init__(self, parent):
        # Inherit properties
        super(CustomTabWidget, self).__init__(parent)

        # Add dictionary with tabs
        self.tabwidgets = parent.tabwidgets
        self.mainwindow = parent
        self.tabindex = {}

        # Connect focus changed to update background (on conditions)
        self.currentChanged.connect(self.tab_focus)

    def tab_focus(self):
        """
        Method to update map background when the tab focus changes

        The background may be executed only after the mainwindow has
        been fully loaded. This is ensured my waiting for at least
        two tabs, and the tab to focused on must by the overview tab.
        """
        if len(self.tabwidgets.keys()) <= 1:
            return None
        
        if self.currentIndex() != self.tabindex['Overzicht']:
            return None
        
        if self.tabwidgets['Overzicht'].mapwidget.WMTS is None:
            self.tabwidgets['Overzicht'].mapwidget._update_background(None)

    def handler(self, item, column_no):
        """
        Handler for opening a tab
        """
        if hasattr(item, 'tabname') and (item.tabname in self.tabwidgets.keys()):
            self._open_or_focus(item.tabname)

    def close_by_name(self, name):
        """
        Close a tab based on the name

        Parameters
        ----------
        name : str
            Name of the tab to close
        """
        # Close the tab
        self.removeTab(self.tabindex[name])

        # Remove the tab too
        # TODO delete all tabs when new loads are loaded
        if name in ['Exporteren', 'Hydraulische belasting steunpunt']:
            self.tabwidgets[name].deleteLater()
            del self.tabwidgets[name]

    def _open_or_focus(self, tabname):
        """
        Method to open a tab if it is not opened yet, or focus on it when it
        is already opened
        """
        # Get all tab names + index
        self.tabindex = {self.tabText(index) : index for index in range(self.count())}
        
        # Check if it exists
        if tabname not in self.tabwidgets.keys():
            raise KeyError('"{}" not known'.format(tabname))
        
        # Open:
        self.addTab(self.tabwidgets[tabname], tabname)
        
        # And focus
        self.tabindex = {self.tabText(index) : index for index in range(self.count())}
        
        # Get all tab names + index
        self.setCurrentIndex(self.tabindex[tabname])

        # Enable of disable next step button
        self.tabwidgets[tabname].set_finished()

        # Execute custom task on tab focus
        self.tabwidgets[tabname].on_focus()

class CustomNavigationToolbar(NavigationToolbar2QT):

    # only display the buttons we need
    toolitems = [t for t in NavigationToolbar2QT.toolitems if
                 t[0] in ('Home', 'Pan', 'Zoom', 'Save')]

    def __init__(self, canvas, widget, update_func):
        super(CustomNavigationToolbar, self).__init__(canvas, widget)
        self.widget = widget
        self.update_func = update_func

    # Overwrite the original function by appending the update function
    def release_zoom(self, event):
        super(CustomNavigationToolbar, self).release_zoom(event)
        self.update_func(event)

    # Overwrite the original function, and append the update function
    def release_pan(self, event):
        super(CustomNavigationToolbar, self).release_pan(event)
        self.update_func(event)

    # Overwrite the original function, and append the update function
    def home(self):
        if hasattr(self.widget, 'WMTS'):
            bounds = self.widget.schematisation.get_harbor_bounds()
        else:
            bounds = None
            
        # If none, use basic home
        if bounds is None:
            super(CustomNavigationToolbar, self).home()
        
        # update the background
        self.update_func(0, limits=bounds)


class ExtendedLineEdit(QtWidgets.QWidget):

    def __init__(self, label, labelwidth=None, browsebutton=False):
        """
        Extended LineEdit class. A browse button can be added, as well as an
        infomessage.

        Parameters
        ----------
        label : str
            label for the line edit
        labelwidth : int
            width (points) of label
        browsebutton : boolean
            Whether to add ad browse button
        info : str
            infomessage to display (not implemented)
        """

        super(QtWidgets.QWidget, self).__init__()
        self.label = label
        self.labelwidth = labelwidth
        self.browsebutton = browsebutton

        self.init_ui()

    def init_ui(self):
        """
        Build ui element
        """

        self.setLayout(QtWidgets.QHBoxLayout())
        self.layout().setSpacing(10)
        self.layout().setContentsMargins(5, 0, 5, 0)

        self.Label = QtWidgets.QLabel()
        self.Label.setText(self.label)
        if self.labelwidth is not None:
            self.Label.setFixedWidth(self.labelwidth)
        self.layout().addWidget(self.Label)

        self.LineEdit = QtWidgets.QLineEdit()
        self.LineEdit.setMinimumWidth(200)
        self.LineEdit.setReadOnly(True)
        self.layout().addWidget(self.LineEdit)

        if self.browsebutton:
            self.BrowseButton = self.browsebutton
            self.BrowseButton.setFixedWidth(25)
            self.layout().addWidget(self.BrowseButton)

    def get_value(self):
        """
        Get value from line edit
        """
        return self.LineEdit.text()

    def set_value(self, value):
        """
        Set value to line edit
        """
        if not isinstance(value, str):
            value = str(value)
        self.LineEdit.setText(value)



class SpinBoxInputLine(QtWidgets.QWidget):

    def __init__(self, label, labelwidth=None, unitlabel='', range=None, info=False):
        """
        Parameters
        ----------
        label : str
            label for the line edit
        labelwidth : int
            width (points) of label
        unitlabel : str
            text to add behind line edit
        info : str
            infomessage to display (not implemented)
        """

        super(QtWidgets.QWidget, self).__init__()

        # Get minimum maximum and step from range
        if range is not None:
            minimum, maximum, step = range
        else:
            minimum, maximum, step = None, None, None

        self.setLayout(QtWidgets.QHBoxLayout())
        self.layout().setSpacing(10)
        self.layout().setContentsMargins(5, 0, 5, 0)

        # Add label, with width if given
        self.Label = QtWidgets.QLabel()
        self.Label.setText(label)
        if labelwidth:
            self.Label.setFixedWidth(labelwidth)
        self.layout().addWidget(self.Label)

        # Add spinbox
        self.doubleSpinBox = QtWidgets.QDoubleSpinBox()
        self.doubleSpinBox.setMinimumWidth(60)
        self.doubleSpinBox.setMaximumWidth(60)
        self.doubleSpinBox.setAlignment(QtCore.Qt.AlignRight)
        self.layout().addWidget(self.doubleSpinBox)
        
        # Add range and step if given
        if minimum:
            self.doubleSpinBox.setMinimum(minimum)
        if maximum:
            self.doubleSpinBox.setMaximum(maximum)
        if step:
            self.doubleSpinBox.setSingleStep(step)

        # Add unit label if given
        if unitlabel:
            self.Label = QtWidgets.QLabel()
            self.Label.setText(unitlabel)
            self.layout().addWidget(self.Label)

        # Add spacer to the right
        self.layout().addItem(QtWidgets.QSpacerItem(0, 0, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Maximum))

class ParameterInputLine(QtWidgets.QWidget):

    def __init__(self, label, labelwidth=None, unitlabel=None, validator=None, default=None):
        """
        LineEdit class extended with a label in front (description)
        and behind (unit).

        Parameters
        ----------
        label : str
            label for the line edit
        labelwidth : int
            width (points) of label
        unitlabel : str
            text to add behind line edit
        info : str
            infomessage to display (not implemented)
        """

        super(QtWidgets.QWidget, self).__init__()

        self.label = label
        self.labelwidth = labelwidth
        self.unitlabel = unitlabel
        self.validator = validator
        self.default_value = default
        if default is not None:
            if not isinstance(default, str):
                self.default_value = str(default)

        
        self.init_ui()

    def init_ui(self):
        """
        Build ui layout
        """
        self.setLayout(QtWidgets.QHBoxLayout())
        self.layout().setSpacing(10)
        self.layout().setContentsMargins(5, 0, 5, 0)

        # Add label
        self.Label = QtWidgets.QLabel()
        self.Label.setText(self.label)
        if self.labelwidth:
            self.Label.setFixedWidth(self.labelwidth)
        self.layout().addWidget(self.Label)

        # Add line edit
        self.LineEdit = QtWidgets.QLineEdit(self.default_value)
        self.LineEdit.setMinimumWidth(40)
        self.LineEdit.setMaximumWidth(60)
        self.LineEdit.setAlignment(QtCore.Qt.AlignRight)
        self.layout().addWidget(self.LineEdit)

        if self.validator is not None:
            self.LineEdit.setValidator(self.validator)

        # Add unit label
        if self.unitlabel is not None:
            self.Label = QtWidgets.QLabel()
            self.Label.setText(self.unitlabel)
            self.layout().addWidget(self.Label)

        # Add spacer to the right
        self.layout().addItem(QtWidgets.QSpacerItem(0, 0, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Maximum))

    def get_value(self):
        """
        Get value from line edit
        """
        return self.LineEdit.text()

    def set_value(self, value):
        """
        Set value to line edit
        """
        if not isinstance(value, str):
            value = str(value)
        self.LineEdit.setText(value)

    def set_enabled(self, enabled):
        """
        Enable of disable widget elements
        """
        # enable or disable elements in the layout
        for i in range(self.layout().count()):
            item = self.layout().itemAt(i)
            if item.widget():
                item.widget().setEnabled(enabled)
        

class ParameterLabel(QtWidgets.QWidget):

    def __init__(self, label, labelwidth=None, value=None, unit=None):
        """
        LineEdit class extended with a label in front (description)
        and behind (unit).

        Parameters
        ----------
        label : str
            label for the line edit
        labelwidth : int
            width (points) of label
        value : str
            value of the parameter
        """

        super(QtWidgets.QWidget, self).__init__()

        self.label = label
        self.labelwidth = labelwidth
        self.value = value
        self.unit = unit
        
        self.init_ui()

    def init_ui(self):
        """
        Build ui layout
        """
        self.setLayout(QtWidgets.QHBoxLayout())
        self.layout().setSpacing(10)
        self.layout().setContentsMargins(5, 0, 5, 0)

        # Add label
        self.key_label = QtWidgets.QLabel(self.label)
        if self.labelwidth:
            self.key_label.setFixedWidth(self.labelwidth)
        self.layout().addWidget(self.key_label)

        # Add line edit
        self.value_label = QtWidgets.QLabel(self.value)
        self.value_label.setAlignment(QtCore.Qt.AlignRight)
        self.value_label.setFixedWidth(60)
        self.layout().addWidget(self.value_label)

        # Add unit label if given
        if self.unit is not None:
            self.unit_label = QtWidgets.QLabel(self.unit)
            self.layout().addWidget(self.unit_label)

        # Add spacer to the right
        self.layout().addItem(QtWidgets.QSpacerItem(0, 0, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Maximum))

    def get_value(self):
        """
        Get value from line edit
        """
        return self.value_label.text()

    def set_value(self, value):
        """
        Set value to line edit
        """
        if not isinstance(value, str):
            value = str(value)
        self.value_label.setText(value)

class ComboboxInputLine(QtWidgets.QWidget):

    def __init__(self, label, labelwidth=None, items=None, default=None, spacer=True):
        """
        LineEdit class extended with a label in front (description)
        and behind (unit).

        Parameters
        ----------
        label : str
            label for the line edit
        labelwidth : int
            width (points) of label
        items : list
            List with items to add to the combobox
        """

        super(QtWidgets.QWidget, self).__init__()

        self.label = label
        self.labelwidth = labelwidth
        self.items = items
        self.default = default

        self.init_ui(spacer)

        # Add default value
        if self.default is not None:
            if not self.default in self.items:
                raise ValueError('{} not in {}'.format(self.default, ', '.join(self.items)))
            else:
                self.combobox.setCurrentText(self.default)


    def init_ui(self, spacer):
        """
        Build ui layout
        """
        self.setLayout(QtWidgets.QHBoxLayout())
        self.layout().setSpacing(10)
        self.layout().setContentsMargins(5, 0, 5, 0)

        # Add label
        self.Label = QtWidgets.QLabel()
        self.Label.setText(self.label)
        if self.labelwidth:
            self.Label.setFixedWidth(self.labelwidth)
        self.layout().addWidget(self.Label)

        # Add line edit
        self.combobox = QtWidgets.QComboBox()
        self.combobox.addItems(self.items)
        self.layout().addWidget(self.combobox)

        # Add spacer to the right
        if spacer:
            self.layout().addItem(QtWidgets.QSpacerItem(0, 0, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Maximum))

    def get_value(self):
        """
        Get value from combobox
        """
        return self.combobox.currentText()

    def set_value(self, value):
        """
        Set value to combobox
        """
        if not isinstance(value, str):
            value = str(value)
        self.combobox.setCurrentText(value)

class CheckBoxInput(QtWidgets.QWidget):

    def __init__(self, labels, nrows, unitlabel=''):
        """
        LineEdit class extended with a label in front (description)
        and behind (unit).

        Parameters
        ----------
        labels : list
            label to fill the grid
        nrows : int
            number of rows
        unitlabel : str
            unit string to append to each label
        """

        super(QtWidgets.QWidget, self).__init__()

        self.labels = labels
        self.nrows = nrows
        if unitlabel:
            self.unitlabel = ' ' if unitlabel[0] != ' ' else '' + unitlabel
        else:
            self.unitlabel = unitlabel
        
        # Build ui
        self.init_ui()

        
    def init_ui(self):
        """
        Build ui layout
        """
        self.setLayout(QtWidgets.QGridLayout())
        self.layout().setSpacing(10)
        self.layout().setContentsMargins(5, 0, 5, 0)

        self.checkboxes = {}
        for i, label in enumerate(self.labels):
            # Add unit label to label to create text element
            text = str(label) + self.unitlabel
            # Create checkbox
            checkbox = QtWidgets.QCheckBox(text)
            checkbox.setChecked(True)
            # Add to dict and layout
            self.checkboxes[label] = checkbox
            self.layout().addWidget(checkbox, i % self.nrows, i // self.nrows)

        # Add spacer as a last element on the right
        self.layout().addItem(
            QtWidgets.QSpacerItem(0, 0, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Maximum),
            0,
            len(self.labels) // self.nrows + 1
        )

    def get_value(self):
        """
        Get value from combobox
        """
        return [label for label, checkbox in self.checkboxes.items() if checkbox.isChecked()]

    def set_value(self, labels):
        """
        Set value to combobox

        Parameters
        ----------
        labels : list
            List with labels that need to be checked
        """
        for label, checkbox in self.checkboxes.items():
            checkbox.setChecked(label in labels)

class SimpleGroupBox(QtWidgets.QGroupBox):

    def __init__(self, widgets, orientation, title=None):

        # Inherit
        QtWidgets.QGroupBox.__init__(self)

        # Layout
        groupbox = QtWidgets.QGroupBox()
        if title is not None:
            self.setTitle(title)
        if (orientation == 'horizontal') or (orientation == 'h'):
            layout = QtWidgets.QHBoxLayout()
        elif (orientation == 'vertical') or (orientation == 'v'):
            layout = QtWidgets.QVBoxLayout()
        else:
            raise ValueError(f'Orientation "{orientation}" not recognized.')
        
        # Add widgets
        for widget in widgets:
            layout.addWidget(widget)

        self.setLayout(layout)

class AbstractTabWidget(QtWidgets.QWidget):

    def __init__(self, mainwindow=None):

        QtWidgets.QWidget.__init__(self, mainwindow)

        # Get mainmodel
        self.mainwindow = mainwindow
        self.mainmodel = mainwindow.mainmodel

        # Get project and settings
        self.project = self.mainmodel.project
        self.settings = self.project.settings
        
        # Get submodels
        self.schematisation = self.mainmodel.schematisation
        self.hydraulic_loads = self.mainmodel.hydraulic_loads
        self.simple_calculation = self.mainmodel.simple_calculation
        self.swan = self.mainmodel.swan
        self.pharos = self.mainmodel.pharos
        self.hares = self.mainmodel.hares
        
        self.modeluncertainties = self.mainmodel.modeluncertainties
        self.export = self.mainmodel.export

        # Initialize the finished state
        self.finished = False
        self.set_finished(self.finished)

    def set_finished(self, finished=None):
        """
        Set the tab to finished or not.
        If the tab is finished, the next button is enabled, and the user can
        continue. By default, this is false.

        Parameters
        ----------
        finished : boolean
            Whether the tab is finished
        """
        # next button enabled
        if finished is not None:
            self.finished = finished

        self.mainwindow.next_button.setEnabled(self.finished)

    def on_focus(self):
        """
        Method to execute when the focus is set on the tab. The default is
        a simple pass, but in some cases an update or initialization is
        necessary.
        """
        pass


class DataFrameWidget(QtWidgets.QTableView):
    """
    Widget to view (and edit) a pandas DataFrame
    """

    def __init__(self, dataframe, editing_enabled=False, sorting_enabled=False, index=True, column_selection=None):
        """
        Constructor

        Parameters
        ----------
        dataframe : pandas.DataFrame
            dataframe with data to be viewed
        parent : parent class
            parent class
        editing_enabled : boolean
            Whether the dataframe can be edited
        sorting_enabled : boolean
            Whether the view can be edited
        index : boolean
            Whether the index is viewed

        """
        self.editable = editing_enabled
        # Create child class
        QtWidgets.QTableView.__init__(self)
        self.horizontalHeader = self.horizontalHeader()
        self.verticalHeader = self.verticalHeader()
        self.index = index
        if not self.index:
            self.verticalHeader.setVisible(self.index)
        # Create the widget
        self.dataframe = dataframe
        self._init_UI()
        # Add the dataframe
        self.add_dataframe(self.dataframe, column_selection)
        # Enable sorting
        if sorting_enabled:
            self._enable_sorting()

        self.set_header_sizes()

    def _init_UI(self):
        # Set tableview properties
        self.setShowGrid(False)
        if not self.editable:
            self.setSelectionBehavior(QtWidgets.QTableView.SelectRows)
        self.setAlternatingRowColors(True)

    def set_header_sizes(self):
        """Set header sizes based on the first dataframe row"""
        self.horizontalHeader.setSectionResizeMode(QtWidgets.QHeaderView.Fixed)

        self.model.layoutAboutToBeChanged.emit()
        
        if hasattr(self.model, 'column_selection'):
            row = self.dataframe[self.model.column_selection].max(axis=0)
            columns = self.model.column_selection
        else:
            row = self.dataframe.max(axis=0)
            columns = self.dataframe.columns

        for i, (element, label) in enumerate(zip(row, columns)):
            width = max(36 + len(str(element)) * 6, 70)
            labelwidth = len(label) * 7
            self.setColumnWidth(i, max(width, labelwidth))

        self.model.layoutChanged.emit()
        
    def get_QTableWidget_size(self, min_column_width):
        self.horizontalHeader.setStretchLastSection(False)
        w = self.verticalHeader.width() + 30
        for i in range(self.model.columnCount()):
            w += max(self.columnWidth(i), min_column_width)
        h = self.horizontalHeader.height() - 12
        for i in range(self.model.rowCount()):
            h += self.rowHeight(i) * 1.1
        return QtCore.QSize(w, int(h))

    def fixed_fit_to_content(self, min_column_width=0):
        size = self.get_QTableWidget_size(min_column_width)
        self.setMaximumSize(size)
        self.setMinimumSize(size)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.horizontalHeader.setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.verticalHeader.setSectionResizeMode(QtWidgets.QHeaderView.Stretch)


    def _enable_sorting(self):
        self.setSortingEnabled(True)
        self.horizontalHeader.sortIndicatorChanged.connect(self._header_triggered)

    def add_dataframe(self, dataframe, column_selection=None):
        # Create and add model
        if not self.editable:
            self.model = HBHModels.PandasModel(dataframe)
        else:
            self.model = HBHModels.PandasModelEditable(dataframe)
        if column_selection is not None:
            self.model = HBHModels.PandasModelSelection(dataframe, self, column_selection)
        self.setModel(self.model)

    def _header_triggered(self, clicked_column=None):
        self.model.sort(clicked_column, order=1)

    def export_dataframe(self):
        """
        Method to export the model dataframe
        """
        # Get path
        file_types = "Excel (*.xlsx);;CSV (*.csv)"
        path, file_type = QtWidgets.QFileDialog.getSaveFileName(None, 'Export table data', '', file_types)

        # Save file
        if file_type == 'CSV (*.csv)':
            with open(path, 'w') as f:
                f.write('sep=;\n')
                self.model._data.to_csv(f, sep=';', index=self.index)
        # Export as xlsx
        elif file_type == 'Excel (*.xlsx)':
            if not (path.endswith('.xlsx') or path.endswith('.xls')):
                path += '.xlsx'
            self.model._data.to_excel(path, index=self.index)

class PharosTableWidget(QtWidgets.QTableWidget):
    """
    DataFrameWidget child specific for the pharos table,
    which has a number of extra columns and rows with
    item delegates
    """
    def __init__(self, parent=None):
        """
        Constructor
        """
        super(PharosTableWidget, self).__init__()

        self.pharos = parent.pharos
        self.settings = parent.settings
        self.verticalHeader().hide()
        

    def set_headers(self):
        """
        # Add headers to table
        """
        greektheta = u'\u03B8'
        greekf = u'\u0192'
        horizontal_header_labels = [greekf, 'L']
        for direction in self.directions:
            direction = int(direction) if direction.is_integer() else direction
            horizontal_header_labels.append(greektheta + ' ' + str(direction))
        
        self.setHorizontalHeaderLabels(horizontal_header_labels)
        self.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)

    def initialize_content(self):
        """
        Fill the tablecontent
        """

        self.blockSignals(True)

        # Drop all rows
        while self.rowCount() > 0:
            self.removeRow(0)

        # Refer to spectrum table from pharos
        table = self.pharos.spectrum_table

        # Determine frequencies (index) and directions (columns)
        self.frequencies = table.index.tolist()
        self.directions = table.columns.tolist()
        
        # Determine size from spectrum
        nf = len(self.frequencies)
        nd = len(self.directions)
        
        # Add extra rows for comboboxes etc.
        self.setColumnCount(nd + 2)
        self.setRowCount(nf + 1)

        # Set checkboxes on first column
        for row, frequency in enumerate(self.frequencies):
            # Add frequency column with checkbox
            checkbox_item = QtWidgets.QTableWidgetItem(str(round(frequency, 3)))
            checkbox_item.setFlags(QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsEnabled)
            is_checked = any((isclose(f, frequency) for f in self.settings['pharos']['frequencies']['checked']))
            checkbox_item.setCheckState(QtCore.Qt.Checked if is_checked else QtCore.Qt.Unchecked)
            # Add to row
            self.setItem(row+1, 0, checkbox_item)

        # Set wave_lengths on second column
        rep_water_depth = self.pharos.settings['pharos']['hydraulic loads']['water depth for wave length']
        for row, frequency in enumerate(self.frequencies):
            # Add wave length L
            wave_length = self.pharos.calc_wave_length(h=rep_water_depth, T=1./frequency)
            item = QtWidgets.QTableWidgetItem(str(round(wave_length, 2)))
            item.setTextAlignment(QtCore.Qt.AlignCenter)
            self.setItem(row + 1, 1, item)

        # Set comboboxes on first row
        for col, direction in enumerate(self.directions):
            # Create combobox with items
            combobox = QtWidgets.QComboBox()
            # Set the direction name to the combobox
            combobox.setObjectName(str(direction))
            # Add changed function to combobox
            combobox.currentIndexChanged.connect(self.update_schematisation)
            # Set cell widget
            self.setCellWidget(0, col + 2, combobox)

        # Fill the table with colors
        for row in range(nf):
            for col in range(nd):
                value = table.iat[row, col]
                item = QtWidgets.QTableWidgetItem(value)
                # Change background color based on value
                color = QtGui.QColor(156, 191, 139) if value else QtGui.QColor(219, 172, 188)
                item.setBackground(color)
                self.setItem(row + 1, col + 2, item)
        
        # Fill the header with directions and symbols
        self.set_headers()

        # Connect function to get checked frequencies
        self.itemChanged.connect(self.update_frequency)
        self.blockSignals(False)
        
        # Reshape the table so the columns fit the content
        self.resizeColumnsToContents()


    def update_schematisation(self):
        """
        Method to change schematisation in settings, when the combobox
        selection has changed.
        """
        # Get combobox who sended
        combobox = self.sender()
        # Get schematisation
        selected_schematisation = combobox.currentText()
        # Get direction
        selected_direction = float(combobox.objectName())

        # Loop through schematisation and directions in settings
        for schematisation, directions in self.settings['pharos']['schematisations'].items():
            if selected_schematisation == schematisation:
                # If the direction is not present, add
                if selected_direction not in directions:
                    self.settings['pharos']['schematisations'][schematisation].append(selected_direction)
            else:
                # If the direction is present in another schematisation, remove it there
                if selected_direction in directions:
                    self.settings['pharos']['schematisations'][schematisation].remove(selected_direction)
    
    def update_frequency(self):
        """
        Method to change frequency in settings, when the checkbox
        selection has changed.
        """

        # Get frequencies
        frequencies = [
            frequency
            for row, frequency in enumerate(self.frequencies)
            if self.item(row + 1, 0).checkState() == QtCore.Qt.Checked
        ]

        # Update settings    
        self.settings['pharos']['frequencies']['checked'] = frequencies

    def update_comboboxes(self):
        """
        Method to update the schematisations in the comboboxes

        Description
        ----------
        For each column:
            clear the items
            add new items
            set the old item, if it is in the new list
        Resize the content
        """

        # Get schematisation from settings
        schematisation_dict = self.settings['pharos']['schematisations']
        schematisations = list(schematisation_dict.keys())

        for col, direction in enumerate(self.directions):
            # Select combobox widget
            combobox = self.cellWidget(0, col + 2)
            combobox.blockSignals(True)
            
            # Add new items
            combobox.clear()
            combobox.addItem('geen')
            imported_schematisation = 'geen'
            for schematisation, directions in schematisation_dict.items():
                # Add item
                combobox.addItem(schematisation)

                # If direction is in settings
                if direction in directions:
                    imported_schematisation = schematisation
                    
            # Set imported schematisation (of 'geen') to combobox
            combobox.setCurrentText(imported_schematisation)

            combobox.blockSignals(False)

        self.resizeColumnsToContents()


class ProgressBoxes(QtWidgets.QWidget):
    """
    Class to keep track of progress with checkboxes.
    The class is updated when the user goes to a new step.
    """

    def __init__(self, parent):
        """Constructor"""
        QtWidgets.QWidget.__init__(self, parent)
        self.setLayout(QtWidgets.QVBoxLayout())
        self.checkboxes = []
        self.current_step = 0

        # Get bold width of the longest text step
        self.setFixedWidth(250)
            
    def set_steps(self, steps):
        """Add steps to widget, and create checkboxes"""
        self.steps = steps
        self._remove_checkboxes()
        self._add_checkboxes()
    
    def _remove_checkboxes(self):
        """Method to remove checkboxers from the widget"""
        while self.layout().count():
            item = self.layout().takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self.current_step = 0
            
    def _add_checkboxes(self):
        """Method to add checkboxes to ui"""
        self.checkboxes = []
        for i, step in enumerate(self.steps):
            checkbox = QtWidgets.QCheckBox(step)
            checkbox.setChecked(False)
            checkbox.clicked.connect(self._get_state)
            checkbox.setObjectName(str(i))
            self.checkboxes.append(checkbox)
            self.layout().addWidget(checkbox)
        # Add spacer
        self.layout().addItem(QtWidgets.QSpacerItem(0, 0, QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Expanding))

    def _get_state(self):
        """"""
        index = int(self.sender().objectName())
        state = (index < self.current_step)
        self.checkboxes[index].setChecked(state)

    def set_current_step(self, current_step):
        """Method to adjust checkboxes to current step"""
        checkstate = True
        self.checkboxes[self.current_step].setStyleSheet("font-weight: normal")
        self.current_step = current_step
        for i, step in enumerate(self.steps):
            if i == current_step:
                checkstate = False
                self.checkboxes[i].setStyleSheet("font-weight: bold")
            self.checkboxes[i].setChecked(checkstate)

class ButtonDelegate(QtWidgets.QItemDelegate):
    """
    A delegate that places a fully functioning QPushButton in every
    cell of the column to which it's applied
    """
    def __init__(self, parent):
        # The parent is not an optional argument for the delegate as
        # we need to reference it in the paint method (see below)
        QtWidgets.QItemDelegate.__init__(self, parent)
        self.tabwidget = parent
        self.tableview = parent.tableview

    def paint(self, painter, option, index):
        # This method will be called every time a particular cell is
        # in view and that view is changed in some way.  We ask the
        # delegates parent (in this case a table view) if the index
        # in question (the table cell) already has a widget associated
        # with it.  If not, create one with the text for this index and
        # connect its clicked signal to a slot in the parent view so
        # we are notified when its used and can do something.
        if not self.tableview.indexWidget(index):
            pb = QtWidgets.QPushButton('...', self.tableview, clicked=self.tabwidget.cellButtonClicked)
            pb.setFixedWidth(30)
            pb.setFixedHeight(30)
            pb.setObjectName(str(index.row()))
            self.tableview.setIndexWidget(index, pb)

