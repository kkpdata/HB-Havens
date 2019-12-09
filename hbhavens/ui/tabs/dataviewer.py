import itertools
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.figure import Figure
from PyQt5 import QtWidgets, QtCore

from hbhavens.ui import widgets, dialogs
from hbhavens.ui.tabs.general import InteractiveLegend, SplittedTab

logger = logging.getLogger(__name__)

class DataViewerTab(SplittedTab):
    """
    Widget with the scatter data viewer.
    """
    def __init__(self, mainwindow=None):
        """
        Constructor of the tab
        """

        # Create left and right widget
        self.scatterplot = ScatterPlotWidget()
        self.dataselector = DataSelectorWidget(scatterplot=self.scatterplot)

        # Create child class
        SplittedTab.__init__(self, mainwindow, leftwidget=self.scatterplot, rightwidget=self.dataselector)

    def add_table(self, method):
        """
        Method to add table. To be called once a process is finished.
        """
        if method == 'simple':
            self.dataselector.add_table(
                dataframe=self.mainmodel.simple_calculation.combinedresults.output,
                name='Eenvoudige methode',
                input_variables=['Location'] + self.mainmodel.hydraulic_loads.input_columns,
                # input_description=[('Windsnelheid', '[m/s]'), ('Waterstand', '[m+NAP]'), ('Windrichting', '[graden tov Noord]')]
                result_variables={
                    'Hs': ['Hs', 'Hs,lg', 'Hs,totaal', 'Hs,out'],
                    'Golfdoordringing': ['Kd', 'Kt', 'Kd,t'],
                    'Golfenergie': ['Ed', 'Et', 'Ed,t', 'Elg'],
                    'Golfrichting': ['Wave direction', 'Wind direction', 'Diffraction direction', 'Combined wave direction']
                }
            )

        elif method == 'swan_iterations':
            calc_res = self.mainmodel.swan.iteration_results
            hlcols = self.mainmodel.hydraulic_loads.input_columns
            table = calc_res.reindex(columns=calc_res.columns.union(hlcols)).sort_index()
            table[hlcols] = self.mainmodel.hydraulic_loads[hlcols].sort_index().values

            self.dataselector.add_table(
                dataframe=table,
                name='Swan steunpunt iteraties',
                input_variables=self.mainmodel.hydraulic_loads.input_columns,
                result_variables={
                    'Hs': ['Hs', 'Hs steunpunt 1', 'Hs steunpunt 2', 'Hs steunpunt 3'],
                    'Tp': ['Tp', 'Tp,s steunpunt 1', 'Tp,s steunpunt 2', 'Tp,s steunpunt 3']
                }
            )

        elif method == 'swan_calculations':
            calc_res = self.mainmodel.swan.calculation_results
            hlcols = self.mainmodel.hydraulic_loads.input_columns
            table = calc_res.reindex(columns=calc_res.columns.union(hlcols)).sort_index()
            nlocations = len(calc_res.result_locations)
            table[hlcols] = np.tile(self.mainmodel.hydraulic_loads[hlcols].sort_index().values, (nlocations, 1))
            
            self.dataselector.add_table(
                dataframe=table,
                name='Swan berekeningen',
                input_variables=['Location'] + hlcols,
                result_variables={
                    'Hs': ['Hs', 'Hm0_D', 'Hm0_TR', 'Hm0_W', 'Hm0 swan'],
                    'Tp': ['Tp', 'Tp_D', 'Tp_TR', 'Tp_W', 'Tp swan'],
                    'Tm-1,0': ['Tm-1,0', 'Tmm10_D', 'Tmm10_TR', 'Tmm10_W', 'Tm-1,0 swan'],
                    'Wave direction': ['Tm-1,0', 'Theta0_D', 'Theta0_TR', 'Theta0_W', 'Wave direction swan']
                }
            )

        elif method == 'pharos_calculations':
            calc_res = self.mainmodel.pharos.calculation_results
            hlcols = self.mainmodel.hydraulic_loads.input_columns
            result_vars = {
                'Hs': ['Hs', 'Hs pharos', 'Hm0 swan', 'Hs totaal'],
                'Tp': ['Tp', 'Tp pharos', 'Tp swan', 'Tp totaal'],
                'Tm-1,0': ['Tm-1,0', 'Tm-1,0 pharos', 'Tm-1,0 swan', 'Tm-1,0 totaal'],
                'Wave direction': ['Wave direction', 'Wave direction pharos', 'Wave direction swan', 'Wave direction totaal']
            }
            rescols = list(itertools.chain(*list(result_vars.values())))
            table = calc_res.reindex(columns=['Location', 'HydraulicLoadId'] + hlcols + rescols).sort_values(by=['Location', 'HydraulicLoadId'])
            
            self.dataselector.add_table(
                dataframe=table,
                name='SWAN & PHAROS',
                input_variables=['Location'] + hlcols,
                result_variables=result_vars
            )

        elif method == 'hares_calculations':
            calc_res = self.mainmodel.hares.calculation_results
            hlcols = self.mainmodel.hydraulic_loads.input_columns
            result_vars = {
                'Hs': ['Hs', 'Hs hares', 'Hm0 swan', 'Hs totaal'],
                'Tp': ['Tp', 'Tp hares', 'Tp swan', 'Tp totaal'],
                'Tm-1,0': ['Tm-1,0', 'Tm-1,0 hares', 'Tm-1,0 swan', 'Tm-1,0 totaal'],
                'Wave direction': ['Wave direction', 'Wave direction hares', 'Wave direction swan', 'Wave direction totaal']
            }
            rescols = list(itertools.chain(*list(result_vars.values())))
            table = calc_res.reindex(columns=['Location', 'HydraulicLoadId'] + hlcols + rescols).sort_values(by=['Location', 'HydraulicLoadId'])
            
            self.dataselector.add_table(
                dataframe=table,
                name='SWAN & HARES',
                input_variables=['Location'] + hlcols,
                result_variables=result_vars
            )

        else:
            raise KeyError(f'Method not recognized: "{method}"')

class ScatterPlotWidget(QtWidgets.QWidget):

    def __init__(self):

        QtWidgets.QWidget.__init__(self)

        self.init_tab()

    def init_tab(self):

        self.setLayout(QtWidgets.QVBoxLayout())
        # self.figure, self.ax = plt.subplots(figsize=(10, 10))
        self.figure = Figure(figsize=(10, 10))

        # Add canvas
        self.canvas = FigureCanvasQTAgg(self.figure)
        def nothing(event, limits=None):
            pass
        self.toolbar = widgets.CustomNavigationToolbar(canvas=self.canvas, widget=self, update_func=nothing)

        canvasbox = QtWidgets.QVBoxLayout()
        canvasbox.addWidget(self.toolbar)
        canvasbox.addWidget(self.canvas)
        canvasbox.addItem(QtWidgets.QSpacerItem(0, 0, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding))

        self.layout().addLayout(canvasbox)

        self.ax = self.figure.add_subplot()
        self.ax.grid()
        self.ax.spines['right'].set_visible(False)
        self.ax.spines['top'].set_visible(False)
        self.ax.tick_params(axis='y', color='0.75')
        self.ax.tick_params(axis='x', color='0.75')

        self.scatters = {}
        self.resultvals = {}

        self.legend = InteractiveLegend(self, self.scatters, loc='upper left')#, title='Klik op punt\nom te verbergen')

        self.canvas.mpl_connect('pick_event', self._onpick)

    def _onpick(self, event):
        self.legend._onpick(event)
        self.update_scatters()

    def remove_scatter(self, param):

        # Remove element if present
        if param in self.scatters.keys():
            self.scatters[param].remove()
            del self.scatters[param]
            self.legend.remove(param)

    def remove_all_scatters(self):
        # Remove present plot elements
        keys = list(self.scatters.keys())
        list(map(self.remove_scatter, keys))
        self.canvas.draw_idle()

    def get_visible(self):
        """Get a list of visible scatters"""
        return [key for key, scatter in self.scatters.items() if scatter.get_visible()]

    def update_axis(self, xlabel, xticks, ylabel, yparams):
        """
        Method to update axis after table has been changed
        """
        self.ax.set_xlabel(xlabel)

        if isinstance(xticks[0], str):
            self.ax.set_xticks(list(range(len(xticks))))
            self.minxspace = 1
            self.ax.set_xticklabels(xticks, rotation=90)
        else:
            self.ax.set_xticks(xticks)
            self.ax.set_xticklabels(xticks)
            self.minxspace = np.diff(xticks).min()

        self.ax.set_ylabel(ylabel)

        # Result parameters
        for i, param in enumerate(yparams):
            self.scatters[param], = self.ax.plot([], [], ms=3, alpha=0.5, ls='', marker='o', color=f'C{i}')
            self.legend.add_item(param, handle=self.ax.scatter([], [], s=20, alpha=0.5, marker='o', color=f'C{i}'), label=param)

        self.legend._update_legend()
        self.figure.tight_layout()


    def update_scatters(self, resultvals=None):

        # Update data if provided
        if resultvals is not None:
            self.resultvals.clear()
            self.resultvals.update(resultvals)

        visible = self.get_visible()
        if len(visible) == 0:
            return None
        else:
            xoffsets = np.linspace(-self.minxspace / 4, self.minxspace / 4, len(visible)+2)[1:-1]

        i = 0
        for param, values in self.resultvals.items():
            if param == 'x':
                continue
            elif param not in visible:
                continue
            self.scatters[param].set_data(self.resultvals['x'] + xoffsets[i], values)
            i += 1

        # Resize
        self.ax.relim()
        self.ax.autoscale_view(True, True, True)
        self.figure.tight_layout()
        self.canvas.draw_idle()

class DataSelectorWidget(QtWidgets.QWidget):

    def __init__(self, scatterplot):

        # Inherit parent
        QtWidgets.QWidget.__init__(self)

        # Link scatterplot widget
        self.scatterplot = scatterplot

        # Define variables
        self.tables = {}
        self.input_variables = {}
        self.result_variables = {}
        self.xvalue = ''
        self.yvalue = ''
        self.selectedtable = ''

        # Initialize widget
        self.setLayout(QtWidgets.QVBoxLayout())

        # Combobox for selecting table
        self.tableselector = widgets.ComboboxInputLine('Tabel:', 100, [''], spacer=False)
        self.tableselector.combobox.currentIndexChanged.connect(self.update_table)
        groupbox = widgets.SimpleGroupBox([self.tableselector], 'v', 'Selecteer een tabel:')
        self.layout().addWidget(groupbox)
        self.layout().addSpacing(10)

        # Combobox for selecting x and y variables
        self.xvariable = widgets.ComboboxInputLine('Variabele op x-as:', 100, [''], spacer=False)
        self.xvariable.combobox.currentIndexChanged.connect(self.get_combination)

        self.yvariable = widgets.ComboboxInputLine('Variabelen op y-as:', 100, [''], spacer=False)
        self.yvariable.combobox.currentIndexChanged.connect(self.get_combination)

        groupbox = widgets.SimpleGroupBox([self.xvariable, self.yvariable], 'v', 'Selecteer een variabele voor de assen:.')
        self.layout().addWidget(groupbox)
        self.layout().addSpacing(10)
        
        # Adjust widths
        for cbox in [self.tableselector, self.xvariable, self.yvariable]:
            cbox.combobox.setMinimumWidth(150)
            cbox.combobox.setSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Fixed)

        # Dataselection
        self.dataselection = {}
        groupbox = QtWidgets.QGroupBox()
        groupbox.setMinimumHeight(400)
        groupbox.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.MinimumExpanding)
        groupbox.setTitle('Selecteer belastingcombinaties voor puntenwolk:')
        groupbox.setLayout(QtWidgets.QVBoxLayout())
        self.dataselectionlayout = QtWidgets.QHBoxLayout()
        groupbox.layout().addLayout(self.dataselectionlayout)
        self.layout().addWidget(groupbox)

    def construct_dataselection(self):
        """
        Method to construct tables with which the data for the scatterplot can be selected
        Changes after the table changes.
        """

        # Get load variables for selected table
        loadvars = {var: sorted(self.tables[self.selectedtable][var].unique())
                    for var in self.input_variables[self.selectedtable]}

        for col, values in loadvars.items():
            vlayout = QtWidgets.QVBoxLayout()
            label = QtWidgets.QLabel(col+':')
            label.setWordWrap(True)
            label.setMinimumHeight(30)
            label.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignBottom)
            vlayout.addWidget(label)
            table = QtWidgets.QTableWidget()
            vlayout.addWidget(table)
            table.verticalHeader().setVisible(False)
            table.verticalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Fixed)
            table.verticalHeader().setDefaultSectionSize(24)
            table.horizontalHeader().setVisible(False)
            table.setRowCount(len(values))
            table.setColumnCount(1)
            table.horizontalHeader().setStretchLastSection(True)
            table.itemSelectionChanged.connect(self._update_data)
            table.values = values
            for i, value in enumerate(values):
                table.setItem(i, 0, QtWidgets.QTableWidgetItem(str(value)))

            self.dataselection[col] = table

            self.dataselectionlayout.addLayout(vlayout)

    def _delete_widgets(self, layout):
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    def destruct_dataselection(self):

        # Remove scatters
        self.scatterplot.remove_all_scatters()
            
        # Remove dataselections
        while self.dataselectionlayout.count():
            layout = self.dataselectionlayout.takeAt(0)
            if layout is not None:
                self._delete_widgets(layout)
        self.dataselection.clear()

        self.xvariable.combobox.setCurrentIndex(0)
        self.yvariable.combobox.setCurrentIndex(0)

        # Clear comboboxes
        while self.xvariable.combobox.count() > 1:
            self.xvariable.combobox.removeItem(1)
        while self.yvariable.combobox.count() > 1:
            self.yvariable.combobox.removeItem(1)

    def add_table(self, dataframe, name, input_variables, result_variables):
        """
        Method with which a table can be made available in the dataviewer

        It is checked if all the input and result variables are present in the given dataframe.
        """

        # Check if all columns are present
        tabcolumns = dataframe.columns.array
        for col in input_variables + list(itertools.chain(*result_variables.values())):
            if col not in tabcolumns:
                raise KeyError(f'Column "{col}" not in dataframe. ({", ".join(tabcolumns)}).')

        # If already present, remove:
        if name in self.tables:
            del self.tables[name]
            del self.input_variables[name]
            del self.result_variables[name]
            self.tableselector.combobox.setCurrentIndex(0)
        else:
            # Reset table-selector
            self.tableselector.combobox.addItem(name)

        # Add to data
        self.tables[name] = dataframe
        self.input_variables[name] = input_variables
        self.result_variables[name] = result_variables

    def get_combination(self):
        """
        Method to select data based on selected variables

        1. Get selected combobox items
        2. If one of the two is None, empty data
        3.
        """
        # Get axis variables
        self.xvalue = self.xvariable.combobox.currentText()
        self.yvalue = self.yvariable.combobox.currentText()

        # Update selection mode
        for col, table in self.dataselection.items():
            if col == self.xvalue:
                table.selectAll()
                table.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
            else:
                table.clearSelection()
                table.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)

        # Remove all the 'old' scatters
        self.scatterplot.remove_all_scatters()

        # If either x or y is not selected, return
        if (self.xvalue == '') or (self.yvalue == ''):
            return None

        # Update axis
        self.scatterplot.update_axis(
            xlabel=self.xvalue,
            xticks=sorted(self.tables[self.selectedtable][self.xvalue].unique()),
            ylabel=self.yvalue,
            yparams=self.result_variables[self.selectedtable][self.yvalue]
        )

    def _update_data(self):
        """
        Update the data in the scatterplot based on the selection
        """
        # Get the selection
        items = {var: [] for var in self.input_variables[self.selectedtable]}
        for column, widget in self.dataselection.items():
            for idx in widget.selectedIndexes():
                if idx.data() is not None:
                    items[column].append(widget.values[idx.row()])

        # Select part of the dataframe based on the selection
        table = self.tables[self.selectedtable]
        idx = table[list(items)].isin(items).all(axis=1)

        if not idx.any():
            return None

        if self.xvalue == '':
            dialogs.NotificationDialog('Kies een variabele voor de x-as.')
            return None
        if self.yvalue == '':
            dialogs.NotificationDialog('Kies een variabele voor de y-as.')
            return None


        # Fill the scatters
        result_params = self.result_variables[self.selectedtable][self.yvalue]
        xvals = table.loc[idx, self.xvalue].values

        if isinstance(xvals[0], str):
            dct = {name: i for i, name in enumerate(sorted(np.unique(xvals)))}
            xvals = list(map(dct.get, xvals))

        # Get visible
        resultvals = {param: column for param, column in table.loc[idx, result_params].iteritems()}
        resultvals['x'] = xvals

        self.scatterplot.update_scatters(resultvals)

    def update_table(self):
        """
        Method called when the selected table is changed. After this
        the variables that can be selected are also changes
        """
        newtext = self.tableselector.combobox.currentText()
        if newtext != self.selectedtable:
            self.destruct_dataselection()
            
        self.selectedtable = self.tableselector.combobox.currentText()
        
        if self.selectedtable == '':
            self.destruct_dataselection()
            return None        

        # Input variable
        self.xvariable.combobox.addItems(self.input_variables[self.selectedtable])
        self.xvariable.combobox.setCurrentIndex(0)

        # Result variable
        self.yvariable.combobox.addItems(list(self.result_variables[self.selectedtable].keys()))
        self.yvariable.combobox.setCurrentIndex(0)

        self.construct_dataselection()
