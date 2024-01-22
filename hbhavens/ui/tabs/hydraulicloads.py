from itertools import chain

from matplotlib.backends.backend_qt5agg import (FigureCanvasQTAgg,
                                                NavigationToolbar2QT)
from matplotlib.cm import Spectral
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
from PyQt5 import QtCore, QtWidgets

from hbhavens.ui import widgets
from hbhavens.io import database



class HydraulicLoadTab(widgets.AbstractTabWidget):
    """
    Tab with a tableview of the hydraulic loads
    """
    def __init__(self, maintabwidget=None):
        """
        Constructor of the tab
        """
        # Create child class
        widgets.AbstractTabWidget.__init__(self, maintabwidget)
        self.maintabwidget = maintabwidget

        # Initialize if the support location has changed
        self.init_ui()

        # Allow continuing to the next step
        self.set_finished(True)

    def init_ui(self):
        """
        Build the tab
        """

        # Add table view
        self.tableview = widgets.DataFrameWidget(
            dataframe=self.hydraulic_loads,
            sorting_enabled=True,
            column_selection=['Description'] + self.hydraulic_loads.input_columns + self.hydraulic_loads.result_columns,
            index=False
        )

        # Create layout
        self.vlayout = QtWidgets.QVBoxLayout()
        hrdio = database.HRDio(self.hydraulic_loads.settings['hydraulic_loads']['HRD'])
        
        # If type 'Zoet' database (DB with water levels and wave conditions)
        if hrdio.get_type_of_hydraulic_load_id() == 2:
            self.waterlevel_button = QtWidgets.QPushButton('Waterstandniveaus bepalen', clicked=self._determine_waterlevels)
            self.waterlevel_button.setFixedWidth(150)
            
            text = QtWidgets.QLabel((
                'Voor databases met waterstanden en golfcondities, meestal overeenkomend de zoete watersystemen, '
                'is het mogelijk de golfcondities terug te rekenen naar de waterstanden. Hiermee kan het aantal '
                'door te rekenen belastingcombinaties met het oog op de rekenintensieve geavanceerde methode sterk verkleind worden.'
            ))
            text.setWordWrap(True)
            text.setContentsMargins(5, 0, 5, 0)
            omrekenbox = widgets.SimpleGroupBox([self.waterlevel_button, text], 'h', '')
            # Add 'omreken'-button
            self.vlayout.addWidget(omrekenbox)
        
        # Add view to layout
        self.vlayout.addWidget(self.tableview)
        # Add exportbutton
        self.export_button = QtWidgets.QPushButton('Exporteren', clicked=self.tableview.export_dataframe)
        hlayout = QtWidgets.QHBoxLayout()
        hlayout.addStretch()
        hlayout.addWidget(self.export_button)
        self.vlayout.addLayout(hlayout)
        # Add layout to widget
        self.setLayout(self.vlayout)

    def on_focus(self):
        """
        If new hydraulic loads are loaded, the columns to show might also need te be changed. Therefor, this function.
        """
        self.tableview.model.set_column_selection(['Description'] + self.hydraulic_loads.input_columns + self.hydraulic_loads.result_columns)

    def _determine_waterlevels(self):
        self.scatter_window = WaterlevelDialog(self)
        self.scatter_window.exec_()


class WaterlevelDialog(QtWidgets.QDialog):

    def __init__(self, hydraulicloadtab):

        QtWidgets.QDialog.__init__(self)

        self.hydraulicloadtab = hydraulicloadtab
        self.hydraulic_loads = self.hydraulicloadtab.hydraulic_loads
        self.settings = self.hydraulicloadtab.settings
        self.mainwindow = self.hydraulicloadtab.mainwindow

        self.element = {}
        self.init_window()

        self.add_waterlevel_lines()
        if not self.hydraulic_loads.recalculated_loads.empty:
            self.plot_regression()

    def init_window(self):

        self.setWindowTitle("HB Havens: waterstandniveaus kiezen")
        self.setWindowFlags(self.windowFlags() & ~QtCore.Qt.WindowContextHelpButtonHint)

        self.setWindowIcon(self.hydraulicloadtab.maintabwidget.getIcon('hbhavens.png'))

        self.setLayout(QtWidgets.QVBoxLayout())
        mainlayout = QtWidgets.QHBoxLayout()
        self.layout().addLayout(mainlayout)
        self.figure = Figure(figsize=(8, 5.5))

        # Add canvas
        self.canvas = FigureCanvasQTAgg(self.figure)
        def nothing(event, limits=None):
            pass
        self.toolbar = widgets.CustomNavigationToolbar(canvas=self.canvas, widget=self, update_func=nothing)

        canvasbox = QtWidgets.QVBoxLayout()
        canvasbox.addWidget(self.toolbar)
        canvasbox.addWidget(self.canvas)
        canvasbox.addStretch()

        mainlayout.addLayout(canvasbox)

        self.ax = self.figure.add_subplot()
        self.ax.grid()
        self.ax.spines['right'].set_visible(False)
        self.ax.spines['top'].set_visible(False)
        self.ax.tick_params(axis='y', color='0.75')
        self.ax.tick_params(axis='x', color='0.75')

        self.ax.set_xlabel('Waterstand [m+NAP]')
        self.ax.set_ylabel('Hs [m]')

        self.figure.tight_layout()

        vlayout = QtWidgets.QVBoxLayout()

        # levelpicker.setFixedWidth(150)
        self.detect_breaks_button = QtWidgets.QPushButton('Zoek niveaus', clicked=self.detect_breaks)
    
        self.table = QtWidgets.QTableWidget()
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setVisible(False)
        self.table.verticalHeader().setDefaultSectionSize(24)
        self.table.verticalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Fixed)
        self.table.setRowCount(1)
        self.table.setColumnCount(1)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.itemChanged.connect(self.update_items)
        
        levelpicker = widgets.SimpleGroupBox([self.detect_breaks_button, self.table], 'v', '1. Bepalen waterstandniveaus')
        levelpicker.setFixedWidth(200)
        levelpicker.setFixedHeight(350)
        vlayout.addWidget(levelpicker)
        vlayout.addSpacing(10)
    
        # Button for regression
        self.get_waveconditions_button = QtWidgets.QPushButton('Bepaal golfcondities', clicked=self.segmented_regression)
        
        self.regrbutton = QtWidgets.QRadioButton('Regressie')
        self.regrbutton.toggled.connect(self.set_recalc_method)
        self.regrbutton.setChecked(self.settings['hydraulic_loads']['recalculate_method'] == 'regression')
        self.intpbutton = QtWidgets.QRadioButton('Interpoleren')
        self.intpbutton.toggled.connect(self.set_recalc_method)
        self.intpbutton.setChecked(self.settings['hydraulic_loads']['recalculate_method'] == 'interpolation')
        
        wavecondbox = widgets.SimpleGroupBox([self.regrbutton, self.intpbutton, self.get_waveconditions_button], 'vertical', '2. Bepaal golfcondities bij waterstanden')
        wavecondbox.setFixedWidth(200)
        vlayout.addWidget(wavecondbox)
        vlayout.addSpacing(10)
        
        # Button for accepting or restoring loads
        self.accept_result_button = QtWidgets.QPushButton('Accepteren', clicked=self.accept_result)
        self.reload_loads_button = QtWidgets.QPushButton('Oorspronkelijke belastingen herladen', clicked=self.reload_loads)
        levelpicker = widgets.SimpleGroupBox([self.accept_result_button, self.reload_loads_button], 'vertical', '3. Accepteren of terugzetten')
        levelpicker.setFixedWidth(200)
        vlayout.addWidget(levelpicker)

        vlayout.addStretch()
        mainlayout.addLayout(vlayout)

        line = QtWidgets.QFrame()
        line.setFrameShape(QtWidgets.QFrame.HLine)
        line.setFrameShadow(QtWidgets.QFrame.Sunken)

        self.layout().addWidget(line)

        self.close_button = QtWidgets.QPushButton('Sluiten', clicked=self.close)
        self.layout().addWidget(self.close_button, 0, QtCore.Qt.AlignRight)


        # Set table from settings
        self.update_breaks()

        # Initialize buttons
        recalced = self.settings['hydraulic_loads']['recalculate_waterlevels']
        self.detect_breaks_button.setEnabled(not recalced)
        self.get_waveconditions_button.setEnabled(not recalced)
        self.accept_result_button.setEnabled(not recalced)
        self.reload_loads_button.setEnabled(recalced)

    def set_recalc_method(self):
        """Set recalculation method. Method is changed in settings
        """
        self.remove_element('result_line')
        self.remove_element('result_scatter')

        if self.regrbutton.isChecked():
            self.settings['hydraulic_loads']['recalculate_method'] = 'regression'
        elif self.intpbutton.isChecked():
            self.settings['hydraulic_loads']['recalculate_method'] = 'interpolation'

    def remove_element(self, param):

        # Remove element if present
        if param in self.element.keys():
            self.element[param].remove()
            del self.element[param]
            # self.legend.remove(param)
        self.canvas.draw_idle()

    def add_waterlevel_lines(self):
        s=5 if self.settings['hydraulic_loads']['recalculate_waterlevels'] else 1
        minwa = self.hydraulic_loads['Wind direction'].min()
        maxwa = self.hydraulic_loads['Wind direction'].max()
        hs, waterlevel, direction = self.hydraulic_loads[['Hs', self.hydraulic_loads.wlevcol, 'Wind direction']].drop_duplicates().values.T        
        self.element['scatter'] = self.ax.scatter(waterlevel, hs, c=direction, s=s, cmap='Spectral')
        self.canvas.draw_idle()
        
    def detect_breaks(self):

        # Run function to detect breaks
        self.mainwindow.setCursorWait()
        self.hydraulic_loads.detect_waterlevel_breaks()
        self.mainwindow.setCursorNormal()
        self.update_breaks()

    def segmented_regression(self):

        # Run function for segmented regression
        self.mainwindow.setCursorWait()
        self.hydraulic_loads.calculate_waveconditions()
        self.mainwindow.setCursorNormal()
        self.plot_regression()

    def update_breaks(self):

        # Get waterlevels from settings
        waterlevels = sorted(self.settings['hydraulic_loads']['waterlevels'])
        # Add values to tables
        self.table.itemChanged.disconnect()
        self.table.setRowCount(len(waterlevels)+1)
        for i, wlev in enumerate(waterlevels):
            self.table.setItem(i, 0, QtWidgets.QTableWidgetItem(str(wlev)))
        self.table.setItem(len(waterlevels), 0, None)
        self.table.itemChanged.connect(self.update_items)

        # Change lines
        for wlev in waterlevels:
            # Add line
            tag = f'vline{wlev:.5f}'
            if tag not in self.element:
                self.element[tag] = self.ax.axvline(wlev, linestyle='--', color='0.2', linewidth=0.75)
            # Update line
            else:
                self.element[tag].set_xdata([wlev, wlev])

        # Remove line
        wlevtags = [f'vline{wlev:.5f}' for wlev in waterlevels]
        for tag in list(self.element):
            if tag.startswith('vline') and tag not in wlevtags:
                self.element[tag].remove()
                del self.element[tag]
        
        self.canvas.draw_idle()

    def update_items(self):
        # Get items
        items = [self.table.item(i, 0) for i in range(self.table.rowCount())]
        waterlevels = set(self.settings['hydraulic_loads']['waterlevels'])
        tablevalues = set(sorted([float(item.data(0)) for item in items if (item is not None and item.data(0) != '')]))

        # Add to waterlevels if not already present
        self.settings['hydraulic_loads']['waterlevels'].extend(list(tablevalues.difference(waterlevels)))

        # Remove values from waterlevels
        for val in waterlevels.difference(tablevalues):
            self.settings['hydraulic_loads']['waterlevels'].remove(val)

        # Sort in place 
        values = self.settings['hydraulic_loads']['waterlevels'][:]
        del self.settings['hydraulic_loads']['waterlevels'][:]
        self.settings['hydraulic_loads']['waterlevels'].extend(sorted(values))

        # Update table
        self.update_breaks()

    def plot_regression(self):

        self.remove_element('result_line')
        self.remove_element('result_scatter')

        lines = []
        colors = []
        for comb, group in self.hydraulic_loads.recalculated_loads.groupby(['Wind direction', 'Wind speed']):
            hs, waterlevel = group[['Hs', 'Water level']].sort_values(by='Water level').drop_duplicates().values.T
            lines.append(list(zip(waterlevel, hs)))
            colors.append(Spectral(comb[0]/360.))

        self.element['result_line'] = self.ax.add_collection(LineCollection(lines, colors=colors, linewidth=0.5))
        colors=list(chain(*([color] * len(line) for line, color in zip(lines, colors))))
        self.element['result_scatter'] = self.ax.scatter(*list(zip(*chain(*lines))), edgecolors=colors, s=25, facecolor='None', lw=1.0)

        self.canvas.draw_idle()

    def accept_result(self):

        self.mainwindow.setCursorWait()
        
        self.hydraulicloadtab.tableview.model.layoutAboutToBeChanged.emit()
        self.hydraulic_loads.adapt_interpolated()
        self.hydraulicloadtab.tableview.model.set_column_selection(['Description'] + self.hydraulic_loads.input_columns + self.hydraulic_loads.result_columns)
        self.hydraulicloadtab.tableview.model.layoutChanged.emit()

        self.detect_breaks_button.setEnabled(False)
        self.get_waveconditions_button.setEnabled(False)
        self.accept_result_button.setEnabled(False)
        self.reload_loads_button.setEnabled(True)

        self.remove_element('scatter')
        self.add_waterlevel_lines()

        self.mainwindow.setCursorNormal()


    def reload_loads(self):
        """
        Method to reload original loads
        """
        self.mainwindow.setCursorWait()
        
        self.hydraulicloadtab.tableview.model.layoutAboutToBeChanged.emit()
        self.hydraulic_loads.restore_non_interpolated()
        self.hydraulicloadtab.tableview.model.set_column_selection(['Description'] + self.hydraulic_loads.input_columns + self.hydraulic_loads.result_columns)
        self.hydraulicloadtab.tableview.model.layoutChanged.emit()

        self.detect_breaks_button.setEnabled(True)
        self.get_waveconditions_button.setEnabled(True)
        self.accept_result_button.setEnabled(True)
        self.reload_loads_button.setEnabled(False)

        self.remove_element('scatter')
        self.add_waterlevel_lines()

        
        self.mainwindow.setCursorNormal()
