import logging
import os

import geopandas as gpd
import matplotlib
import numpy as np
import pandas as pd
from descartes import PolygonPatch
from matplotlib.backends.backend_qt5agg import (FigureCanvasQTAgg,
                                                NavigationToolbar2QT)
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.figure import Figure
from matplotlib.patches import Arrow, PathPatch
from matplotlib.patches import Polygon as mplPolygon
from PyQt5 import Qt, QtCore, QtGui, QtWidgets
from shapely.geometry import (
    LineString, MultiLineString, MultiPolygon, Point, Polygon)

from hbhavens import io
from hbhavens.core import geometry
from hbhavens.ui import threads, widgets
from hbhavens.ui.models import WMTS
from hbhavens.ui.tabs.general import InteractiveLegend, SplittedTab

logger = logging.getLogger(__name__)

import time


class MoveThread(QtCore.QThread):
    """Thread to prevent calling update background too often.
    """
    def __init__(self, mapwidget, plotter):
        """
        Construct
        """
        super(MoveThread, self).__init__()
        self.mapwidget = mapwidget
        self.plotter = plotter
        self.step = 0
        
    def run(self):
        self.step = 0
        while self.step < 30:
            time.sleep(0.01)
            self.step += 1
        # Update mapwidget
        self.mapwidget._update_background(None)
        self.plotter.splittermoved

class OverviewMapTab(SplittedTab):
    """
    Widget with the background map

    This widget works based on a WMTS, which contains background tiles
    within and around the axes limits. Each action can update the bounding
    box, and thus the tile set.
    """

    def __init__(self, mainwindow):
        # Leftwidget
        self.mapwidget = MapWidget(mainmodel=mainwindow.mainmodel, maintab=self)
        
        # Rightwidget
        self.dataselector = DataSelectorWidget(main=self, hydraulic_loads=mainwindow.mainmodel.hydraulic_loads)

        super(OverviewMapTab, self).__init__(mainwindow, self.mapwidget, self.dataselector)

        self.plotter = ResultsPlotter(mainwindow.mainmodel.simple_calculation, self.mapwidget, dataselector=self.dataselector)

        self.move_thread = MoveThread(self.mapwidget, self.plotter)

    def on_moved(self):
        # Call parent method
        super(OverviewMapTab, self).on_moved()
        # Check if the thread is still running from an older background update
        if self.move_thread.isRunning():
            self.move_thread.quit()
            self.move_thread.step = 0
        # Start the thread
        self.move_thread.start()

    def update_load(self):
        self.plotter.set_load()

    def add_table(self, method):
        """
        Method to add table with results. To be called once a process is finished.
        """
        if method == 'simple':
            # Create results table
            table = self.mainmodel.simple_calculation.diffraction.output[['Lr', 'Beq', 'Kd', 'breakwater', 'Diffraction direction']]
            table = table.join(self.mainmodel.simple_calculation.transmission.output[['Kt']])
            table = table.join(self.mainmodel.simple_calculation.wavegrowth.output[['maxdir', 'Hs,lg', 'Feq']])
            table = table.join(self.mainmodel.simple_calculation.wavebreaking.output[['Hs,max', 'Breaking length']])
            table = table.join(self.mainmodel.simple_calculation.combinedresults.output.set_index(table.index.names)[['Hs,totaal', 'Hs,out', 'Combined wave direction']])
            table['breaking_fraction'] = 1 - np.minimum(table['Hs,out'].fillna(0.0) / table['Hs,totaal'].fillna(0.0), 1.0)
            table.loc[table['Hs,totaal'].eq(0.0), 'breaking_fraction'] = 0.0
            table.drop('Hs,max', axis=1, inplace=True)
            
            self.dataselector.add_table(
                dataframe=table,
                name='Eenvoudige methode',
                result_variables=['Diffractie (Kd)', 'Transmissie (Kt)', 'Lokale Golfgroei (Hs,lg)', 'Golfbreking (-)', 'Gecombineerd (Hs)'],
                input_variables=self.mainmodel.hydraulic_loads.input_columns[:],
                angle_variables=None
            )

            # Specifically add simple results to plotter
            self.plotter.add_simple_results()

        # No data is added for iterations, since it is only the support location
        elif method == 'swan_iterations':
            pass

        elif method == 'swan_calculations':
            resultvars = [
                'Hm0_D', 'Hm0_TR', 'Hm0_W', 'Hm0 swan',
                'Tp_D', 'Tp_TR', 'Tp_W', 'Tp swan',
                'Tmm10_D', 'Tmm10_TR', 'Tmm10_W', 'Tm-1,0 swan',
            ]
            angle_vars = ['Theta0_D', 'Theta0_TR', 'Theta0_W', 'Wave direction swan']
            angle_variables = {key: var for key, var in zip(resultvars, angle_vars * 3)}

            cols = self.mainmodel.swan.calculation_results.columns.intersection(resultvars + angle_vars + ['Location', 'HydraulicLoadId'])
            self.dataselector.add_table(
                dataframe=self.mainmodel.swan.calculation_results.reindex(columns=cols).set_index(['Location', 'HydraulicLoadId']),
                name='SWAN berekeningen',
                input_variables=self.mainmodel.hydraulic_loads.input_columns[:],
                result_variables=resultvars,
                angle_variables=angle_variables
            )

        elif method == 'pharos_calculations':
            resultvars = [
                'Hs pharos', 'Hm0 swan', 'Hs totaal',
                'Tp pharos', 'Tp swan', 'Tp totaal',
                'Tm-1,0 pharos', 'Tm-1,0 swan', 'Tm-1,0 totaal'
            ]
            angle_vars = ['Wave direction pharos', 'Wave direction swan', 'Wave direction totaal']
            angle_variables = {key: var for key, var in zip(resultvars, angle_vars * 3)}

            cols = self.mainmodel.pharos.calculation_results.columns.intersection(resultvars + angle_vars + ['Location', 'HydraulicLoadId'])
            self.dataselector.add_table(
                dataframe=self.mainmodel.pharos.calculation_results.reindex(columns=cols).set_index(['Location', 'HydraulicLoadId']),
                name='SWAN & PHAROS',
                input_variables=self.mainmodel.hydraulic_loads.input_columns[:],
                result_variables=resultvars,
                angle_variables=angle_variables
            )

        elif method == 'hares_calculations':
            resultvars = [
                'Hs hares', 'Hm0 swan', 'Hs totaal',
                'Tp hares', 'Tp swan', 'Tp totaal',
                'Tm-1,0 hares', 'Tm-1,0 swan', 'Tm-1,0 totaal'
            ]
            angle_vars = ['Wave direction hares', 'Wave direction swan', 'Wave direction totaal']
            angle_variables = {key: var for key, var in zip(resultvars, angle_vars * 3)}

            cols = self.mainmodel.hares.calculation_results.columns.intersection(resultvars + angle_vars + ['Location', 'HydraulicLoadId'])
            self.dataselector.add_table(
                dataframe=self.mainmodel.hares.calculation_results.reindex(columns=cols).set_index(['Location', 'HydraulicLoadId']),
                name='SWAN & HARES',
                input_variables=self.mainmodel.hydraulic_loads.input_columns[:],
                result_variables=resultvars,
                angle_variables=angle_variables
            )
        
        else:
            raise KeyError(f'Method not recognized: "{method}"')

                        
class MapWidget(QtWidgets.QWidget):
    """
    Widget with the background map

    This widget works based on a WMTS, which contains background tiles
    within and around the axes limits. Each action can update the bounding
    box, and thus the tile set.
    """

    element_dct = {
        'harborarea': {
            'kwargs': {'facecolor': 'C0', 'alpha': 0.4, 'edgecolor': 'C0'},
            'handle': 'patch',
            'label': 'haventerrein'
        },
        'breakwaters': {
            'kwargs': {'colors': 'C1', 'linewidths': 2, 'alpha': 1.0},
            'handle': 'element',
            'label': 'havendam(men)'
        },
        'flooddefence': {
            'kwargs': {'colors': 'C3', 'linewidths': 2, 'alpha': 1.0},
            'handle': 'element',
            'label': 'waterkering'
        },
        'support_locations': {
            'kwargs': {'color': '0.1', 'marker': 'x', 'ms': 6, 'mew': 2, 'alpha': 1.0},
            'handle': 'element',
            'label': 'HRD-locaties'
        },
        'result_locations': {
            'kwargs': {'color': '0.1', 'marker': 'o', 'ms': 2, 'alpha': 1.0},
            'handle': 'element',
            'label': 'uitvoerlocaties'
        },
        'entrance': {
            'kwargs': {'lw': 1.5, 'color': '0.1', 'dashes': (2,2), 'alpha': 1.0},
            'handle': 'element',
            'label': 'haveningang'
        },
        'inner': {
            'kwargs': {'color': 'grey', 'alpha': 0.3},
            'handle': 'patch',
            'label': 'binnengaats gebied'
        },
        'support_location': {
            'kwargs': {'color': '0.1', 'marker': 'o', 'ms': 10, 'alpha': 1.0, 'mfc': 'none'},
            'handle': 'element',
            'label': 'steunpuntlocatie'
        }
    }

    def __init__(self, mainmodel, maintab):
        """
        Constructor of the tab
        """

        # Create child class
        QtWidgets.QWidget.__init__(self)

        self.landboundary = io.geometry.read_landboundary()
            
        # Dictionaries for legend and toggling items
        self.plot_elements = {}
        
        self.bbox = (0., 300000., 280000., 620000.)
        self.labels = {}
        self.legend = InteractiveLegend(self, self.plot_elements)

        self.mainmodel = mainmodel
        self.main = maintab
        self.schematisation = mainmodel.schematisation
        self.init_widget()
        self.load_project()

        self.update_tiles_thread = threads.UpdateTilesThread(self)


    def init_widget(self):
        # Aangepast Svasek 31/10/2018 - Ander gebruik van figure, waardoor er in Spyder geen extra figuur opent
        # Create figure
        self.figure = Figure(figsize=(self.geometry().width() / 100., self.geometry().height() / 100.))

        # Add canvas
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.mpl_connect('pick_event', self.legend._onpick)

        # this is the Navigation widget
        # it takes the Canvas widget and a parent
        self.toolbar = widgets.CustomNavigationToolbar(canvas=self.canvas, widget=self, update_func=self._update_background)

        # Add checkbox for showing labels
        hbox = QtWidgets.QHBoxLayout()
        self.label_checkbox = QtWidgets.QCheckBox('Laat labels zien')
        self.label_checkbox.setChecked(False)
        self.label_checkbox.stateChanged.connect(self.show_labels)
        hbox.addWidget(self.label_checkbox, 0, QtCore.Qt.AlignLeft)

        # Create WMTS selection
        label = QtWidgets.QLabel('Achtergrondkaartlaag:')
        hbox.addWidget(label, 0, QtCore.Qt.AlignRight)
        self.wmts_combobox = QtWidgets.QComboBox()

        self.comboboxitems = [
            ('Geen achtergrond', '', ''),
            ('OpenTopo Achtergrondkaart', 'layer.png', 'opentopoachtergrondkaart'),
            ('Luchtfoto PDOK', 'layer.png', '2017_ortho25'),
            ('AHN2 5m DTM', 'layer.png', 'ahn2_05m_ruw'),
            ('BRT achtergrondkaart Grijs', 'layer.png', 'brtachtergrondkaartgrijs')
        ]

        # Get path to icon data
        for text, iconname, userdata in self.comboboxitems:
            # Get path to icon
            iconpath = os.path.join(self.mainmodel.datadir,'icons',iconname)
            # Add item to combobox with icon
            self.wmts_combobox.addItem(QtGui.QIcon(iconpath), text, userdata)

        self.wmts_combobox.setCurrentIndex(0)
        hbox.addWidget(self.wmts_combobox)
        self.wmts_combobox.currentIndexChanged.connect(self._set_WMTS_layer)

        # create an axis
        self.ax = self.figure.add_axes([0, 0, 1, 1])
        self.bbox = self._get_filled_bbox(self.bbox)
        self.ax.set_xlim(self.bbox[0], self.bbox[2])
        self.ax.set_ylim(self.bbox[1], self.bbox[3])

        # Add WMTS
        self.WMTS = None
        self._set_WMTS_layer(0)

        self.landboundary, = self.ax.plot(*self.landboundary, color='0.8', zorder=-20, lw=0.75)
        if self.WMTSlayer == '':
            self.landboundary.set_visible(True)
        self.ax.axis('off')

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        layout.addLayout(hbox)

        self.setLayout(layout)

    
    def load_project(self):
        """
        Set project elements visible
        """
        if not self.schematisation.result_locations.empty:
            self.set_visible('result_locations')
        if not self.schematisation.harborarea.empty:
            self.set_visible('harborarea')
        if not self.schematisation.breakwaters.empty:
            self.set_visible('breakwaters')
            if len(self.schematisation.breakwaters) == 2 or hasattr(self.schematisation, 'entrance_coordinate'):
                self.set_visible('inner')
                self.set_visible('entrance')
        if not self.schematisation.flooddefence.empty:
            self.set_visible('flooddefence')
        if not self.schematisation.support_locations.empty:
            self.set_visible('support_locations')
        if not pd.isnull(self.schematisation.support_location).all():
            self.set_visible('support_location')

    def show_labels(self, state=QtCore.Qt.Checked):
        """
        Update labels on canvas

        Parameters
        ----------
        state : QtCore.Qt.[state]
            State of the checkbox
        """
        if state == QtCore.Qt.Checked:
            # Both result and support locations
            for locations, column in zip(['result_locations', 'support_locations'], ['Naam', 'Name']):
                # If not in visible elements, continue
                if locations not in self.plot_elements.keys():
                    continue
                
                # Plot each label
                for row in getattr(self.schematisation, locations).itertuples():
                    labelkey = locations[0] + getattr(row, column)
                    if labelkey not in self.labels.keys():
                        self.labels[labelkey] = self.ax.text(*row.geometry.coords[0], s=getattr(row, column), ha='left', va='bottom')

        else:
            # Remove all labels
            for key, label in self.labels.items():
                label.remove()
            self.labels.clear()
                
        self.canvas.draw()


    def _set_WMTS_layer(self, idx):
        """
        Change the WMTS layer
        """
        try:
            activated_idx = idx
            if idx == -1:
                return None
    
            item = self.comboboxitems[idx]
            if not item:
                return None
    
            matched_idx = self.wmts_combobox.findData(item[2])
            assert activated_idx == matched_idx
    
            # Get layer name from combobox
            self.WMTSlayer = item[2]
    
            # Remove the present WMTS
            if self.WMTS is not None:
                self.WMTS.clean_all()
                self.WMTS = None
                self.landboundary.set_visible(True)
    
            # Skip if no background
            if self.WMTSlayer != '':
                self.landboundary.set_visible(False)
                # Add Web Mapping Tile Service
                self.WMTS = WMTS(
                    'http://geodata.nationaalgeoregister.nl/tiles/service',
                    'EPSG:28992',
                    self.WMTSlayer
                )
    
            self._update_background(None)
        except Exception as e:
            print(e)

    def _get_filled_bbox(self, limits):
        # Get canvas size
        geo = self.canvas.geometry()
        
        canvas_width = (geo.width()) / 100.
        canvas_height = (geo.height()) / 100.
        canvas_ratio = max(1, canvas_width) / max(1, canvas_height)

        # Get axis size
        left, lower, right, upper = limits

        ax_width = right - left
        ax_height = upper - lower
        ax_ratio = (ax_width / ax_height)

        # Als de assen te hoog zijn, corrigeer de breedte
        if canvas_ratio > ax_ratio:
            ax_width = ax_width / (ax_ratio / canvas_ratio)
            center = 0.5 * (left + right)
            left = center - 0.5 * ax_width
            right = center + 0.5 * ax_width

        # Als de assen te breed zijn
        elif canvas_ratio < ax_ratio:
            ax_height = ax_height / (canvas_ratio / ax_ratio)
            center = 0.5 * (upper + lower)
            lower = center - 0.5 * ax_height
            upper = center + 0.5 * ax_height

        # Return bbox
        return (left, lower, right, upper)

    def _update_background(self, event, limits=None):
        """
        Update the bounding box
        """
        # Get the limits of the frame for the bouding box
        if not limits:
            limits = (self.ax.get_xlim()[0], self.ax.get_ylim()[0], self.ax.get_xlim()[1], self.ax.get_ylim()[1])
        self.bbox = self._get_filled_bbox(limits)
        # Reset the axis limits
        self.ax.set_xlim(self.bbox[0], self.bbox[2])
        self.ax.set_ylim(self.bbox[1], self.bbox[3])

        # Scale markers in plotter
        self.main.plotter.set_location_values()
        self.canvas.draw_idle()

        # Skip if no layer
        if self.WMTSlayer == '':
            return None

        # Check if the thread is still running from an older background update
        if self.update_tiles_thread.isRunning():
            self.update_tiles_thread.quit()
            self.update_tiles_thread.wait()
            logger.debug('Tile update thread interrupted.')

        # Start the thread
        self.update_tiles_thread.start()

    def remove_plot_element(self, element):
        """
        Remove plotted element from view

        Parameters
        ----------
        element : str
            element that is removed
        """
        # Remove element if present
        if element in self.plot_elements.keys():
            self.plot_elements[element].remove()
            del self.plot_elements[element]
            self.legend.remove(element)
            
    def set_visible(self, element, ax=None):
        """
        Plot a given element

        Parameters
        ----------
        element : str
            element to be plotted
        """
        
        if ax is None:
            ax = self.ax

        # Remove the element if it is already plotted
        self.remove_plot_element(element)

        # Get plotting properties
        plot_props = self.element_dct[element]
        kwargs = plot_props['kwargs']

        # If element is geodataframe
        geometry = getattr(self.schematisation, element)
        if isinstance(geometry, (gpd.GeoDataFrame, gpd.GeoSeries)):
            if geometry.empty:
                return None
            self.plot_elements[element] = self._plot_gdf(gdf=geometry, kwargs=kwargs, ax=ax)
        elif isinstance(geometry, LineString):
            self.plot_elements[element], = ax.plot(*np.vstack(geometry.coords[:]).T, **kwargs)
        elif isinstance(geometry, (Polygon, MultiPolygon)):
            self.plot_elements[element] = ax.add_collection(PatchCollection([PolygonPatch(geometry)], **kwargs))
        else:
            raise TypeError('Geometry type not recognized.')

        # Add legend handles
        if plot_props['handle'] == 'patch':
            self.legend.add_item(element, handle=PolygonPatch(Polygon([(0,0), (0,1), (1,1), (1,0)]), **kwargs), label=plot_props['label'])
        else:
            self.legend.add_item(element, handle=self.plot_elements[element], label=plot_props['label'])

        # Update legend
        self.legend._update_legend()
        # Update labels
        if self.label_checkbox.isChecked() and element in ['support_locations', 'result_locations']:
            self.show_labels()
        self.canvas.draw_idle()


    def _plot_gdf(self, gdf, kwargs={}, ax=None):
        """
        Plot geodataframe to axis
        This function will plot a default style for all geometry types.

        Paramaters
        ----------
        gdf : geopandas.GeoDataFrame
            geodataframe with data to plot
        kwargs : dictionary
            dictionary with keyword arguments for plotting
        ax : matplotlib.pyplot.axes
            axes to plot to
        """  
        if ax is None:
            ax = self.ax

        # get geometries
        if isinstance(gdf, gpd.GeoSeries):
            geometries = [gdf['geometry']]
        else:
            geometries = gdf['geometry'].values.tolist()

        # for line and pts
        if isinstance(geometries[0], LineString):
            # Create collection
            segments = [np.vstack(geo.coords[:]) for geo in geometries]
            collection = ax.add_collection(LineCollection(segments, **kwargs))

        if isinstance(geometries[0], Polygon):
            # Create collection
            polygons = [PolygonPatch(geo) for geo in geometries]
            collection = ax.add_collection(PatchCollection(polygons, **kwargs))

        if isinstance(geometries[0], Point):
            kwargs['linestyle'] = ''
            crds = np.vstack([pt.coords[0] for pt in geometries])
            collection, = ax.plot(*crds.T, **kwargs)

        return collection


class DataSelectorWidget(QtWidgets.QWidget):

    def __init__(self, main, hydraulic_loads):

        # Inherit parent
        QtWidgets.QWidget.__init__(self)

        # Link scatterplot widget
        self.main = main
        self.hydraulic_loads = hydraulic_loads

        # Define variables
        self.tables = {}
        self.input_variables = {}
        self.result_variables = {}
        self.angle_variables = {}
        self.result_variable = ''
        self.selected_table = ''
        self.selected_process = ''
        self.hydraulic_load_id = None

        self._init_ui()

    def _init_ui(self):
        """
        Set up UI design
        """
        self.setLayout(QtWidgets.QVBoxLayout())

        # Combobox for selecting table
        self.tableselector = widgets.ComboboxInputLine('Tabel:', 100, [''], spacer=False)
        self.tableselector.combobox.currentIndexChanged.connect(self.update_table)
        groupbox = widgets.SimpleGroupBox([self.tableselector], 'v', 'Selecteer een methode:')
        self.layout().addWidget(groupbox)
        self.layout().addSpacing(10)

        # Add parameter selection
        self.parameter_combobox = widgets.ComboboxInputLine('Proces/parameter:', 100, [''], spacer=False)
        self.parameter_combobox.combobox.currentIndexChanged.connect(self.set_parameter)
        self.cbfigure = Figure(figsize=(1, 0.4))
        self.cbcanvas = FigureCanvasQTAgg(self.cbfigure)
        self.cbcanvas.setContentsMargins(5, 5, 5, 5)
        self.cbax = self.cbfigure.add_axes([0.1, 0.5, 0.8, 0.48])
        self.cbax.set_yticks([])
        self.cbax.set_xticks([])
        self.colorbar = matplotlib.colorbar.ColorbarBase(
            self.cbax, cmap=matplotlib.cm.RdYlGn_r, norm=matplotlib.colors.Normalize(vmin=0, vmax=1), orientation='horizontal')
        self.cbax.set_visible(False)
        self.cmaprange = None
        
        groupbox = widgets.SimpleGroupBox([self.parameter_combobox, self.cbcanvas], 'v', 'Selecteer een parameter:.')
        self.layout().addWidget(groupbox)
        self.layout().addSpacing(10)

        # Adjust widths
        for cbox in [self.tableselector, self.parameter_combobox]:
            cbox.combobox.setMinimumWidth(150)
            cbox.combobox.setSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Fixed)

        # Dataselection
        self.dataselection = {}
        groupbox = QtWidgets.QGroupBox()
        # groupbox.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.MinimumExpanding)
        groupbox.setTitle('Belastingcombinatie:')
        self.dataselectionlayout = QtWidgets.QVBoxLayout()
        groupbox.setLayout(self.dataselectionlayout)
        self.load_label = QtWidgets.QLabel('Kies een methode en parameter.')
        italic=QtGui.QFont()
        italic.setItalic(True)
        self.load_label.setFont(italic)
        self.load_label.setContentsMargins(5, 5, 5, 5)
        self.dataselectionlayout.addWidget(self.load_label)
        self.layout().addWidget(groupbox)

        self.layout().addStretch()
        self.setSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)

    def set_parameter(self):
        # Update selected proces
        self.selected_process = self.parameter_combobox.combobox.currentText()
        
        # Update colorbar, get range
        if self.selected_table == 'Eenvoudige methode':
            table = self.tables[self.selected_table]
            if self.selected_process == 'Diffractie (Kd)':
                self.cmaprange = (0.0, table['Kd'].max())
            elif self.selected_process == 'Transmissie (Kt)':
                self.cmaprange = (0.0, table['Kt'].max())
            elif self.selected_process == 'Lokale Golfgroei (Hs,lg)':
                self.cmaprange = (0.0, table['Hs,lg'].max())
            elif self.selected_process == 'Golfbreking (-)':
                self.cmaprange = (0, 1.0)
            elif self.selected_process == 'Gecombineerd (Hs)':
                self.cmaprange = (0.0, table['Hs,out'].max())
            else:
                self.cmaprange = None
            self.set_colorbar()
        
        elif self.selected_table == '':
            self.cmaprange = None
            self.set_colorbar()
            self.load_label.setText('Kies een methode en parameter.')
        
        else:
            if self.selected_process == '':
                self.cmaprange = None
                self.load_label.setText('Kies een methode en parameter.')
            else:
                vals = self.tables[self.selected_table][self.selected_process]
                self.cmaprange = (vals.min(), vals.max())
            self.set_colorbar()

        self.main.plotter.set_parameter()

    def set_colorbar(self):
        # If None, set axes invisible
        if self.cmaprange is None:
            self.cbax.set_visible(False)
            
        # If range is given. Change ticks
        else:
            self.cbax.set_visible(True)
            cbar_ticks = np.linspace(*self.cmaprange, num=6)
            self.colorbar.set_ticks(np.linspace(0, 1, num=6))
            self.colorbar.set_ticklabels([f'{val:.2f}' for val in cbar_ticks])
            self.colorbar.draw_all()
            
        self.cbcanvas.draw_idle()
        
    def construct_dataselection(self):
        """
        Method to construct tables with which the data for the scatterplot can be selected
        Changes after the table changes.
        """
        # First add items to parameter combobox
        self.parameter_combobox.combobox.addItems(self.result_variables[self.selected_table])

        # Get load variables for selected table
        loadvars = {var: sorted(self.hydraulic_loads[var].unique())
                    for var in self.input_variables[self.selected_table]}

        for col, values in loadvars.items():
            combobox = widgets.ComboboxInputLine(col+':', 100, list(map(str, values)), spacer=False)
            combobox.combobox.currentIndexChanged.connect(self.get_hydraulic_load_id)
            self.dataselection[col] = combobox
            self.dataselectionlayout.insertWidget(min(1, self.dataselectionlayout.count()-1), combobox)       
        
    def _delete_widgets(self, layout):
        while layout.count() > 1:
            item = layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    def destruct_dataselection(self):

        # Remove dataselections
        self._delete_widgets(self.dataselectionlayout)
        self.dataselection.clear()

        # Set process to ''
        self.parameter_combobox.combobox.setCurrentIndex(0)
        
        # Clear comboboxes
        while self.parameter_combobox.combobox.count() > 1:
            self.parameter_combobox.combobox.removeItem(1)
        
    def add_table(self, dataframe, name, input_variables, result_variables, angle_variables):
        """
        Method with which a table can be made available in the dataviewer

        It is checked if all the input and result variables are present in the given dataframe.
        """

        # Check if all columns are present
        tabcolumns = dataframe.columns.array
        # for col in input_variables + list(itertools.chain(*result_variables.values())):
        #     if col not in tabcolumns:
        #         raise KeyError(f'Column "{col}" not in dataframe. ({", ".join(tabcolumns)}).')

        # If already present, remove:
        if name in self.tables:
            del self.tables[name]
            del self.input_variables[name]
            del self.result_variables[name]
            del self.angle_variables[name]
            self.tableselector.combobox.setCurrentIndex(0)
        else:
            # Reset table-selector
            self.tableselector.combobox.addItem(name)

        # Add to data
        self.tables[name] = dataframe
        self.input_variables[name] = input_variables
        self.result_variables[name] = result_variables
        self.angle_variables[name] = angle_variables

    def get_hydraulic_load_id(self):
        """
        Method to select data based on selected variables

        1. Get selected combobox items
        2. If one of the two is None, empty data
        3.
        """
        # Update selection mode
        idx = np.ones(len(self.hydraulic_loads), dtype='bool')
        for key, combobox in self.dataselection.items():
            idx &= self.hydraulic_loads[key].eq(float(combobox.combobox.currentText())).array
        
        loadid = self.hydraulic_loads.index[idx]
        if len(loadid) > 1:
            raise ValueError('Multiple load id\'s match selection.')
        
        # Normal case
        elif len(loadid) == 1:
            self.hydraulic_load_id = loadid[0]
            text = self.hydraulic_loads.loc[loadid[0], self.hydraulic_loads.result_columns].to_dict()
            text = '\n'.join([f'{k}:\t{v:.1f}' for k, v in text.items()])
            self.load_label.setText(text)
        
        # No data case
        else:
            self.hydraulic_load_id = None
            self.load_label.setText('Belastingcombinatie komt niet voor.')
        
        # Call set data in plotter
        self.main.plotter.set_load()
            
    def update_table(self):
        """
        Method called when the selected table is changed. After this
        the variables that can be selected are also changes
        """
        # Get selected table
        self.selected_table = self.tableselector.combobox.currentText()
        # Destruct old data selection
        self.destruct_dataselection()
        # Set new data selection
        if self.selected_table != '':
            self.construct_dataselection()
            self.load_label.setText('Kies een methode en parameter.')
        else:
            return None

        # Update table
        self.table = self.tables[self.selected_table]
        # Get idx as array for faster indexing
        if 'HydraulicLoadId' in self.table.columns.array:
            self.loadidx = self.table['HydraulicLoadId'].to_numpy()
        elif 'HydraulicLoadId' in self.table.index.names:
            level = self.table.index.names.index('HydraulicLoadId')
            self.loadidx = self.table.index.get_level_values(level)
        else:
            raise KeyError('Expected "HydraulicLoadId" in index or column.')
    
class ResultsPlotter:

    def __init__(self, simple_calculation, mapwidget, dataselector):
        """
        Constuctor, copy from parent, add the results, init UI
        """
        # Links to other classes
        self.simple = simple_calculation
        self.mainmodel = simple_calculation.mainmodel
        self.ax = mapwidget.ax
        self.canvas = mapwidget.canvas
        self.mapwidget = mapwidget

        self.dataselector = dataselector

        self.transmission = self.simple.transmission

        self.area_union = self.mainmodel.schematisation.area_union
        
        # Prepare data
        self.result_locations = self.mainmodel.schematisation.result_locations
        self.breakwaters = self.mainmodel.schematisation.breakwaters

        self.hydraulic_loads = self.mainmodel.hydraulic_loads
        self.load_descriptions = self.mainmodel.hydraulic_loads.hydraulicloadid_dict
    
        # Add location points
        self.elements = {}
        
        # Build UI
        self.bg_limits = (0, 0, 0, 0)
        self.cmap = matplotlib.cm.get_cmap('RdYlGn_r')
        self.splittermoved = False

        self.visible = True

    def add_simple_results(self):
        """
        Method to add the simple results. This can only be called after the
        results have been generated. The function generated some coordinates
        needed for plotting, preprocesses the load tables.
        """
        # Coordinates and data
        self.coords = {f'bwhead{i}': row.breakwaterhead.coords[0] for i, row in enumerate(self.mainmodel.schematisation.breakwaters.itertuples())}
        if 'bwhead1' in list(self.coords):
            self.coords['bwhead2'] = np.vstack([self.coords['bwhead0'], self.coords['bwhead1']]).mean(axis=0)
        else:
            self.coords['bwhead2'] = self.mainmodel.schematisation.entrance.centroid.coords[0]
        
        # Add first parameter and load
        self.set_parameter()

    def initialize_geometries(self):
        """
        Initialize geometries specifically for a process
        """
        process = self.dataselector.selected_process

        if process == 'Diffractie (Kd)':
            # Lines from location to breakwater head
            for name in self.result_locations['Naam'].array:
                self.elements[name], = self.ax.plot([], [], color='grey', lw=0.75)
            # Width between breakwater heads perp to flow direction
            self.elements['Beq'], = self.ax.plot([], [], color='k', lw=1.25)
            # Representative Wave length
            self.elements['Lr'], = self.ax.plot([], [], color='k', lw=1.25)

        elif process == 'Transmissie (Kt)':
            for breakwater in self.breakwaters.itertuples():
                # Shading area per breakwater
                self.elements[breakwater.Index] = self.ax.add_patch(mplPolygon([(0, 0), (0, 0)], color='grey', lw=0.75, alpha=0.2))
                # Text at breakwater
                pt = breakwater.geometry.interpolate(breakwater.geometry.length / 2)
                rotation = np.degrees(geometry.get_orientation(breakwater.geometry, pt))
                self.elements[f'vb_{breakwater.Index}'] = self.ax.text(pt.x, pt.y, '', rotation=(rotation+90) % 180 - 90, va='bottom', ha='center')
        
        elif process == 'Lokale Golfgroei (Hs,lg)':
            # Fetch lines
            for name in self.result_locations['Naam'].array:
                self.elements[name], = self.ax.plot([], [], color='grey', lw=0.75)

        elif process == 'Golfbreking (-)':
            # Wave direction lines
            for name in self.result_locations['Naam'].array:
                self.elements[name], = self.ax.plot([], [], color='grey', lw=0.75)

        # If no data visualisation
        if process == '':
            self.mapwidget.set_visible('support_locations')
            self.mapwidget.set_visible('result_locations')
            self.canvas.draw_idle()

        else:
            # Scatter
            self.resultxy = [np.array([row.geometry.x, row.geometry.y]) for row in self.result_locations.sort_values(by='Naam').itertuples()]
            self.rotations = np.zeros(len(self.resultxy))
            self.values = np.zeros(len(self.resultxy))
            self.markerpath = np.array([[0.0, -0.14], [0.4, -0.36], [0.0, 0.5], [-0.4, -0.36], [0.0, -0.14]])
            theta = np.linspace(0, 2*np.pi, 50)
            self.circlepath = np.c_[np.cos(theta) * 0.3, np.sin(theta) * 0.3]

            self.elements['scatter'] = PatchCollection(
                [PathPatch(matplotlib.path.Path(self.markerpath * 300 + crd[None, :]), facecolor='none', edgecolor='k') for crd in self.resultxy])
            self.ax.add_collection(self.elements['scatter'])
            
            self.set_location_values(np.zeros(len(self.result_locations)))
            self.mapwidget.remove_plot_element('support_locations')
            self.mapwidget.remove_plot_element('result_locations')

    
    def set_visible(self, visible):
        """Method to make drawn elements (in)visible
        
        Parameters
        ----------
        visible : bool
        """
        if visible == self.visible:
            return None
        self.visible = visible
        for key in list(self.elements.keys()):
            self.elements[key].set_visible(visible)
        self.canvas.draw_idle()

    def terminate_geometries(self):

        for key in list(self.elements.keys()):
            self.elements[key].remove()
            del self.elements[key]

        if self.dataselector.selected_process == '':
            # if hasattr(self, 'location_scatter'):
                # self.elements['scatter'].set_visible(False)
            if hasattr(self, 'arrow'):
                self.arrow.set_visible(False)

            self.mapwidget.set_visible('support_locations')
            self.mapwidget.set_visible('result_locations')

    def set_diffraction_lines(self, bwhead, Lr, Beq, wave_direction):
        """
        Get lines for diffraction
        """
        for row in self.result_locations.itertuples():
            if np.isnan(bwhead[row.Naam]):
                x, y = [], []
            else:
                crd = self.coords[f'bwhead{int(bwhead[row.Naam])}']
                x, y = [row.geometry.x, crd[0]], [row.geometry.y, crd[1]]
            self.elements[row.Naam].set_data(x, y)

        # Plot a line for the representative wave length and equivalent opening width
        lineLr = geometry.extend_point_to_linestring(self.coords['bwhead2'], wave_direction, (-0.5 * Lr, 0.5 * Lr))
        lineBeq = geometry.extend_point_to_linestring(self.coords['bwhead2'], (wave_direction + 90) % 360, (-0.5 * Beq, 0.5 * Beq))

        self.elements['Lr'].set_data(*lineLr.T)
        self.elements['Beq'].set_data(*lineBeq.T)

    def set_transmission_lines(self, wave_direction, freeboard):
        """
        Get lines for transmission
        """
        for i, breakwater in enumerate(self.breakwaters.itertuples()):
            # Determine transmission zones
            leftline, rightline = self.transmission.determine_transmission_zone(breakwater.Index, wave_direction)
            # Set lines
            area = np.vstack([leftline, rightline[::-1]])
            
            self.elements[breakwater.Index].set_xy(area)
            self.elements[f'vb_{breakwater.Index}'].set_text(f'vb: {freeboard[i]:.2f} m')

    def set_fetch_lines(self, direction, lengths):
        """
        Get lines for transmission
        """
        if isinstance(direction, (float, int)):
            raise TypeError()

        for row in self.result_locations.itertuples():
            line = geometry.extend_point_to_linestring(row.geometry, direction[row.Naam], (0, -lengths[row.Naam]))
            self.elements[row.Naam].set_data(*line.T)

    def set_breaking_lines(self, direction):
        """
        Get lines for transmission
        """
        if isinstance(direction, (float, int)):
            direction = {row.Naam: direction for row in self.result_locations.itertuples()}

        for row in self.result_locations.itertuples():
            line = geometry.extend_point_to_linestring(row.geometry, direction[row.Naam], (0, -10000), as_LineString=True)
            line = self.area_union.intersection(line)
            if isinstance(line, LineString):
                self.elements[row.Naam].set_data(*line.coords.xy)
            elif isinstance(line, MultiLineString):
                self.elements[row.Naam].set_data(*line[0].coords.xy)
            else:
                self.elements[row.Naam].set_data([], [])
                    
    def set_location_values(self, values=None, rotation=None):
        """
        Set colors to scatter. Colormap used is RdYlGn_r, is 0.0, the color is grey.
        """
        if self.dataselector.selected_process == '' or self.dataselector.selected_table == '':
            return None

        collection = self.elements['scatter']

        # Adjust colors
        if values is not None:
            self.values = values
            colors = [self.cmap(val) if val > 0.0 else (0.7, 0.7, 0.7, 1.0) for val in values]
            collection.set_facecolors(colors)
            collection.set_edgecolors('0.4')

        if rotation is not None:
            # Convert rotation to (array of) radians
            if isinstance(rotation, (int, float)):
                self.rotations = [np.radians(rotation)] * len(values)
            else:
                self.rotations = [np.radians(rot) for rot in rotation]
        
        # Get scale
        scale = 10 + (self.mapwidget.bbox[2] - self.mapwidget.bbox[0]) / 80
        collection.set_linewidth(self.mapwidget.geometry().width() / 2000)
        
        # Adjust paths for rotation and scale
        paths = []
        for rot, val, xy in zip(self.rotations, self.values, self.resultxy):
            if val > 0.0:
                xcrds, ycrds = geometry.rotate_coordinates((0, 0), rot, *self.markerpath.T)
            else:
                xcrds, ycrds = self.circlepath.T
            paths.append(PathPatch(matplotlib.path.Path(np.array(list(zip(xcrds * scale + xy[0], ycrds * scale + xy[1]))))))
        collection.set_paths(paths)
                    

    def set_load(self):
        """
        Changes the location for which the data is visualized.
        """
        if self.dataselector.selected_process == '':
            return None

        if self.dataselector.hydraulic_load_id is None:
            self.set_visible(False)
            return None

        self.set_visible(True)
        # Get data for load combination and process
        process_data = self._get_process_load_data()
        # Update geometries
        self._update_geometries(process_data)
        
        # Draw
        self.draw_geometries()
        
    def get_background(self):
        """
        Method to save background
        """
        
        for element in self.elements.values():
            element.set_visible(False)
            
        self.canvas.draw()
        self.bg_limits = self.ax.get_xlim() + self.ax.get_ylim()
        self.background = self.canvas.copy_from_bbox(self.ax.bbox)
        self.layer = self.mapwidget.WMTSlayer
                
        for element in self.elements.values():
            element.set_visible(True)
            
    def draw_geometries(self):
        """Update elements and restore background.
        """
        # Check if the bounding box has changed
        if self.bg_limits != (self.ax.get_xlim() + self.ax.get_ylim()) or self.mapwidget.WMTSlayer != self.layer or self.splittermoved:
            self.get_background()
            self.splittermoved = False
        
        # Update
        self.canvas.restore_region(self.background)
        for element in self.elements.values():
            self.ax.draw_artist(element)
        
        self.canvas.blit(self.ax.bbox)
        
    def _get_process_load_data(self):

        table = self.dataselector.tables[self.dataselector.selected_table]
        maxvals = table.max(axis=0)

        # Prepare data
        columns = table.columns.tolist()
        hydraulic_load_id = self.dataselector.hydraulic_load_id
        idx = (self.dataselector.loadidx == hydraulic_load_id)
        selectie = table.loc[idx].reset_index('HydraulicLoadId', drop=True)
        process = self.dataselector.selected_process

        if process == 'Diffractie (Kd)':
            process_data = {
                'wavedir': self.hydraulic_loads.at[hydraulic_load_id, 'Wave direction'],
                'Lr': selectie.iat[0, columns.index('Lr')],
                'Beq': selectie.iat[0, columns.index('Beq')],
                'Kd': selectie['Kd'] / maxvals['Kd'],
                'bwhead': selectie['breakwater'].to_dict(),
                'diffr_angle': (selectie['Diffraction direction'] + 180) % 360
            }
        
        elif process == 'Transmissie (Kt)':
            process_data = {
                'wavedir': self.hydraulic_loads.at[hydraulic_load_id, 'Wave direction'],
                'Kt': selectie['Kt'] / maxvals['Kt'],
                'freeboard': [bw.hoogte - self.hydraulic_loads.at[hydraulic_load_id, self.hydraulic_loads.wlevcol] for bw in self.breakwaters.itertuples()]
            }

        elif process == 'Lokale Golfgroei (Hs,lg)':
            process_data = {
                'winddir': self.hydraulic_loads.at[hydraulic_load_id, 'Wind direction'],
                'wavedir': self.hydraulic_loads.at[hydraulic_load_id, 'Wave direction'],
                'wavedirs': selectie['maxdir'].to_dict(),
                'Hs': selectie['Hs,lg'] / maxvals['Hs,lg'],
                'Feq': selectie['Feq'].to_dict()
            }

        elif process == 'Golfbreking (-)':
            process_data = {
                'wavedir': self.hydraulic_loads.at[hydraulic_load_id, 'Wave direction'],
                'breaking_fraction': selectie['breaking_fraction'] / maxvals['breaking_fraction'],
                'Breaking length': selectie['Breaking length'].to_dict()
            }

        elif process == 'Gecombineerd (Hs)':
            process_data = {
                'wavedir': self.hydraulic_loads.at[hydraulic_load_id, 'Wave direction'],
                'values': selectie['Hs,out'],
                'angles': (selectie['Combined wave direction'] + 180) % 360
            }

        # Advanced method
        else:
            dct = self.dataselector.angle_variables[self.dataselector.selected_table]
            process_data = {
                'wavedir': self.hydraulic_loads.at[hydraulic_load_id, 'Wave direction'],
                'values': selectie[self.dataselector.selected_process] / maxvals[self.dataselector.selected_process],
                'angles': (selectie[dct[self.dataselector.selected_process]] + 180) % 360
            }
        
        return process_data
                
    def _update_geometries(self, process_data):
        
        process = self.dataselector.selected_process
        # Diffraction: Lr, Beq, wave direction, Kd
        if process == 'Diffractie (Kd)':
            self.set_wave_direction(process_data['wavedir'])
            self.set_location_values(process_data['Kd'].sort_index().array, rotation=process_data['diffr_angle'])
            self.set_diffraction_lines(process_data['bwhead'], process_data['Lr'], process_data['Beq'], process_data['wavedir'])

        elif process == 'Transmissie (Kt)':
            self.set_wave_direction(process_data['wavedir'])
            self.set_location_values(process_data['Kt'].sort_index().array, rotation=(process_data['wavedir'] + 180) % 360)
            self.set_transmission_lines(process_data['wavedir'], process_data['freeboard'])

        elif process == 'Lokale Golfgroei (Hs,lg)':
            self.set_wave_direction(process_data['wavedir'])
            self.set_location_values(process_data['Hs'].sort_index().array, rotation=(process_data['winddir'] + 180) % 360)
            self.set_fetch_lines(process_data['wavedirs'], process_data['Feq'])

        elif process == 'Golfbreking (-)':
            self.set_wave_direction(process_data['wavedir'])
            self.set_location_values(process_data['breaking_fraction'], rotation=(process_data['wavedir'] + 180) % 360)
            self.set_breaking_lines(process_data['wavedir'])

        else:
            self.set_wave_direction(process_data['wavedir'])
            self.set_location_values(process_data['values'].sort_index().array, rotation=process_data['angles'].sort_index().array)

    def set_wave_direction(self, wave_direction):

        scale = 20 + (self.mapwidget.bbox[2] - self.mapwidget.bbox[0]) / 50

        supportloc = self.mainmodel.schematisation.support_location['geometry'].coords[0]
        line = geometry.extend_point_to_linestring(supportloc, wave_direction, (-scale, scale))
        if 'arrow' in self.elements:
            self.elements['arrow'].remove()
            
        x0, y0 = line[0]
        dx, dy = (line[1] - line[0])
        self.elements['arrow'] = self.ax.arrow(*line[0], *(line[1] - line[0]), length_includes_head=True, head_width=scale/4, head_length=scale/2, color='k')
    
    def set_parameter(self):
        """
        Sets the parameter for which the data is visualized.
        """
        # Terminate old geometries
        self.terminate_geometries()
        # Initialize geometries
        self.initialize_geometries()
        if not self.dataselector.selected_process == '':
            # Set load
            self.set_load()
        # Draw
        self.draw_geometries()