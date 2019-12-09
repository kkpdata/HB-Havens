# -*- coding: utf-8 -*-
"""
Created on  : 23-11-2017
Author      : Guus Rongen, Johan Ansink, HKV Lijn in Water
Project     : PR3594.10
Description : TabWidget classes

"""

import csv
import itertools
import logging
import os
import time
from datetime import datetime

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from descartes import PolygonPatch
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.lines import Line2D
# Toegevoegd Svasek 31/10/2018 - Ander gebruik van figure, waardoor er in Spyder geen extra figuur opent
from matplotlib.figure import Figure
from matplotlib.patches import Arrow
import matplotlib
from PyQt5 import Qt, QtCore, QtGui, QtWidgets
from shapely.geometry import LineString, Point, Polygon, MultiPolygon

from hbhavens import core, io, ui
from hbhavens.core.general import replace_column
from hbhavens.ui import models as HBHModels
from hbhavens.ui import widgets, threads, dialogs

from hbhavens.ui.tabs.general import SplittedTab

logger = logging.getLogger(__name__)


class WelcomeTab(widgets.AbstractTabWidget):
    """
    Tab widget to with welcome information.
    The user gives a projectname, username and emailadres in this tab.
    """
    def __init__(self, parent=None):
        """
        Constructor of the tab
        """
        # Create child class
        widgets.AbstractTabWidget.__init__(self, parent)

        self.projectSettings = self.project.getGroupSettings('project')

        self.init_ui()

    def init_ui(self):
        """
        Build ui elements
        """
        vbox = QtWidgets.QVBoxLayout()

        # Add text, seperated by spacers
        text = ['Hydraulische Belastingen Havens',
            'Welkom bij de applicatie Hydraulische Belastingen Havens (' + self.mainmodel.appName + ')\nVersie: ' + self.mainmodel.appVersion + '\nDatum: ' + self.mainmodel.appDate,
            'Met deze applicatie kunt u Hydraulische Belastingen bepalen binnen zeehavens.\nDeze software is onderdeel van het WBI 2017. Elke beoordeling die uitgevoerd wordt met hydraulische belastingen afgeleid met deze software, valt in de categorie toets op maat.\nVoor meer informatie wordt verwezen naar de gebruikershandleiding ' + self.mainmodel.appName + '.',
            'Deze applicatie bevat een wizard die u stap voor stap begeleidt in het proces van het bepalen van de Hydarulische Belastingen in (zee)havens.\nVoer eerst uw gebruikersnaam en emailadres in. Uw naam en emailadres worden gelogd bij het uitvoeren van berekeningen.',
            'U kunt starten met de eerste stap uit de wizard (in het geval dat u een nieuwe haven gaat schematiseren), maar u kunt ook een bestaand project inladen en met de knoppen linksonder naar een bepaalde stap navigeren.',]

        for i, t in enumerate(text):
            label = QtWidgets.QLabel()
            label.setText(t)
            label.setWordWrap(True)

            # Title larger
            if i == 0:
                label.setFont(QtGui.QFont('SansSerif', 20))
            vbox.addWidget(label)

        hbox = QtWidgets.QHBoxLayout()
        hbox.addLayout(vbox)

        piclabel = QtWidgets.QLabel()
        logo_path = os.path.join(self.mainmodel.datadir, 'logo_wbi_2017.jpg')
        if not os.path.exists(logo_path):
            raise OSError('Path "{}" not found'.format(logo_path))
        wbi2017pic = QtGui.QPixmap(logo_path)
        piclabel.setPixmap(wbi2017pic)
        piclabel.setFixedWidth(wbi2017pic.width())
        hbox.addWidget(piclabel, 0, QtCore.Qt.AlignTop)
        
        vbox = QtWidgets.QVBoxLayout()
        vbox.addLayout(hbox)

        flo = QtWidgets.QFormLayout()
        flo.setVerticalSpacing(5)
        flo.setSpacing(5)

        self.projectnaam = QtWidgets.QLineEdit()
        self.projectnaam.setObjectName("projectnaam")
        self.projectnaam.setMaximumWidth(300)
        self.projectnaam.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        self.projectnaam.textChanged[str].connect(self.textChangedProjectName)
        flo.addRow('Projectnaam:', self.projectnaam)

        self.gebruikersnaam = QtWidgets.QLineEdit()
        self.gebruikersnaam.setObjectName("gebruikersnaam")
        self.gebruikersnaam.setMaximumWidth(300)
        self.gebruikersnaam.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        self.gebruikersnaam.textChanged[str].connect(self.textChangedName)
        flo.addRow('Gebruikersnaam:', self.gebruikersnaam)

        self.emailadres = QtWidgets.QLineEdit()
        self.emailadres.setObjectName("emailadres")
        self.emailadres.setMaximumWidth(300)
        self.emailadres.textChanged[str].connect(self.textChangedEmail)
        flo.addRow('Emailadres:', self.emailadres)

        vbox.addItem(flo)
        vbox.addItem(QtWidgets.QSpacerItem(200, 100, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding))

        vbox.setSpacing(20)

        self.setLayout(vbox)

        # set data
        self.projectnaam.setText(self.projectSettings['name'])
        self.gebruikersnaam.setText(self.projectSettings['user']['name'])
        self.emailadres.setText(self.projectSettings['user']['email'])

    def on_focus(self):
        self.check_done()

    def check_done(self):
        """
        Check if all lines are filled, and we may continue
        """
        if self.projectnaam.text() and self.gebruikersnaam.text() and self.emailadres.text():
            self.set_finished(True)
        else:
            self.set_finished(False)


    def textChangedProjectName(self):
        if self.projectnaam.isModified():
            self.projectSettings['name'] = self.projectnaam.text()
            self.mainwindow.setDirty()
            self.check_done()

    def textChangedName(self):
        if self.gebruikersnaam.isModified():
            if 'user' in self.projectSettings:
                self.projectSettings['user']['name'] = self.gebruikersnaam.text()
            self.mainwindow.setDirty()
            self.check_done()

    def textChangedEmail(self, text):
        if self.emailadres.isModified():
            if 'user' in self.projectSettings:
                self.projectSettings['user']['email'] = self.emailadres.text()
            self.mainwindow.setDirty()
            self.check_done()

                    
class SchematisationTab(widgets.AbstractTabWidget):
    """
    Tab widget with the schematisation input
    """
    def __init__(self, parent=None):
        """
        Constructor of the tab
        """
        # Create child class
        widgets.AbstractTabWidget.__init__(self, parent)

        # Pointers to elements in main
        self.overview_tab = parent.tabwidgets['Overzicht']
        self.tabwidgets = parent.tabwidgets

        # Construct the tab
        self.init_ui()

        # Fill the ui from project
        self.openData()

    def init_ui(self):
        """
        Constructor for the tab widget
        """
        # Harbor geometry frame
        #----------------------------------------------------------------
        harborgeoframe = QtWidgets.QGroupBox()
        harborgeoframe.setTitle('1. Havengeometrie')

        self.paths = {}

        vlayout = QtWidgets.QVBoxLayout()

        # Adding dike sections
        hlayout = QtWidgets.QHBoxLayout()
        hlayout.setSpacing(10)
        hlayout.setContentsMargins(5, 0, 5, 0)
        label = QtWidgets.QLabel('Normtraject(en):')
        label.setFixedWidth(150)
        hlayout.addWidget(label)
        self.section_label = QtWidgets.QLabel('')

        hlayout.addWidget(self.section_label)
        self.set_flood_defence_ids()
        self.add_section_button = QtWidgets.QPushButton('+', clicked=self._add_flooddefence)
        self.add_section_button.setFixedWidth(25)
        self.del_section_button = QtWidgets.QPushButton('-', clicked=self._del_flooddefence)
        self.del_section_button.setFixedWidth(25)
        hlayout.addWidget(self.add_section_button)
        hlayout.addWidget(self.del_section_button)

        hlayout.addItem(QtWidgets.QSpacerItem(0, 0, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Maximum))
        vlayout.addLayout(hlayout)

        # Harbor area
        self.harborareabrowse = widgets.ExtendedLineEdit(
            label='Haventerrein:',
            labelwidth=150,
            browsebutton=QtWidgets.QPushButton('...', clicked=self._load_harborarea)
        )
        vlayout.addWidget(self.harborareabrowse)

        # Break waters
        self.breakwaterbrowse = widgets.ExtendedLineEdit(
            label='Havendammen:',
            labelwidth=150,
            browsebutton=QtWidgets.QPushButton('...', clicked=self._load_breakwater)
        )
        vlayout.addWidget(self.breakwaterbrowse)

        # Harbor entrance
        #----------------
        self.entrance_layout = QtWidgets.QHBoxLayout()
        self.entrance_layout.setSpacing(10)
        self.entrance_layout.setContentsMargins(5, 0, 5, 0)
        # Description
        label = QtWidgets.QLabel('Coordinaat haveningang:')
        label.setFixedWidth(150)
        self.entrance_layout.addWidget(label)
        # Coordinate label
        self.entrance_layout.addWidget(QtWidgets.QLabel('x;y ='))
        # Coordinate lineedit
        self.coord = QtWidgets.QLineEdit()
        self.coord.setFixedWidth(150)
        regexp = QtCore.QRegExp("^\d+(?:[\.\,]\d+)?[;]\d+(?:[\.\,]\d+)?$")
        self.coord.setValidator(QtGui.QRegExpValidator(regexp))
        self.coord.editingFinished.connect(self._parse_input_coord)
        self.entrance_layout.addWidget(self.coord)

        self.entrance_layout.addItem(QtWidgets.QSpacerItem(0, 0, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Maximum))

        self.enable_elements(self.entrance_layout, False)

        vlayout.addLayout(self.entrance_layout)


        self.repbottomlevel = widgets.ParameterInputLine('Representatieve bodemligging:', 150, unitlabel='m+NAP')
        self.repbottomlevel.LineEdit.editingFinished.connect(self._check_rep_bottom)
        regexp = QtCore.QRegExp("^-?\d+(?:[\.\,]\d+)?$")
        self.repbottomlevel.LineEdit.setValidator(QtGui.QRegExpValidator(regexp))
        vlayout.addWidget(self.repbottomlevel)

        harborgeoframe.setLayout(vlayout)

        vbox = QtWidgets.QVBoxLayout()

        vbox.addWidget(harborgeoframe)

        # Hydraulic Loads frame
        #----------------------------------------------------------------
        hydraulicloadframe = QtWidgets.QGroupBox()
        hydraulicloadframe.setTitle('2. Hydraulische Belastingen')
        vlayout = QtWidgets.QVBoxLayout()

        self.hrdbrowse = widgets.ExtendedLineEdit(label='HRD:',
            labelwidth=150,
            browsebutton=QtWidgets.QPushButton('...', clicked=self._load_HRD))
        vlayout.addWidget(self.hrdbrowse)

        hbox = QtWidgets.QHBoxLayout()
        label = QtWidgets.QLabel()
        label.setText('Steunpuntlocatie:')
        label.setFixedWidth(150)
        hbox.addWidget(label)
        hbox.setSpacing(10)
        hbox.setContentsMargins(5, 0, 5, 0)
        self.combobox = QtWidgets.QComboBox()
        self.combobox.currentIndexChanged.connect(self._support_location_selected)
        hbox.addWidget(self.combobox)
        vlayout.addLayout(hbox)

        hydraulicloadframe.setLayout(vlayout)

        vbox.addWidget(hydraulicloadframe)

        # HRD result locations
        #----------------------------------------------------------------
        hrdlocationsframe = QtWidgets.QGroupBox()
        hrdlocationsframe.setTitle('3. Uitvoerlocaties')
        vlayout = QtWidgets.QVBoxLayout()

        self.gen_hr_locations_button = QtWidgets.QPushButton('Genereer uitvoerlocaties')
        self.gen_hr_locations_button.setFixedWidth(150)

        self.gen_hr_locations_button.clicked.connect(self._generate_locations_clicked)

        vlayout.addWidget(self.gen_hr_locations_button)

        self.result_location_browse = widgets.ExtendedLineEdit(
            label='Uitvoerlocaties:',
            labelwidth=150,
            browsebutton=QtWidgets.QPushButton(
                '...',
                clicked=self._load_result_locations
                )
            )
        vlayout.addWidget(self.result_location_browse)

        hrdlocationsframe.setLayout(vlayout)

        vbox.addWidget(hrdlocationsframe)

        # Add spacer to fill the bottom
        vbox.addItem(QtWidgets.QSpacerItem(200, 50, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding))

        self.setLayout(vbox)

    def on_focus(self):
        """
        On focus, check if:
            - The simple calculation has been carried out
            - The SWAN input files are generated for the first iteration step
        If so, enable warning the user when something is changed.
        """

        simple_check = self.settings['simple']['finished']
        advanced_check = self.settings['swan']['calculations']['I1']['input_generated']

        if simple_check or advanced_check:
            self.ask_to_proceed = True
        else:
            self.ask_to_proceed = False

    def may_proceed(self):
        """
        Notify the user that further progress has been made, and ask permission to continue
        """
        proceed = True
        if self.ask_to_proceed == True:
            proceed = dialogs.QuestionDialog.question(
                self,
                self.mainmodel.appName,
                'Er zijn al vervolgstappen gezet met de huidige schematisatie. Door de schematisatie aan te passen worden bepaalde stappen opnieuw geïnitialiseerd, waarmee de voortgang ongedaan wordt gemaakt.\n\nWeet u zeker dat u de schematisatie wilt aanpassen?'
            )

            # We check only once
            self.ask_to_proceed = False

        return proceed

    def openData(self):
        """
        Fill line edits etcetera with data from project file
        """
        # Set data
        if self.settings['schematisation']['flooddefence_ids']:
            self.section_label.setText(', '.join(self.settings['schematisation']['flooddefence_ids']))

        if self.settings['schematisation']['harbor_area_shape']:
            self.harborareabrowse.LineEdit.setText(self.settings['schematisation']['harbor_area_shape'])

        if self.settings['schematisation']['breakwater_shape']:
            self.breakwaterbrowse.LineEdit.setText(self.settings['schematisation']['breakwater_shape'])
            if len(self.schematisation.breakwaters) == 1:
                self.enable_elements(self.entrance_layout, True)

        if self.settings['schematisation']['entrance_coordinate']:
            self.coord.setText('{};{}'.format(*self.settings['schematisation']['entrance_coordinate']))

        if self.settings['schematisation']['representative_bedlevel']:
            self.repbottomlevel.set_value(str(self.settings['schematisation']['representative_bedlevel']))

        if self.settings['hydraulic_loads']['HRD']:
            self.fill_supportloc_combobox()
            self.hrdbrowse.LineEdit.setText(self.settings['hydraulic_loads']['HRD'])

        if self.settings['schematisation']['support_location_name']:
            self.combobox.currentIndexChanged.disconnect()
            self.combobox.setCurrentText(self.settings['schematisation']['support_location_name'])
            self.combobox.currentIndexChanged.connect(self._support_location_selected)


        if self.settings['schematisation']['result_locations_shape']:
            self.result_location_browse.set_value(self.settings['schematisation']['result_locations_shape'])
            self.set_finished(True)


    def enable_elements(self, layout, boolean):
        for i in range(layout.count()):
            item = layout.itemAt(i)
            if item.widget():
                item.widget().setEnabled(boolean)

    def _check_rep_bottom(self):
        """
        Method exectued when editing is finished for the rep bottom level

        1. Read input from line edit
        2. Replace comma with dot if necessary
        3. Check if the given bottom level is lower than the lowest harbor terrain
        4. If no errors, save
        """

        # Get the old bottom level from the settings
        old_level = str(self.settings['schematisation']['representative_bedlevel'])

        # Read from line edit
        rep_bottom_input = self.repbottomlevel.LineEdit.text()

        # Ask if the use may proceed
        if rep_bottom_input != old_level:
            if not self.may_proceed():
                self.repbottomlevel.LineEdit.setText(old_level)
                return None
        # Replace comma with dot if presenct
        rep_bottom_input = float(rep_bottom_input.replace(',', '.'))

        valid = self.schematisation.check_bedlevel(rep_bottom_input)
        if not valid:
            # Reset and pass error
            self.repbottomlevel.LineEdit.setText(old_level)
            raise ValueError('Opgegeven bodemniveau hoger dan het laagste haventerrein. Kies een lager niveau, of pas de hoogte van het haventerrein aan.')
        else:
            # If no errors
            self.settings['schematisation']['representative_bedlevel'] = rep_bottom_input
            self.mainwindow.setDirty()

    def _parse_input_coord(self):
        """Parse the input coordinate for the harbor entrance"""
        # Read from line edit
        crd = tuple([float(i) for i in self.coord.text().replace(',', '.').split(';')])
        # Add to schematisation
        valid = self.schematisation.check_entrance_coordinate(crd)
        # Check if succeeded
        if valid:
            self.schematisation.entrance_coordinate = crd
            self.settings['schematisation']['entrance_coordinate'] = crd
            self.schematisation.generate_harbor_bound()
            self.overview_tab.mapwidget.set_visible('inner')
            self.overview_tab.mapwidget.set_visible('entrance')
            self.mainwindow.setDirty()
        else:
            self.coord.setText('')
            raise ValueError('Opgegeven ligt niet binnen het haventerrein. Geef een ander coordinaat, of pas het haventerrein aan.')

    def _load_rep_bottom(self):
        """
        Load representative bottom level from saved project.
        """
        # Read from settings
        rep_bottom_input = self.settings['schematisation']['representative_bedlevel']
        # Set to line edit
        self.repbottomlevel.set_value(str(rep_bottom_input))
        
    def _load_result_locations(self, path=None):
        """
        Method to select file path for harbor HRD locations.
        Also the plotting is called from here.
        """
        # Ask to proceed
        if not self.may_proceed():
            return None

        # Get path with dialog
        if not path:
            path = QtWidgets.QFileDialog.getOpenFileName(self.result_location_browse.LineEdit, 'Open shapefile met uitvoerlocaties', '', "Shape file (*.shp)")[0]
        if not path:
            return None
        # Save path to project structure
        self.settings['schematisation']['result_locations_shape'] = path
        self.mainwindow.setDirty()

        # Add text to line edit
        self.result_location_browse.set_value(path)

        # Load
        self.schematisation.add_result_locations(path)
        self.overview_tab.mapwidget.set_visible('result_locations')

        self.set_finished(True)

    def _generate_locations_clicked(self):
        """
        Class method executed on press generate locations button press.
        Opens a window with which locations can be generated.
        """
        # Ask to proceed
        if not self.may_proceed():
            return None

        self.generate_window = GenerateHarborLocationsWindow(self)
        self.generate_window.exec_()
        # Add to schematisation tab
        if self.generate_window.succeeded:
            self.result_location_browse.LineEdit.setText(self.generate_window.path)
            self._load_result_locations(path=self.generate_window.path)

    def _support_location_selected(self):
        """
        Class method executed on press generate locations button press.
        Opens a window with which locations can be generated.
        """
        text = self.combobox.currentText()
        if not text:
            return None

        if not np.size(self.schematisation.support_locations):
            NotificationDialog('Er is nog geen HRD ingeladen, doe dit eerst.', severity='warning')
            return None
        
        # Check if the selected support location is equal to the location
        # in the settings
        if self.settings['schematisation']['support_location_name']:
            # If the location is different from the already chosen location
            if text != self.settings['schematisation']['support_location_name']:
                # Check if the user really wants to adjust the location
                change_location = self.may_proceed()
            # If equal, do nothing
            else:
                change_location = False
        else:
            # As long as the location is unselected, do nothing
            change_location = True

        if change_location:
            # Set or change support location
            self.schematisation.set_selected_support_location(text)
            # Notify next tabs that the location had changed, and the initialization should be done again
            self.swan.location_changed = True
            # Set project dirty
            self.mainwindow.setDirty()
            # Visualize in overview
            self.overview_tab.mapwidget.set_visible('support_location')
        else:
            # Reset the combobox
            self.combobox.setCurrentText(self.settings['schematisation']['support_location_name'])

    def _load_harborarea(self):
        """
        Method to import file path for harbor area and add it to the
        harbor area class. Also the plotting is called from here.
        """
        if not self.may_proceed():
            return None

        # Get path with dialog
        path = QtWidgets.QFileDialog.getOpenFileName(self.harborareabrowse.LineEdit, 'Open shapefile met haventerrein', '', "Shape file (*.shp)")[0]
        if not path:
            return None
        try:
            # Call the schematisation class method to load geometry
            self.schematisation.add_harborarea(path)
            
            self.mainwindow.setDirty()
            # Visualize
            self.overview_tab.mapwidget.set_visible('harborarea')
            # Adjust line edit
            self.harborareabrowse.LineEdit.setText(path)
        except Exception as e:
            print(e)

    def _load_breakwater(self):
        """
        Method to import file path for breakwater and add it to the
        harbor area class. Also the plotting is called from here.
        """
        # Ask if the use may proceed
        if not self.may_proceed():
            return None

        # Get path with dialog
        path = QtWidgets.QFileDialog.getOpenFileName(self.breakwaterbrowse.LineEdit, 'Open shapefile met havendammen', '', "Shape file (*.shp)")[0]
        if not path:
            return None
        # Try to call schematisation class method to load breakwaters
        try:
            self.schematisation.add_breakwater(path)
            self.overview_tab.mapwidget.set_visible('breakwaters')
            
            self.mainwindow.setDirty()
            # Adjust lineedit
            self.breakwaterbrowse.LineEdit.setText(path)
            # Enable extra input coordinate if 1 breakwater
            if len(self.schematisation.breakwaters) == 1:
                self.enable_elements(self.entrance_layout, True)
            else:
                self.enable_elements(self.entrance_layout, False)
                self.overview_tab.mapwidget.set_visible('inner')
                self.overview_tab.mapwidget.set_visible('entrance')
        except Exception as e:
            print(e)

    def _add_flooddefence(self):
        """
        Method to add flood defence
        """
        self.flooddefence_window = dialogs.ChooseFloodDefenceWindow(self)
        self.flooddefence_window.exec_()

    def _del_flooddefence(self):
        """
        Method to remove flood defence
        """
        self.flooddefence_window = dialogs.RemoveFloodDefenceWindow(self)
        self.flooddefence_window.exec_()

    def set_flood_defence_ids(self):
        """
        Update flood defence id tabwidgets
        """
        flooddefence_ids = self.settings['schematisation']['flooddefence_ids']
        if flooddefence_ids:
            self.section_label.setText(', '.join(flooddefence_ids))
        else:
            self.section_label.setText('[voeg een normtraject toe]')
        self.mainwindow.setDirty()


    def _load_HRD(self):
        """
        Method to load a HRD database
        """
        # Get path with dialog
        path = QtWidgets.QFileDialog.getOpenFileName(
            self.hrdbrowse.LineEdit,
            'Open SQLite-database met hydraulische randvoorwaarden (HRD)',
            '',
            "SQLite database (*.sqlite)"
        )[0]
        
        if not path:
            return None

        if self.settings['hydraulic_loads']['HRD'] != '':
            # If the location is different from the already chosen location
            if path != self.settings['hydraulic_loads']['HRD']:
                # Ask to proceed
                change_hrd = self.may_proceed()
            else:
                change_hrd = False
        else:
            change_hrd = True

        if change_hrd:
            # Add to schematisation
            self.hydraulic_loads.add_HRD(path)
            # Set project dirty
            self.mainwindow.setDirty()
            # Plot
            self.overview_tab.mapwidget.set_visible('support_locations')
            # Adjust line edit
            self.hrdbrowse.set_value(path)
            # Add support locations to combobox
            self.fill_supportloc_combobox()
            self.settings['schematisation']['support_location_name'] = ''

    def fill_supportloc_combobox(self):
        """
        Fill combobox with names
        """
        if not np.size(self.schematisation.support_locations):
            return None

        # Get support locations
        support_location_names = [''] + self.schematisation.support_locations['Name'].values.tolist()
        self.combobox.clear()
        self.combobox.addItems(support_location_names)
        self.combobox.setCurrentText('')

class CalculationMethodTab(widgets.AbstractTabWidget):
    """
    Class to choose calculation method. The next steps in the process are
    determined according to the chosen method
    """
    def __init__(self, parent=None):
        """
        Constructor of the tab
        """
        # Create child class
        widgets.AbstractTabWidget.__init__(self, parent)
        
        # Block signals, they are enabled again after loading the settings
        self.blockSignals(True)
        
        # build UI
        self.init_ui()

        # load from settings
        self.load_from_settings()

        # Allow continuing to the next step
        self.set_finished(True)

        # Enable connections again
        self.blockSignals(False)


    def init_ui(self):
        """
        Constuct tab.
        """

        # Method Choice frame
        #----------------------------------------------------------------
        # Breakwaters
        condition_geometry_frame = QtWidgets.QGroupBox('1. Is er sprake van een convex enkelvoudig havenbassin met maximaal 2 havendammen')
        condition_geometry_frame.setLayout(QtWidgets.QVBoxLayout())
        self.condition_geometry_yesbutton = QtWidgets.QRadioButton('Ja', toggled=self.conditionGeometrystate)
        self.condition_geometry_nobutton = QtWidgets.QRadioButton('Nee', toggled=self.conditionGeometrystate)
        condition_geometry_frame.layout().addWidget(self.condition_geometry_yesbutton)
        condition_geometry_frame.layout().addWidget(self.condition_geometry_nobutton)
        # Reflection
        condition_reflection_frame = QtWidgets.QGroupBox('2. Treedt significante reflectie op in de haven?')
        condition_reflection_frame.setLayout(QtWidgets.QVBoxLayout())
        self.condition_reflection_yesbutton = QtWidgets.QRadioButton('Ja', toggled=self.conditionReflectionstate)
        self.condition_reflection_nobutton = QtWidgets.QRadioButton('Nee', toggled=self.conditionReflectionstate)
        condition_reflection_frame.layout().addWidget(self.condition_reflection_yesbutton)
        condition_reflection_frame.layout().addWidget(self.condition_reflection_nobutton)
        # Current
        condition_current_frame = QtWidgets.QGroupBox('3. Treedt significante stroming op in de haven?')
        condition_current_frame.setLayout(QtWidgets.QVBoxLayout())
        self.condition_current_yesbutton = QtWidgets.QRadioButton('Ja', toggled=self.conditionCurrentstate)
        self.condition_current_nobutton = QtWidgets.QRadioButton('Nee', toggled=self.conditionCurrentstate)
        condition_current_frame.layout().addWidget(self.condition_current_yesbutton)
        condition_current_frame.layout().addWidget(self.condition_current_nobutton)

        choicebox = QtWidgets.QWidget()
        choicebox.setLayout(QtWidgets.QHBoxLayout())

        self.simple_button = QtWidgets.QPushButton('Eenvoudige methode', clicked=self.handleSimpleButton)
        self.simple_button.setFixedWidth(150)
        self.simple_button.setFixedHeight(150)
        self.simple_button.setCheckable(True)

        self.advanced_button = QtWidgets.QPushButton('Geavanceerde methode', clicked=self.handleAdvancedButton)
        self.advanced_button.setFixedWidth(150)
        self.advanced_button.setFixedHeight(150)
        self.advanced_button.setCheckable(True)

        self.swan_checkbox = QtWidgets.QRadioButton('SWAN')
        self.swan_checkbox.toggled.connect(self.enable_swan)
        self.swan_checkbox.setFixedWidth(120)
        
        # Aangepast Svasek 03/10/18 - Pharos optie omgezet naar een Radiobutton
        self.pharos_checkbox = QtWidgets.QRadioButton('SWAN + PHAROS')
        self.pharos_checkbox.toggled.connect(self.enable_pharos)
        self.pharos_checkbox.setFixedWidth(120)
        
        # Toegevoegd Svasek 03/10/18 - Hares optie toegevoegd
        self.hares_checkbox = QtWidgets.QRadioButton('SWAN + HARES')
        self.hares_checkbox.toggled.connect(self.enable_hares)
        self.hares_checkbox.setFixedWidth(120)

        adv_layout = QtWidgets.QHBoxLayout()
        adv_layout.addWidget(self.advanced_button)
        cb_layout = QtWidgets.QVBoxLayout()
        cb_layout.addWidget(self.swan_checkbox)
        cb_layout.addWidget(self.pharos_checkbox)
        # Toegevoegd Svasek 03/10/18 - Hares knop aan de layout toegevoegd
        cb_layout.addWidget(self.hares_checkbox)
        
        cb_layout.addItem(QtWidgets.QSpacerItem(0, 0, QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Expanding))
        adv_layout.addLayout(cb_layout)

        choicebox.layout().addWidget(self.simple_button)
        choicebox.layout().addLayout(adv_layout)
        choicebox.setFixedHeight(170)
        # adv_layout.setAlignment(QtCore.Qt.AlignRight)

        motivationframe = QtWidgets.QGroupBox('Overweging')

        self.motivationtext = QtWidgets.QTextEdit()
        self.motivationtext.textChanged.connect(self.motivationtextChanged)

        motivationbox = QtWidgets.QVBoxLayout()
        motivationbox.addWidget(self.motivationtext)
        # motivationbox.addStretch(1)
        motivationframe.setLayout(motivationbox)

        pagevbox = QtWidgets.QVBoxLayout()

        pagevbox.addWidget(condition_geometry_frame)
        pagevbox.addWidget(condition_reflection_frame)
        pagevbox.addWidget(condition_current_frame)
        pagevbox.addWidget(choicebox)
        pagevbox.addWidget(motivationframe)

        self.setLayout(pagevbox)
        
    def load_from_settings(self):
        """
        Set project settings to UI
        """

        # Check SWAN and disable
        if not self.settings['calculation_method']['include_hares'] and not self.settings['calculation_method']['include_pharos']:
            self.swan_checkbox.setChecked(True)

        # Check PHAROS
        if self.settings['calculation_method']['include_hares'] and self.settings['calculation_method']['include_pharos']:
            raise ValueError('Both PHAROS and HARES are enabled in the settings. Choose one.')
    
        self.pharos_checkbox.setChecked(self.settings['calculation_method']['include_pharos'])
        self.hares_checkbox.setChecked(self.settings['calculation_method']['include_hares'])

        # Set motivation text        
        self.motivationtext.setText(self.settings['calculation_method']['motivation'])
        
        # Toggle radio buttons
        if self.settings['calculation_method']['condition_geometry']:
            self.condition_geometry_yesbutton.setChecked(True)
        else:
            self.condition_geometry_nobutton.setChecked(True)

        if self.settings['calculation_method']['condition_reflection']:
            self.condition_reflection_yesbutton.setChecked(True)
        else:
            self.condition_reflection_nobutton.setChecked(True)

        if self.settings['calculation_method']['condition_flow']:
            self.condition_current_yesbutton.setChecked(True)
        else:
            self.condition_current_nobutton.setChecked(True)

        # Set buttons based on method
        if self.settings['calculation_method']['method'] == 'advanced':
            self.handleAdvancedButton()
        elif self.settings['calculation_method']['method'] == 'simple':
            self.handleSimpleButton()

    def motivationtextChanged(self):
        """
        Change motivation text in settings on changed
        """
        self.settings['calculation_method']['motivation'] = self.motivationtext.toPlainText()
        self.mainwindow.setDirty()

    # Aangepast Svasek 03/10/18 - uitvinken van Hares toegevoegd
    def handleSimpleButton(self):

        self.simple_button.setChecked(True)
        self.advanced_button.setChecked(False)
        self.mainwindow.set_calculation_method('simple')
        self.settings['calculation_method']['method'] = 'simple'
        self.swan_checkbox.setEnabled(False)
        self.swan_checkbox.setChecked(False)
        self.pharos_checkbox.setEnabled(False)
        self.pharos_checkbox.setChecked(False)
        self.hares_checkbox.setEnabled(False)
        self.hares_checkbox.setChecked(False)
        self.mainwindow.setDirty()

    def enable_swan(self, state):
        if self.signalsBlocked():
            return None
        self.settings['calculation_method']['include_pharos'] = False
        self.settings['calculation_method']['include_hares'] = False
        self.mainwindow.set_calculation_method('advanced')
        self.mainwindow.setDirty()


    # Aangepast Svasek 03/10/18 - Uitvinken van Hares toegevoegd
    def enable_pharos(self, state):
        if self.signalsBlocked():
            return None
        self.settings['calculation_method']['include_pharos'] = True
        self.settings['calculation_method']['include_hares'] = False
        self.mainwindow.set_calculation_method('advanced')
        self.mainwindow.setDirty()

    # Toegevoegd Svasek 03/10/18 - Optie om Hares te selecteren toegevoegd
    def enable_hares(self, state):
        if self.signalsBlocked():
            return None
        self.settings['calculation_method']['include_hares'] = True
        self.settings['calculation_method']['include_pharos'] = False
        self.mainwindow.set_calculation_method('advanced')
        self.mainwindow.setDirty()

    def handleAdvancedButton(self):
        self.advanced_button.setChecked(True)
        self.simple_button.setChecked(False)
        self.swan_checkbox.setEnabled(True)
        self.pharos_checkbox.setEnabled(True)
        
        # Toegevoegd Svasek 03/10/18 - Optie om Hares te selecteren toegevoegd
        self.hares_checkbox.setEnabled(True)
        
        self.mainwindow.set_calculation_method('advanced')
        self.settings['calculation_method']['method'] = 'advanced'
        self.mainwindow.setDirty()

    def conditionGeometrystate(self):
        """ Check state and change conditionGeometry accordingly"""
        if self.condition_geometry_yesbutton.isChecked():
            self.settings['calculation_method']['condition_geometry'] = True
        else:
            self.settings['calculation_method']['condition_geometry'] = False
        self.mainwindow.setDirty()
        self.toggle_button()

    def conditionReflectionstate(self):
        """ Check state and change conditionReflection accordingly"""
        if self.condition_reflection_yesbutton.isChecked():
            self.settings['calculation_method']['condition_reflection'] = True
        else:
            self.settings['calculation_method']['condition_reflection'] = False
        self.mainwindow.setDirty()
        self.toggle_button()

    def conditionCurrentstate(self):
        """ Check state and change conditionCurrent accordingly"""
        if self.condition_current_yesbutton.isChecked():
            self.settings['calculation_method']['condition_flow'] = True
        else:
            self.settings['calculation_method']['condition_flow'] = False
        self.mainwindow.setDirty()
        self.toggle_button()

    def toggle_button(self):
        """
        Toggle button based on conditionCurrentstate
        """
        if self.signalsBlocked():
            return None

        if (
            self.settings['calculation_method']['condition_geometry'] and
            not self.settings['calculation_method']['condition_reflection'] and
            not self.settings['calculation_method']['condition_flow']
        ):
            self.handleSimpleButton()
        else:
            self.handleAdvancedButton()

class GenerateHarborLocationsWindow(QtWidgets.QDialog):
    """
    Dialog window class to generate result location inside
    the harbor. This class requests the input needed to generate
    the locations. The generation itself is done in the
    schematisation class.
    """
    def __init__(self, parent=None):
        """
        Constructor of the window
        """
        super(GenerateHarborLocationsWindow, self).__init__(parent)
        self.schematisation = parent.schematisation
        self.connectedLineEdit = parent.result_location_browse.LineEdit
        self._initUI()
        self.succeeded = False

    def _initUI(self):
        """
        Set up UI design
        """

        self.setLayout(QtWidgets.QVBoxLayout())

        regexp = QtCore.QRegExp("^\d+(?:[\.\,]\d+)?$")
        # Distance to crest
        self.dist_to_crest = widgets.ParameterInputLine('Afstand tot kruinlijn:', 150, unitlabel='m')
        self.dist_to_crest.LineEdit.setValidator(QtGui.QRegExpValidator(regexp))
        self.layout().addWidget(self.dist_to_crest)

        # Distance along crest
        self.dist_along_crest = widgets.ParameterInputLine('Volgafstand:', 150, unitlabel='m')
        self.dist_along_crest.LineEdit.setValidator(QtGui.QRegExpValidator(regexp))
        self.layout().addWidget(self.dist_along_crest)

        # Length over which the normal is calculated
        self.midlength = widgets.ParameterInputLine('Middellengte:', 150, unitlabel='m')
        self.midlength.LineEdit.setText('20')
        self.midlength.LineEdit.setValidator(QtGui.QRegExpValidator(regexp))

        self.layout().addWidget(self.midlength)

        # Button to generate locations
        buttonbox = QtWidgets.QHBoxLayout()
        self.generate_button = QtWidgets.QPushButton('Genereer', clicked=self._call_generate_function)
        buttonbox.addWidget(self.generate_button)

        self.savebutton = QtWidgets.QPushButton('Opslaan als', clicked=self._save_harbor_locations)
        self.savebutton.setEnabled(False)
        buttonbox.addWidget(self.savebutton)

        self.layout().addLayout(buttonbox)

        line = QtWidgets.QFrame()
        line.setFrameShape(QtWidgets.QFrame.HLine)
        line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.layout().addWidget(line)

        self.closebutton = QtWidgets.QPushButton('Sluiten', clicked=self.close)
        self.layout().addWidget(self.closebutton, 0, QtCore.Qt.AlignRight)

        self.setWindowTitle('HB Havens: Uitvoerlocaties')

    def _save_harbor_locations(self):
        """
        Save generated locations to file.
        1. Retrieves path with file dialog
        2. Checks if path already exists, if not: saved and added to lineedit.
            If is does, not saved and not added.
        """
        # Get save path
        path = QtWidgets.QFileDialog.getSaveFileName(None, 'Save file', '', "Shape file (*.shp)")[0]
        if not path:
            return None
        else:
            self.path = path

        # Check timedifference to check modified
        if os.path.exists(self.path):
            olddate = datetime.fromtimestamp(os.path.getmtime(self.path))
            already_existed = True
        else:
            already_existed = False

        # Save files
        self.schematisation.result_locations.to_file(self.path)

        if os.path.exists(self.path):
            # Check new date
            newdate = datetime.fromtimestamp(os.path.getmtime(self.path))
            # Check if modified
            if already_existed:
                if newdate > olddate:
                    self.succeeded = True
            else:
                self.succeeded = True


    def _call_generate_function(self):
        """
        Method to call the input parser, and generate the locations
        """
        distance = float(self.dist_to_crest.get_value())
        interval = float(self.dist_along_crest.get_value())
        interp_length = float(self.midlength.get_value())

        self.schematisation.generate_result_locations(distance, interval, interp_length)
        self.savebutton.setEnabled(True)

def ConfirmOverwrite(self, path):
    """
    Function to confirm overwrite when saving file.

    Parameters
    ----------
    path : str
        Save path
    """

    confirmation = QtWidgets.QMessageBox.question(self,
            'Bevestig Opslaan Als',
            '{} bestaat al.\nWilt u het bestand overschrijven?'.format(os.path.basename(path)),
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)

    self.overwrite = (confirmation == QtWidgets.QMessageBox.Yes)


class SimpleCalculationTab(widgets.AbstractTabWidget):
    """
    Tab for the simple calculation
    The user can pick some settings (physical processes to take into account)
    and run the calculation.
    """
    def __init__(self, parent=None):
        """
        Constructor of the tab
        """
        # Create child class
        widgets.AbstractTabWidget.__init__(self, parent)

        # Allow continuing to the next step
        self.set_finished(False)
        
        self.calculation = self.simple_calculation
        self.processes = []
        self.thread = threads.SimpleCalculationThread(self)
        self.initTab()

    def initTab(self):

        self.setLayout(QtWidgets.QVBoxLayout())
        self.layout().setSpacing(10)

        # Settings groupbox
        #-----------------------------------------------------------------------
        settings_groupbox = QtWidgets.QGroupBox()
        settings_groupbox.setTitle('Instellingen')
        settings_groupbox.setLayout(QtWidgets.QVBoxLayout())
        settings_groupbox.layout().addWidget(QtWidgets.QLabel('Selecteer de fysische process die meegenomen moeten worden:'))

        self.checkboxes = {}
        for process in ['Diffractie', 'Transmissie', 'Lokale golfgroei', 'Golfbreking']:
            checkbox = QtWidgets.QCheckBox(process)
            self.checkboxes[process] = checkbox
            settings_groupbox.layout().addWidget(checkbox)

        self.checkboxes['Lokale golfgroei'].setEnabled(False)

        self.layout().addWidget(settings_groupbox)

        # Calculation groupbox
        #-----------------------------------------------------------------------
        calculation_groupbox = QtWidgets.QGroupBox()
        calculation_groupbox.setTitle('Rekenen')
        calculation_groupbox.setLayout(QtWidgets.QVBoxLayout())

        # Start calculation button
        self.start_calc_button = QtWidgets.QPushButton('Start', clicked=self.start_calc)
        self.start_calc_button.setFocus()
        hbox = QtWidgets.QHBoxLayout()
        hbox.addWidget(self.start_calc_button)

        # Progress bar
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setRange(0, 11)
        hbox.addWidget(self.progress_bar)

        calculation_groupbox.layout().addLayout(hbox)

        # Porgress message
        self.progress_message = QtWidgets.QLabel('Druk op start om de berekening te starten.')
        calculation_groupbox.layout().addWidget(self.progress_message)

        self.layout().addWidget(calculation_groupbox)

        # Add export fetch lengths button
        self.export_fetch_button = QtWidgets.QPushButton('Exporteer strijklengtes', clicked=self._export_fetch_lengths)
        self.export_fetch_button.setFixedWidth(150)
        self.export_fetch_button.setEnabled(False)
        self.layout().addWidget(self.export_fetch_button, 0, QtCore.Qt.AlignRight)

        self.layout().addItem(QtWidgets.QSpacerItem(0, 0, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding))

    def on_focus(self):
        """On focus, check if the simple calculation is still finished"""
        if self.settings['simple']['finished']:
            self.progress_bar.setValue(self.progress_bar.maximum())
            self.progress_message.setText('Berekening voltooid.')
            self.set_finished(True)
        else:
            self.progress_bar.setValue(0)
            self.set_finished(False)

        for process, checkbox in self.checkboxes.items():
            self.checkboxes[process].setChecked(process in self.settings['simple']['processes'])

    def _export_fetch_lengths(self):
        """Export fetch lengths to shapefile"""
        # Get path
        file_types = "Shape file (*.shp)"
        path, file_type = QtWidgets.QFileDialog.getSaveFileName(None, 'Export fetch lengths to shape file', '', file_types)
        self.mainwindow.setCursorWait()
        self.calculation.wavegrowth.fetchlines.reset_index().to_file(path)
        self.mainwindow.setCursorNormal()

    def update_progress(self, add_value, message=''):
        """Update progress bar"""
        if add_value == 0:
            self.progress_bar.setValue(0)
        else:    
            self.progress_bar.setValue(self.progress_bar.value() + add_value)
            if self.progress_bar.value() == self.progress_bar.maximum():
                self.done()
                self.export_fetch_button.setEnabled(True)
            self.progress_message.setText(message)

    def start_calc(self):
        """
        Start the simple calculation.
        """
        self.set_finished(False)
        self.progress_bar.setValue(0)
        # Collect processes to Calculate
        self.settings['simple']['processes'] = [process for process in self.checkboxes.keys() if self.checkboxes[process].isChecked()]
        self.steps_per_process = {
            'Diffractie': 1,
            'Transmissie': 1,
            'Lokale golfgroei': 5,
            'Golfbreking': 1,
        }
        self.progress_bar.setMaximum(sum([self.steps_per_process[process] for process in self.settings['simple']['processes']])+3)
        self.thread.start()

    def done(self):
        """Execute when calculation is finished"""
        self.set_finished(True)
        self.simple_calculation.results_changed = True
        logger.info('Adding results to visualisation tabs after simple calculation is finished.')
        # Add simple results to mapwidget and data widget
        self.mainwindow.tabwidgets['Overzicht'].add_table(method='simple')
        self.mainwindow.tabwidgets['Datavisualisatie'].add_table(method='simple')
        ui.dialogs.NotificationDialog('De resultaten van de eenvoudige methode zijn toegevoegd aan het overzicht en de datavisualisatie.')


class SimpleCalculationResultTab(widgets.AbstractTabWidget):
    """
    Tab widget with result tables from the simple calculation
    """
    def __init__(self, parent=None):
        """
        Constructor of the tab
        """
        # Create child class
        widgets.AbstractTabWidget.__init__(self, parent)
        self.initUI()

        # Allow continuing to the next step
        self.set_finished(True)

        self.calculation = self.simple_calculation

        self.standard_tables = [
            'Gecombineerd',
        ]

    def on_focus(self):
        """
        On focus, check if the results have changed, if so, reload the dataframes
        """
        if self.simple_calculation.results_changed:
            self.simple_calculation.results_changed = False
            self.add_results()
        
    def add_results(self):
        # Add results to tab
        dfs = []
        descriptions = []

        dataframes = [
            self.calculation.diffraction.output.reset_index().round(3),
            self.calculation.transmission.output.reset_index().round(3),
            self.calculation.wavegrowth.output.reset_index().round(3),
            self.calculation.wavebreaking.output.reset_index().round(3)
        ]


        # Add results depending on the modelled processes
        for description, df in zip(['Diffractie', 'Transmissie', 'Lokale golfgroei', 'Golfbreking'], dataframes):
            if description in self.settings['simple']['processes']:
                dfs.append(df)
                descriptions.append(description)

        # Add combined output (always present)
        dfs += [
            self.calculation.combinedresults.output.round(3),
        ]
        
        for df in dfs:
            # Replace HydraulicLoadId for Description, for clarity
            replace_column(
                dataframe=df,
                in_tag='HydraulicLoadId',
                out_tag='Load combination',
                dictionary=self.hydraulic_loads.description_dict
            )
        
        descriptions += self.standard_tables

        # Add dataframes to view
        self.add_dataframes(dfs, descriptions)
        
    def initUI(self):
        # Add view
        self.tableview = QtWidgets.QTableView(self)
        # Create layout
        vlayout = QtWidgets.QVBoxLayout()
        # Add combobox to select dataframe
        self.combobox = QtWidgets.QComboBox()
        self.combobox.currentIndexChanged.connect(self.change_model)
        vlayout.addWidget(self.combobox)

        # Add view to layout
        vlayout.addWidget(self.tableview)
        hbox = QtWidgets.QHBoxLayout()

        export_button = QtWidgets.QPushButton('Exporteren')
        export_button.clicked.connect(self._export_model)
        hbox.addWidget(export_button, 0, QtCore.Qt.AlignRight)
        
        # Add layout to widget
        vlayout.addLayout(hbox)
        self.setLayout(vlayout)

        # Set tableview properties
        self.tableview.setSortingEnabled(True)
        self.tableview.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Fixed)
        self.tableview.setShowGrid(False)
        self.tableview.setSelectionBehavior(QtWidgets.QTableView.SelectRows)
        self.tableview.setAlternatingRowColors(True)

        # Add header interaction
        self.horizontalHeader = self.tableview.horizontalHeader()
        self.horizontalHeader.sortIndicatorChanged.connect(self.headerTriggered)
        
        # Disable index column
        self.tableview.verticalHeader().setVisible(False)
        

    def add_dataframes(self, dataframes, names):
    
        self.models = {}
        for i in range(self.combobox.count()):
            self.combobox.currentIndexChanged.disconnect()
            self.combobox.removeItem(0)
            self.combobox.currentIndexChanged.connect(self.change_model)
        self.names = names

        # Create and add model
        for name, dataframe in zip(self.names, dataframes):
            # dataframe.index.name = 'ID'
            self.models[name] = HBHModels.PandasModel(dataframe)

        self.index = 0
        self.tableview.setModel(self.models[self.names[self.index]])
        self.combobox.addItems(names)
        self.combobox.setCurrentIndex(self.index)

    def change_model(self):

        self.index = self.combobox.currentIndex()
        if self.names[self.index] in list(self.models.keys()) + self.standard_tables:
            self.tableview.setModel(self.models[self.names[self.index]])
            self.set_header_sizes(self.models[self.names[self.index]])

    def set_header_sizes(self, model):
        """Set header sizes based on the first dataframe row"""
        dataframe = model._data
        row = dataframe.max(axis=0)
        columns = dataframe.columns
        for i, (element, label) in enumerate(zip(row, columns)):
            width = max(36 + len(str(element)) * 6, 70)
            labelwidth = len(label) * 7
            
            self.tableview.setColumnWidth(i, max(width, labelwidth))

    def _export_model(self):

        # Choose path
        file_types = "CSV (*.csv);;Excel (*.xlsx)"
        path, file_type = QtWidgets.QFileDialog.getSaveFileName(None, 'Export table data', '', file_types)

        # Save file
        if 'csv' in file_type:
            with open(path, 'w') as f:
                f.write('sep=;\n')
                self.models[self.names[self.index]]._data.to_csv(f, sep=';')
        elif 'xlsx' in file_type:
            self.models[self.names[self.index]]._data.to_excel(path)
        
    def headerTriggered(self, clicked_column=None):
        self.models[self.names[self.index]].sort(clicked_column, order=1)


class SwanCalculationTab(widgets.AbstractTabWidget):
    """
    Widget class to configure advanced calculation. This Widget is used for
    different similar steps in the calculation, with different configuration
    """
    def __init__(self, step, swantab=None):
        """
        Constructor of the tab
        """
        # Create child class
        super(SwanCalculationTab, self).__init__(swantab)

        if not step in ['I1', 'I2', 'I3', 'D', 'TR', 'W']:
            raise ValueError('Step "{}" not recognized'.format(step))

        # Get step and determine steptype
        self.step = step
        self.steptype = 'iteration' if self.step.startswith('I') else 'calculation'

        # Construct tab
        self.initTab()
        
        self.on_focus()
        
        # Create thread for IO
        self.generate_thread = threads.GenerateSwanFilesThread(self)
        self.import_thread = threads.ImportFilesThread(self)

    def initTab(self):
        """
        Construct tableview
        """
        boldFont = QtGui.QFont()
        boldFont.setBold(True)

        # Choose dataframe to add to table view
        self.get_tableview()

        self.vlayout = QtWidgets.QVBoxLayout()
        self.vlayout.setSpacing(10)

        # Generate input files
        hbox = QtWidgets.QHBoxLayout()
        generateButton = QtWidgets.QPushButton('Genereer invoerbestanden', clicked=self._generateButton)
        generateButton.setFixedWidth(200)
        self.generate_progress = QtWidgets.QProgressBar()
        self.generate_progress.setTextVisible(False)
        self.generate_progress.setRange(0, len(self.swan.iteration_results)+2)

        hbox.addWidget(generateButton)
        hbox.addWidget(self.generate_progress)

        self.vlayout.addLayout(hbox)

        rekenenLabel = QtWidgets.QLabel()
        rekenenLabel.setText('De SWAN berekeningen moet u zelf handmatig opstarten!')
        rekenenLabel.setFont(boldFont)
        self.vlayout.addWidget(rekenenLabel)

        # Read output files
        hbox = QtWidgets.QHBoxLayout()
        importButton = QtWidgets.QPushButton('Lees uitvoerbestanden SWAN', clicked=self._importButton)
        importButton.setFixedWidth(200)
        self.import_progress = QtWidgets.QProgressBar()
        self.import_progress.setTextVisible(False)
        self.import_progress.setRange(0, len(self.swan.iteration_results))

        hbox.addWidget(importButton)
        hbox.addWidget(self.import_progress)

        self.vlayout.addLayout(hbox)

        self.vlayout.addWidget(self.tableview)

        self.export_button = QtWidgets.QPushButton('Exporteren', clicked=self.tableview.export_dataframe)
        self.vlayout.addWidget(self.export_button, 0, QtCore.Qt.AlignRight)

        # Add layout to widget
        self.setLayout(self.vlayout)
        
    def get_tableview(self):
        """
        Get dataframe to show in table view
        """

        if self.step == 'I1':
            columns = ['Load combination'] + list(self.swan.result_parameters.keys()) + self.swan.iter_columns['I1']
            self.tableview = widgets.DataFrameWidget(self.swan.iteration_results, sorting_enabled=True, column_selection=columns, index=False)
        elif self.step == 'I2':
            columns = ['Load combination', 'Wave direction', 'Hs rand 2', 'Tp,s rand 2'] + self.swan.iter_columns['I2']
            self.tableview = widgets.DataFrameWidget(self.swan.iteration_results, sorting_enabled=True, column_selection=columns, index=False)
        elif self.step == 'I3':
            columns = ['Load combination', 'Wave direction', 'Hs rand 3', 'Tp,s rand 3'] + self.swan.iter_columns['I3']
            self.tableview = widgets.DataFrameWidget(self.swan.iteration_results, sorting_enabled=True, column_selection=columns, index=False)
        elif self.step == 'D':
            columns = ['Location', 'Load combination', 'Hs rand 3', 'Tp,s rand 3', 'X', 'Y', 'Normaal'] + self.swan.calc_columns['D']
            self.tableview = widgets.DataFrameWidget(self.swan.calculation_results, sorting_enabled=True, column_selection=columns, index=False)
        elif self.step == 'TR':
            columns = ['Location', 'Load combination', 'Hs rand 3', 'Tp,s rand 3', 'X', 'Y', 'Normaal'] + self.swan.calc_columns['TR']
            self.tableview = widgets.DataFrameWidget(self.swan.calculation_results, sorting_enabled=True, column_selection=columns, index=False)
        elif self.step == 'W':
            columns = ['Location', 'Load combination', 'Hs rand 3', 'Tp,s rand 3', 'X', 'Y', 'Normaal'] + self.swan.calc_columns['W']
            self.tableview = widgets.DataFrameWidget(self.swan.calculation_results, sorting_enabled=True, column_selection=columns, index=False)

    def on_focus(self):
        """
        Set the GUI elements based on the progress from the project file.
        """
        # Set the generate file progress to 100% if generated
        if self.settings['swan']['calculations'][self.step]['input_generated']:
            self.generate_progress.setValue(len(self.swan.iteration_results)+2)

        # Set import progress based on step and fill of the table
        if self.steptype == 'iteration':
            # For step I1, I2 and I3, check if Hs Steunpunt is known
            if ((not self.swan.iteration_results['Hs steunpunt 1'].eq(0.0).all() and self.step == 'I1') or
                (not self.swan.iteration_results['Hs steunpunt 2'].eq(0.0).all() and self.step == 'I2') or
                (not self.swan.iteration_results['Hs steunpunt 3'].eq(0.0).all() and self.step == 'I3')):
                self.import_progress.setValue(len(self.swan.iteration_results))
                self.set_finished(True)
            else:
                self.import_progress.setValue(0)
                self.set_finished(False)

        elif self.steptype == 'calculation':
            if ((not pd.isnull(self.swan.calculation_results.iloc[0]['Hm0_D']) and self.step == 'D') or
                (not pd.isnull(self.swan.calculation_results.iloc[0]['Hm0_TR']) and self.step == 'TR') or
                (not pd.isnull(self.swan.calculation_results.iloc[0]['Hm0_W']) and self.step == 'W')):
                self.import_progress.setValue(len(self.swan.iteration_results))
                self.set_finished(True)
            else:
                self.import_progress.setValue(0)
                self.set_finished(False)

        # Enable coninuing for D and TR, since the results from these steps are not
        # necessary for the next steps
        if self.step in ['D', 'TR']:
            self.set_finished(True)

    def _importButton(self):
        """
        Import SWAN results
        """
#        self.set_finished(True)
        
        self.set_finished(False)
        self.import_progress.setValue(0)
        # Emit layout to be changed
        self.tableview.model.layoutAboutToBeChanged.emit()
        self.import_thread.start()
        self.mainwindow.setDirty()

    def add_results(self):
        """Add results to visualisation"""
        # Add iteration results
        if self.step == 'I3' and not self.swan.iteration_results['Hs steunpunt 3'].eq(0.0).all():
            # Check if also filled
            logger.info(f'Adding SWAN iterations to visualisation tabs on tab finished ({self.step}).')
            self.mainwindow.tabwidgets['Overzicht'].add_table(method='swan_iterations')
            self.mainwindow.tabwidgets['Datavisualisatie'].add_table(method='swan_iterations')
            dialogs.NotificationDialog('De resultaten van de SWAN iteraties zijn toegevoegd aan de datavisualisatie.')

        elif self.step == 'W':
            logger.info(f'Adding SWAN calculations to visualisation tabs on tab finished.')
            self.mainwindow.tabwidgets['Overzicht'].add_table(method='swan_calculations')
            self.mainwindow.tabwidgets['Datavisualisatie'].add_table(method='swan_calculations')
            dialogs.NotificationDialog('De resultaten van de SWAN berekeningen zijn toegevoegd aan het overzicht en de datavisualisatie.')
        


    def sort_tableview(self):
        """
        Method to re-sort table after import. Before importing the
        tables will be sorted, so with this function the view
        can remain similar for the user.
        """
        # Resort output
        if self.steptype == 'iteration':
            if self.tableview.model.sort_column is not None:
                self.swan.iteration_results.sort_values(
                    by=self.tableview.model.sort_column,
                    ascending=self.tableview.model.sort_order,
                    inplace=True
                )
        else:
            if self.tableview.model.sort_column is not None:
                self.swan.calculation_results.sort_values(
                    by=self.tableview.model.sort_column,
                    ascending=self.tableview.model.sort_order,
                    inplace=True
                )
                logger.info('Resorted by {}'.format(self.tableview.model.sort_column))

    def _generateButton(self):
        """
        Generate SWAN input files

        1. Checks if files already exist, and may be overwritten
        2. Opens dialog to get files and path for generating input
        3. Generates input
        4. Updates project file for iteration

        """

        generate = True
        # Check if the files can be overwritten
        # condition1 = self.swanParameters['calculations'][self.step]['input_generated']
        condition = False
        if self.settings['swan']['swanfolder']:
            checkpath = os.path.join(self.settings['swan']['swanfolder'], self.steptype+'s', self.step, 'inputs')
            if os.path.exists(checkpath):
                # Determine the content of the 'inputs' folder
                condition = np.size(os.listdir(checkpath))

        if condition:
            generate = dialogs.QuestionDialog.question(
                self,
                self.mainmodel.appName,
                'De invoer voor deze stap is al gegenereerd.\nWeet u zeker dat u deze wilt overschrijven?'
            )

        # If they may be written
        if generate:
            # Launch dialog to get paths and generate
            self.swanParametersDialog = dialogs.GetSwanParametersDialog(self)
            self.swanParametersDialog.exec_()

            # If succeeded, generate the directory tree
            if self.swanParametersDialog.succeeded:

                self.generate_progress.setValue(0)
                if self.step == 'W':
                    self.set_finished(False)

                self.tableview.model.layoutAboutToBeChanged.emit()
                self.generate_thread.start()

                # update project file
                self.mainwindow.setDirty()
                self.import_progress.setValue(0)

class AdvancedCalculationSwanResultTab(widgets.AbstractTabWidget):
    """
    Tab with swan result table
    """
    def __init__(self, parent=None):
        """
        Constructor of the tab
        """
        # Create child class
        widgets.AbstractTabWidget.__init__(self, parent)
        self.init_ui()

        self.set_finished(True)


    def init_ui(self):
        # Add layout to widget
        self.setLayout(QtWidgets.QVBoxLayout())
        # Add view
        columns = [
            'Location', 'X', 'Y', 'Normaal', 'Load combination',
            'Hs', 'Tp', 'Tm-1,0', 'Wave direction',
            'Hs rand 3', 'Tp,s rand 3', 'Hm0_D', 'Tmm10_D',
            'Tp_D', 'Theta0_D', 'Hm0_TR', 'Tmm10_TR', 'Tp_TR', 'Theta0_TR',
            'Hm0_W', 'Tmm10_W', 'Tp_W', 'Theta0_W',
            'Hm0 swan', 'Tm-1,0 swan', 'Tp swan', 'Wave direction swan'
        ]
        self.tableview = widgets.DataFrameWidget(self.swan.calculation_results, sorting_enabled=True, column_selection=columns, index=False)
        
        # Create layout
        self.layout().setSpacing(10)

        # Add view to layout
        self.layout().addWidget(self.tableview)

        self.export_button = QtWidgets.QPushButton('Exporteren', clicked=self.tableview.export_dataframe)
        self.layout().addWidget(self.export_button, 0, QtCore.Qt.AlignRight)


class PharosInitializeTab(widgets.AbstractTabWidget):
    """
    Widget class to configure Pharos advanced calculation
    """
    def __init__(self, parent=None):
        """
        Constructor of the tab
        """
        # Create child class
        widgets.AbstractTabWidget.__init__(self, parent)

        # Construct tab
        self.init_ui()

    def init_ui(self):
        """
        Build UI elements
        """
        self.setLayout(QtWidgets.QVBoxLayout())

        # Add button to initialize table
        self.initialize_button = QtWidgets.QPushButton(
            'Initialiseer tabel',
            clicked=self.init_table_clicked
        )
        self.layout().addWidget(self.initialize_button, 0, QtCore.Qt.AlignLeft)
        
        # Add dataframe
        self.table_widget = widgets.PharosTableWidget(self)
        self.layout().addWidget(self.table_widget)

        # If the spectrum table is filled, load it:
        if not self.pharos.spectrum_table.empty:
            self.table_widget.initialize_content()
            # Set schematisations to comboxes
            self.load_schematisations()
            # Enable continuing
            self.set_finished(True)

    def init_table_clicked(self):
        """
        Class to launch dialog to get the schematisation parameters
        """
        # Launch dialog to get paths and pharos parameters
        self.pharos_parameters_dialog = dialogs.GetPharosParametersDialog(self)
        self.pharos_parameters_dialog.exec_()

        if self.pharos_parameters_dialog.succeeded:
            # Initialize Pharos 2D Spectrum
            self.pharos.initialize()
            self.pharos.fill_spectrum_table()
            
            # Remove directions from schematisations that are no longer present
            for schematisation, dirs in self.settings['pharos']['schematisations'].items():
                self.settings['pharos']['schematisations'][schematisation] = self.pharos.theta[np.in1d(self.pharos.theta, dirs)].tolist()
            
            # Initialize table
            self.table_widget.initialize_content()
            # Set schematisations to comboxes
            self.load_schematisations()
            # Enable continuing
            self.set_finished(True)

    def load_schematisations(self):
        """
        Method to load new schematisations. The old list is emptied and
        new schematisations are added bases on the content of the folder.
        """
        schematisations = []
        schematisations_dict = self.settings['pharos']['schematisations']

        # Determine the pharos schematisations
        folder = self.settings['pharos']['paths']['schematisation folder']
        for item in os.listdir(folder):
            # Todo: check for folder content

            # Currently, check if folder is directory
            if os.path.isdir(os.path.join(folder, item)):
                schematisations.append(item)
                # If item is already in settings, continue
                if item in schematisations_dict.keys():
                    continue
                # Else, add a new list
                self.settings['pharos']['schematisations'][item] = []

        # Add new dictionairy with elements we found in the schematisation folder
        self.settings['pharos']['schematisations'] = {
            key: values for key, values in self.settings['pharos']['schematisations'].items() if key in schematisations
        }
        
        # Update comboboxes in widget
        self.table_widget.update_comboboxes()

class PharosCalculationTab(widgets.AbstractTabWidget):
    """
    Widget class to configure pharos advanced calculation.
    """
    def __init__(self, parent=None):
        """
        Constructor of the tab
        """
        # Create child class
        widgets.AbstractTabWidget.__init__(self, parent)

        self.mainwindow.setCursorWait()

        # Construct tab
        self.init_ui()
        self.init_from_project()
        self.mainwindow.setCursorNormal()

        # Create thread for IO
        self.generate_thread = threads.GeneratePharosFilesThread(self)
        self.import_thread = threads.ImportPharosFilesThread(self)


    def init_ui(self):
        """
        Construct tableview
        """

        boldFont = QtGui.QFont()
        boldFont.setBold(True)

        # Dataframe to add to table view
        columns = [
            'Location', 'X', 'Y', 'Normaal', 'Load combination',
            'Hs', 'Tp', 'Tm-1,0', 'Wave direction',
            'Hs pharos', 'Tp pharos', 'Tm-1,0 pharos', 'Wave direction pharos'
        ]
        self.tableview = widgets.DataFrameWidget(
            dataframe=self.pharos.calculation_results,
            sorting_enabled=True,
            column_selection=columns,
            index=False
        )

        self.vlayout = QtWidgets.QVBoxLayout()
        self.vlayout.setSpacing(10)

        # Generate input files
        hbox = QtWidgets.QHBoxLayout()
        generateButton = QtWidgets.QPushButton('Genereer invoerbestanden', clicked=self._generateButton)
        generateButton.setFixedWidth(200)
        self.generate_progress = QtWidgets.QProgressBar()
        self.generate_progress.setTextVisible(False)
        
        hbox.addWidget(generateButton)
        hbox.addWidget(self.generate_progress)

        self.vlayout.addLayout(hbox)

        rekenenLabel = QtWidgets.QLabel()
        rekenenLabel.setText('De PHAROS berekeningen moet u zelf handmatig opstarten!')
        rekenenLabel.setFont(boldFont)
        self.vlayout.addWidget(rekenenLabel)

        # Read output files
        hbox = QtWidgets.QHBoxLayout()
        importButton = QtWidgets.QPushButton('Lees uitvoerbestanden PHAROS', clicked=self.import_results)
        importButton.setFixedWidth(200)
        self.import_progress = QtWidgets.QProgressBar()
        self.import_progress.setTextVisible(False)

        hbox.addWidget(importButton)
        hbox.addWidget(self.import_progress)

        self.vlayout.addLayout(hbox)

        self.vlayout.addWidget(self.tableview)

        # Add layout to widget
        self.setLayout(self.vlayout)

    def init_from_project(self):
        """
        Set the GUI elements based on the progress from the project file.
        """
        # Set the generate file progress to 100% if generated
        if self.settings['pharos']['input_generated']:
            self.generate_progress.setValue(self.generate_progress.maximum())

    def import_results(self):
        """
        Import PHAROS results
        """

        self.set_finished(False)
        self.import_progress.setValue(0)

        self.tableview.model.layoutAboutToBeChanged.emit()
        self.import_thread.start()

        
        self.mainwindow.setDirty()

        self.tableview.model.layoutChanged.emit()

    def add_results(self):
        """Add results to visualisation"""
        logger.info('Adding PHAROS calculations to visualisation tabs on result import.')
        self.mainwindow.tabwidgets['Overzicht'].add_table(method='pharos_calculations')
        self.mainwindow.tabwidgets['Datavisualisatie'].add_table(method='pharos_calculations')
        dialogs.NotificationDialog('De resultaten van de PHAROS berekeningen zijn toegevoegd aan het overzicht en de datavisualisatie.')

    def sort_tableview(self):
        """
        Method to re-sort table after import. Before importing the
        tables will be sorted, so with this function the view
        can remain similar for the user.
        """
        # Resort output
        if self.tableview.model.sort_column is not None:
            self.swan.iteration_results.sort_values(
                by=self.tableview.model.sort_column,
                ascending=self.tableview.model.sort_order,
                inplace=True
            )
            logger.info('Resorted by {}'.format(self.tableview.model.sort_column))


    def _generateButton(self):
        """
        Generate PHAROS input files

        1. Checks if files already exist, and may be overwritten
        2. Generates input

        """

        generate = True
        # Check if the files can be overwritten
        if self.settings['pharos']['input_generated']:
            generate = dialogs.QuestionDialog.question(
                self,
                self.mainmodel.appName,
                'De invoer voor deze berekeneningen is al gegenereerd. Weet u zeker dat u deze wilt overschrijven?'
            )

        # If they may be written
        if generate:
            self.generate_progress.setValue(0)
            self.import_progress.setValue(0)
            self.generate_thread.start()

    def on_focus(self):
        """
        On focus of this tab, change the progress bar range and
        status
        """
        # Determine number of calculations
        num_calcs = len(self.pharos.get_combinations())
        self.generate_progress.setRange(0, num_calcs)
        # Set the generate file progress to 100% if generated
        if self.settings['pharos']['input_generated']:
            self.generate_progress.setValue(self.generate_progress.maximum())

        self.import_progress.setRange(0, self.generate_progress.maximum() + len(self.hydraulic_loads)*10)
        
        if not pd.isnull(self.pharos.calculation_results.iloc[0]['Hs pharos']):
            self.import_progress.setValue(self.import_progress.maximum())
            self.set_finished(True)
        else:
            self.import_progress.setValue(0)
            self.set_finished(False)

class AdvancedCombinedResultsTab(widgets.AbstractTabWidget):
    """
    Tab for pharos or hares calculation results
    """
    def __init__(self, parent=None):
        """
        Constructor of the tab
        """
        # Create child class
        widgets.AbstractTabWidget.__init__(self, parent)

        self.initUI()

        self.set_finished(True)

    def initUI(self):
        # Add layout to widget
        self.setLayout(QtWidgets.QVBoxLayout())
        
        # Add view        
        # Aangepast Svasek 04/10/18 - if-else optie met Hares toegevoegd
        if self.settings['calculation_method']['include_pharos']: # PHAROS results
            view_columns = ['Location', 'Load combination', 'Hs pharos', 'Tp pharos',
                            'Tm-1,0 pharos', 'Wave direction pharos',
                            'Hm0 swan', 'Tm-1,0 swan', 'Tp swan', 'Wave direction swan',
                            'Hs totaal', 'Tm-1,0 totaal', 'Tp totaal', 'Wave direction totaal']
    
            self.tableview = widgets.DataFrameWidget(
                dataframe=self.pharos.calculation_results,
                sorting_enabled=True,
                column_selection=view_columns,
                index=False
            )

        else: # HARES results
            view_columns = ['Location', 'Load combination', 'Hs hares', 'Tp hares',
                            'Tm-1,0 hares', 'Wave direction hares',
                            'Hm0 swan', 'Tm-1,0 swan', 'Tp swan', 'Wave direction swan',
                            'Hs totaal', 'Tm-1,0 totaal', 'Tp totaal', 'Wave direction totaal']
    
            self.tableview = widgets.DataFrameWidget(
                dataframe=self.hares.calculation_results,
                sorting_enabled=True,
                column_selection=view_columns,
                index=False
            )
            
        # Create layout
        self.layout().setSpacing(10)

        # Add view to layout
        self.layout().addWidget(self.tableview)

        self.export_button = QtWidgets.QPushButton('Exporteren', clicked=self.tableview.export_dataframe)
        self.layout().addWidget(self.export_button, 0, QtCore.Qt.AlignRight)


# Toegevoegd Svasek 03/10/18 - Hares tab toegevoegd
class HaresCalculationTab(widgets.AbstractTabWidget):
    """
    Widget class to configure Hares advanced calculation.
    """
    def __init__(self, parent=None):
        """
        Constructor of the tab
        """
        # Create child class
        widgets.AbstractTabWidget.__init__(self, parent)
        
        self.mainwindow.setCursorWait()
        
        # Construct tab
        self.init_ui()

        self.init_from_project()

        self.mainwindow.setCursorNormal()

        # Create thread for IO
        self.import_thread = threads.ImportHaresFilesThread(self)


    def init_ui(self):
        """
        Construct tableview
        """
        boldFont = QtGui.QFont()
        boldFont.setBold(True)

        # Dataframe to add to table view
        columns = [
            'Location', 'X', 'Y', 'Normaal', 'Load combination',
            'Hs', 'Tp', 'Tm-1,0', 'Wave direction',
            'Hs hares', 'Tp hares', 'Tm-1,0 hares', 'Wave direction hares'
        ]

        self.tableview = widgets.DataFrameWidget(
            dataframe=self.hares.calculation_results,
            sorting_enabled=True,
            column_selection=columns,
            index=False
        )

        self.vlayout = QtWidgets.QVBoxLayout()
        self.vlayout.setSpacing(10)

        rekenenLabel = QtWidgets.QLabel()
        rekenenLabel.setText('De HARES berekeningen moet u zelf handmatig opstarten!')
        rekenenLabel.setFont(boldFont)
        self.vlayout.addWidget(rekenenLabel)

        # Read output files
        hbox = QtWidgets.QHBoxLayout()
        importButton = QtWidgets.QPushButton('Lees uitvoerbestanden HARES', clicked=self.import_results)
        importButton.setFixedWidth(200)
        self.import_progress = QtWidgets.QProgressBar()
        self.import_progress.setTextVisible(False)
        self.import_progress.setRange(0,len(self.hares.calculation_results['Location']))

        hbox.addWidget(importButton)
        hbox.addWidget(self.import_progress)

        self.vlayout.addLayout(hbox)

        self.vlayout.addWidget(self.tableview)

        # Add layout to widget
        self.setLayout(self.vlayout)

    def init_from_project(self):
        """
        Set the GUI elements based on the progress from the project file.
        """
        pass
    
    def import_results(self):
        """
        Import HARES results
        """
        
        self.HaresFolderDialog = dialogs.GetHaresFolderDialog(self)
        self.HaresFolderDialog.exec_()
        
        if self.HaresFolderDialog.succeeded:

            self.set_finished(False)
            self.import_progress.setValue(0)
    
            self.tableview.model.layoutAboutToBeChanged.emit()
            self.import_thread.start()

            self.mainwindow.setDirty()
            
            self.tableview.model.layoutChanged.emit()

            #TODO: Add results

    def add_results(self):
        """Add results to visualisation"""
        logger.info('Adding HARES calculations to visualisation tabs on result import.')
        self.mainwindow.tabwidgets['Overzicht'].add_table(method='hares_calculations')
        self.mainwindow.tabwidgets['Datavisualisatie'].add_table(method='hares_calculations')
        dialogs.NotificationDialog('De resultaten van de HARES berekeningen zijn toegevoegd aan het overzicht en de datavisualisatie.')

    def sort_tableview(self):
        """
        Method to re-sort table after import. Before importing the
        tables will be sorted, so with this function the view
        can remain similar for the user.
        """
        # Resort output
        if self.tableview.model.sort_column is not None:
            self.swan.iteration_results.sort_values(
                by=self.tableview.model.sort_column,
                ascending=self.tableview.model.sort_order,
                inplace=True
            )
            logger.info('Resorted by {}'.format(self.tableview.model.sort_column))


    def on_focus(self):
        """
        On focus of this tab, change the progress bar range and
        status
        """

#        self.import_progress.setRange(0, self.generate_progress.maximum() + len(self.hydraulic_loads)*10)
        self.import_progress.setRange(0, len(self.hares.calculation_results['Location']))
        
        if not pd.isnull(self.hares.calculation_results.iloc[0]['Hs hares']):
            self.import_progress.setValue(self.import_progress.maximum())
            self.set_finished(True)
        else:
            self.import_progress.setValue(0)
            self.set_finished(False)

class ModelUncertaintyTab(widgets.AbstractTabWidget):
    """
    Tab widget to view and edit model uncertainties
    """
    def __init__(self, parent=None):
        """
        Constructor of the tab
        """
        # Create child class
        widgets.AbstractTabWidget.__init__(self, parent)

        # Construct UI
        self._initUI()

    def _initUI(self):

        # Create layout
        self.vlayout = QtWidgets.QVBoxLayout()
        self.vlayout.setSpacing(10)

        # Label with instructions
        label = QtWidgets.QLabel()
        label.setText("""Selecteer één of meerdere rijen en klik op "Definïeer onzekerheden" om de modelonzekerheid op te geven.
        Druk om "Bekijk scatterplot" om de resultaten in het steunpunt en de uitvoerlocatie te bekijken.""")
        self.vlayout.addWidget(label)

        # Add view
        self.tableview = widgets.DataFrameWidget(
            self.modeluncertainties.table,
            editing_enabled=False,
            sorting_enabled=True,
            index=False
        )
        # self.tableview.horizontalHeader.setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        # Add view to layout
        self.vlayout.addWidget(self.tableview)

        # Add define uncertainties button
        buttons = QtWidgets.QHBoxLayout()
        self.define_uncertainties_button = QtWidgets.QPushButton('Definïeer onzekerheden')
        self.define_uncertainties_button.setFixedWidth(150)
        self.define_uncertainties_button.clicked.connect(self._define_uncertainties)
        buttons.addWidget(self.define_uncertainties_button)

        # Add watch scatter button
        self.watch_scatter_button = QtWidgets.QPushButton('Bekijk scatterplot')
        self.watch_scatter_button.setFixedWidth(150)
        self.watch_scatter_button.clicked.connect(self._watch_scatter)
        buttons.addWidget(self.watch_scatter_button)

        # Add spacer
        buttons.addItem(QtWidgets.QSpacerItem(0, 0, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Maximum))

        # Add exportbutton
        self.export_button = QtWidgets.QPushButton('Exporteren')
        self.export_button.clicked.connect(self._export_model)
        buttons.addWidget(self.export_button)
        self.vlayout.addLayout(buttons)

        # Add layout to widget
        self.setLayout(self.vlayout)

    def on_focus(self):
        if not self.modeluncertainties.table['Optie'].isnull().any():
            self.set_finished(True)
        else:
            self.set_finished(False)

    def adjust_selection(self, uncertainties, option):
        """
        Set uncertainties for selected locations based on option
        """
        # Get indices of selected locations
        # Fill in option
        # Reshape the uncertainties
        uncertainties = uncertainties.model._data.stack()
        index = [' '.join(i[::-1]) for i in uncertainties.index.to_numpy()]
        uncertainties.index = index
        # Add uncertainties to modeluncertainties model
        rows = list(set([self.modeluncertainties.table.index.array[i.row()] for i in self.tableview.selectedIndexes()]))
        
        for rowidx in rows:
            # Add option
            self.modeluncertainties.table.at[rowidx, 'Optie'] = option
            # Add values
            for key, val in uncertainties.items():
                self.modeluncertainties.table.at[rowidx, key] = val

        # Resize columns
        # self.tableview.horizontalHeader.setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)

        # Check if everything is filled in
        if not self.modeluncertainties.table['Optie'].isnull().any():
            self.set_finished(True)

    def _export_model(self):

        self.path = QtWidgets.QFileDialog.getSaveFileName(None, 'Save file', '', "CSV/Excel (*.csv, *.xlsx)")[0]

        # Replace Greek characters for export
        df = self.tableview.model._data

        # Save file
        if self.path.endswith('.xlsx'):
            self.tableview.model._data.to_excel(self.path)
        elif self.path.endswith('.csv'):
            with open(self.path, 'w') as f:
                f.write('sep=;\n')
                self.tableview.model._data.to_csv(f, sep=';')
        else:
            raise ValueError('File extension not recognized. Choose *.csv or *.xlsx.')

    def _define_uncertainties(self):
        self.define_window = dialogs.DefineUncertaintiesDialog(self)

        # Open Dialog
        self.define_window.exec_()

    def _watch_scatter(self):
        self.scatter_window = dialogs.ResultScatterDialog(self)
        # Open Dialog
        self.scatter_window.exec_()


class ExportToDatabaseTab(widgets.AbstractTabWidget):
    """
    Tab widget to configure export to database.
    The user gives the exportnames and paths in this tab.
    """
    def __init__(self, parent=None):
        """
        Constructor of the tab
        """
        # Create child class
        widgets.AbstractTabWidget.__init__(self, parent)

        self.export_settings = self.project.getGroupSettings('export')

        # Construct UI
        self.init_ui()

        # Load from project
        if self.export_settings['HLCD']:
            self.hlcd_line_edit.LineEdit.setText(self.export_settings['HLCD'])
        if self.export_settings['export_succeeded']:
            self.progress_bar.setValue(100)
            

    def model_flags(self, index):
        """
        New flags function for the dataframe model to disable editing for the
        first column.
        """
        if index.column() == 0:
            return Qt.Qt.ItemIsEnabled | Qt.Qt.ItemIsSelectable
        else:
            return Qt.Qt.ItemIsEnabled | Qt.Qt.ItemIsSelectable | Qt.Qt.ItemIsEditable


    def init_ui(self):
        # Create layout
        self.setLayout(QtWidgets.QVBoxLayout())
        self.layout().setSpacing(10)

        # Add hlcd browse
        HLCDframe = QtWidgets.QGroupBox()
        HLCDframe.setTitle('HLCD config-databases:')
        HLCDframe.setLayout(QtWidgets.QVBoxLayout())
        self.hlcd_line_edit = widgets.ExtendedLineEdit(label='HLCD:', browsebutton=QtWidgets.QPushButton('...', clicked=self._load_HLCD))
        self.config_info_label = QtWidgets.QLabel('Merk op dat de locatie van config-database niet opgegeven hoeft te worden. Deze wordt bepaald op basis van de naam van de HRD')

        self.meta_database_checkbox = QtWidgets.QCheckBox('Exporteer HLCD en config-databases')
        self.meta_database_checkbox.setChecked(True)
        self.meta_database_checkbox.stateChanged.connect(self._enable_hlcd)
        self.meta_database_checkbox.setChecked(self.export_settings['export_HLCD_and_config'])
        HLCDframe.layout().addWidget(self.meta_database_checkbox)
        HLCDframe.layout().addWidget(self.hlcd_line_edit)
        HLCDframe.layout().addWidget(self.config_info_label)
        self.layout().addWidget(HLCDframe)

        # Add tableview
        self.HRDframe = QtWidgets.QGroupBox()
        self.HRDframe.setTitle('Geef per locatie een exportnaam en de HRD waaraan de data toegevoegd worden.')
        self.HRDframe.setLayout(QtWidgets.QVBoxLayout())

        self.tableview = widgets.DataFrameWidget(
            self.export.export_dataframe,
            editing_enabled=True,
            sorting_enabled=False,
            index=False
        )
        self.tableview.horizontalHeader.setStretchLastSection(False)
        self.tableview.setItemDelegateForColumn(3, widgets.ButtonDelegate(self))
        self.tableview.setColumnWidth(3, 30)
        # self.tableview.setColumnWidth(0, 100)
        self.tableview.setColumnWidth(1, 200)
        # self.tableview.setColumnWidth(2, 100)
        # self.tableview.setColumnWidth(3, 100)
        self.tableview.setWordWrap(False)
        self.tableview.setTextElideMode(QtCore.Qt.ElideLeft)
        self.tableview.horizontalHeader.setSectionResizeMode(2, QtWidgets.QHeaderView(Qt.Qt.Horizontal).Stretch)
        self.tableview.model.flags = self.model_flags

        # Add constraint
        self.tableview.model.add_constraints('SQLite-database', os.path.exists, 'Pad niet gevonden.')

        # Add view to layout
        self.HRDframe.layout().addWidget(self.tableview)

        buttons = QtWidgets.QHBoxLayout()
        buttons.addItem(QtWidgets.QSpacerItem(0, 0, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Maximum))

        # Add export button
        buttons.addWidget(QtWidgets.QPushButton('Exporteren', clicked=self._export_table))
        # Add import button
        buttons.addWidget(QtWidgets.QPushButton('Importeren', clicked=self._import_table))
        self.HRDframe.layout().addLayout(buttons)

        self.layout().addWidget(self.HRDframe)

        hlayout = QtWidgets.QHBoxLayout()
        # Add export to db button
        self.export_to_database_button = QtWidgets.QPushButton('Exporteren naar database', clicked=self._export_to_database)
        hlayout.addWidget(self.export_to_database_button)

        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        hlayout.addWidget(self.progress_bar)

        self.layout().addLayout(hlayout)

    def _enable_hlcd(self, state):
        """
        Enable or disable the export of hlcd and config database, based on the
        changed state of the checkbox.

        Parameters
        ----------
        state : QtCore.Qt state
            checked or unchecked
        """

        # get checked as boolean
        checked = state == QtCore.Qt.Checked
        # Change settings
        self.export_settings['export_HLCD_and_config'] = checked
        # enable or disable elements in the layout
        layout = self.hlcd_line_edit.layout()
        for i in range(layout.count()):
            item = layout.itemAt(i)
            if item.widget():
                item.widget().setEnabled(checked)
        # enable or disable label with info on config
        self.config_info_label.setEnabled(checked)


    def _load_HLCD(self):
        """
        Method to load a HLCD database. The database is loaded and modified in
        the export routine.
        """
        # Get path with dialog
        path = QtWidgets.QFileDialog.getOpenFileName(
            self.hlcd_line_edit.LineEdit,
            'Open SQLite-database (HLCD) om aan te passen',
            '',
            "SQLite database (*.sqlite)"
        )[0]
        if not path:
            return None
        # Save path to project structure
        self.export_settings['HLCD'] = path
        # Add to schematisation
        self.mainmodel.export.add_HLCD(path)
        self.mainwindow.setDirty()
        # Adjust line edit
        self.hlcd_line_edit.set_value(path)

    def cellButtonClicked(self):
        # This slot will be called when our button is clicked.
        # self.sender() returns a refence to the QPushButton created
        # by the delegate, not the delegate itself.
        path = QtWidgets.QFileDialog.getOpenFileName(None, 'Open SQLite-database om uitvoer aan toe te voegen (HRD)', '', "SQLite database (*.sqlite)")[0]
        if not path:
            return None
        idx = int(self.sender().objectName())
        dfidx = self.export.export_dataframe.index[idx]
        self.export.export_dataframe.set_value(dfidx, 'SQLite-database', path)
        
    def _import_table(self):
        """
        Import table
        """
        # Get path with dialog
        file_types = "Excel (*.xlsx);;CSV (*.csv)"
        path, file_type = QtWidgets.QFileDialog.getOpenFileName(None, 'Open csv exporteertabel', '', file_types)

        # Save path to project structure
        if file_type == 'CSV (*.csv)':
            # If the file is a csv, the first row can contain the delimiter,
            # since this is how HB Havens exports. Therefor, check the first line
            # and get the delimiter. Else, derive it with the csv.Sniffer
            with open(path, 'r') as f:
                line = f.readline()
                if line.startswith('sep'):
                    sep = line.split('=')[1].strip()
                    df = pd.read_csv(f, sep=sep)
                else:
                    sep = csv.Sniffer().sniff(line).delimiter
                    df = pd.read_csv(f, sep=sep, names=line.split(sep), header=None)
        elif file_type == 'Excel (*.xlsx)':
            df = pd.read_excel(path)
        else:
            return None

        checkcols = self.export.dfcolumns[:]
        if not all(np.in1d(checkcols, df.columns.values.tolist())):
            raise ValueError('Niet alle kolomnamen ({}) komen voor in de tabel.'.format(', '.join(checkcols)))

        present_indices = np.in1d(self.export.location_names, df['Naam'].squeeze())
        if not all(present_indices):
            raise ValueError('Niet alle locatienamen ({}) komen voor in de tabel.'.format(
                ', '.join(np.array(self.export.location_names)[~present_indices].tolist())))

        for _, row in df.iterrows():
            sqlpath = row['SQLite-database']
            if pd.isnull(sqlpath):
                continue
            if not os.path.exists(sqlpath):
                raise OSError('Pad naar database niet gevonden: {}'.format(sqlpath))

        # Als geen fouten, verander data
        df['...'] = np.nan
        # Empty old df
        if not self.export.export_dataframe.empty:
            self.export.export_dataframe.drop(self.export.export_dataframe.index, inplace=True)
        # Change model data
        self.tableview.model.layoutAboutToBeChanged.emit()
        columns = self.export.dfcolumns[:]
        self.export.export_dataframe[columns] = df[columns]
        self.tableview.model.layoutChanged.emit()
            
    def _export_table(self):
        """
        Export table
        """
        self.tableview.export_dataframe()

    def _update_progress(self, add_value):
        """Update progress bar"""
        if add_value == 0:
            self.progress_bar.setValue(0)
        else:    
            self.progress_bar.setValue(round(self.progress_bar.value() + add_value))

    def _export_to_database(self):
        self.progress_bar.setValue(0)

        # Check if the export table is complete
        if self.export.export_dataframe['SQLite-database'].isnull().any():
            export = dialogs.QuestionDialog.question(
                self,
                self.mainmodel.appName,
                (
                    'Niet alle paden zijn ingevuld, alleen de ingevulde locaties worden geëxporteerd.'
                    '\nWeet u zeker dat u deze wilt doorgaan?'
                )
            )
            if not export:
                return None
        
        # Check if all locations are filled in
        subset = self.export.export_dataframe[['SQLite-database', 'Exportnaam']].dropna(subset=['SQLite-database'])
        if subset.isnull().any().any():
            raise ValueError('Niet alle locatienamen zijn ingevuld')
        # Check if all location names are unique
        if len(subset['Exportnaam'].unique()) != len(subset['Exportnaam']):
            raise ValueError('Niet alle locatienamen zijn uniek')

        # Export per database
        self.mainwindow.setCursorWait()
        self.export.export_output_to_database(progress_function=self._update_progress)
        self.mainwindow.setCursorNormal()

        # Set progress_bar
        self.export_settings['export_succeeded'] = True
        self.progress_bar.setValue(100)
        self.mainwindow.setDirty()