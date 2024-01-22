# -*- coding: utf-8 -*-
"""
Created on  : Tue Aug 22 17:02:55 2017
Author      : Guus Rongen, HKV Lijn in Water
Project     : PR3594.10.00
Description :
"""

import logging
import os
import sys
import traceback

from PyQt5 import Qt, QtCore, QtGui, QtWidgets

from hbhavens.core.models import MainModel
from hbhavens.ui import dialogs
from hbhavens.ui import tabs, widgets
from hbhavens.ui.tabs.hydraulicloads import HydraulicLoadTab
from hbhavens.ui.tabs.overviewmap import OverviewMapTab
from hbhavens.ui.tabs.dataviewer import DataViewerTab

logger = logging.getLogger(__name__)


class MainWindow(QtWidgets.QMainWindow):

    def __init__(self):
        super(MainWindow, self).__init__()

        self.appsettings = QtCore.QSettings()
        self.setWindowTitle("HB Havens")

        self.setWindowIcon(self.getIcon('hbhavens.png'))

        self.setCursor(Qt.Qt.ArrowCursor)

        # self.no_exceptions = True

        # Definineer tabs
        self.tabwidgets = {}
        self.tabs_per_step = {}
        self.tabs_per_step['general'] = [
            'Welkom',
            'Schematisering',
            'Hydraulische belasting steunpunt',
            'Keuze methode',
        ]

        self.tabs_per_step['finish'] = [
            'Modelonzekerheden',
            'Exporteren',
        ]

        self.tabs_per_step['simple'] = self.tabs_per_step['general'] + [
            'Rekenen',
            'Resultaten',
        ] + self.tabs_per_step['finish']

        self.tabs_per_step['pharos'] = [
            'PHAROS - Initialiseren',
            'PHAROS - Rekenen',
            'Gecombineerd - Resultaten',
        ]
        
        # Toegevoegd Svasek 03/10/18 - Initialiseer Hares tab
        self.tabs_per_step['hares'] = [
            'HARES - Rekenen',
            'Gecombineerd - Resultaten',
        ]

        self.tabs_per_step['swan'] = [
            'SWAN - Condities modelrand - Stap 1',
            'SWAN - Condities modelrand - Stap 2',
            'SWAN - Condities modelrand - Stap 3',
            'SWAN - Rekenen - D',
            'SWAN - Rekenen - TR',
            'SWAN - Rekenen - W',
            'SWAN - Totaal',
        ]

        self.tabs_per_step['advanced'] = (
            self.tabs_per_step['general'] +
            self.tabs_per_step['swan'] +
            self.tabs_per_step['finish']
        )

        self.tabs_per_step['advanced_with_pharos'] = (
            self.tabs_per_step['general'] +
            self.tabs_per_step['swan'] +
            self.tabs_per_step['pharos'] +
            self.tabs_per_step['finish']
        )

        # Toegevoegd Svasek 03/10/18 - Initialiseer de opzet voor een berekening met Hares
        self.tabs_per_step['advanced_with_hares'] = (
            self.tabs_per_step['general'] +
            self.tabs_per_step['swan'] +
            self.tabs_per_step['hares'] +
            self.tabs_per_step['finish']
        )

        self.step = 0

        # Always initiate a main model
        self.mainmodel = MainModel()

        # Construct user interface
        self.init_ui()

        def test_exception_hook(exctype, value, tback):
            """
            Function that catches errors and gives a Notification
            instead of a crashing application.
            """
            sys.__excepthook__(exctype, value, tback)
            self.setCursorNormal()
            dialogs.NotificationDialog(
                text='\n'.join(traceback.format_exception_only(exctype, value)),
                severity='critical',
                details='\n'.join(traceback.format_tb(tback))
            )
        

        sys.excepthook = test_exception_hook

        
    def set_calculation_method(self, methodname):
        # Set method
        self.method = methodname
        # Add to project
        self.mainmodel.project.settings['calculation_method']['method'] = self.method
        
        # Check if include PHAROS
        if self.method == 'advanced' and self.mainmodel.project.settings['calculation_method']['include_pharos']:
            self.steps = self.tabs_per_step[self.method+'_with_pharos']
        
        # Check if include HARES
        elif self.method == 'advanced' and self.mainmodel.project.settings['calculation_method']['include_hares']:
            self.steps = self.tabs_per_step[self.method+'_with_hares']    
        
        # Alleen SWAN
        else:
            self.steps = self.tabs_per_step[self.method]

        # Update progress overview
        if hasattr(self, 'progress_overview'):
            self.progress_overview.set_steps(self.steps)
            self.progress_overview.set_current_step(self.step)

    def clearMain(self):
        """
        Method to clear all widgets from main window
        """
        # Loop through elements in mainhbox and buttonbox
        layouts = [self.buttonbox, self.mainhbox] if hasattr(self, 'buttonbox') else [self.mainhbox]
            
        for layoutbox in layouts:
            for i in reversed(range(layoutbox.count())):
                # Select the widget
                obj = layoutbox.itemAt(i).widget()
                # ..or the layout
                if not obj:
                    obj = layoutbox.itemAt(i).layout()
                obj.setParent(None)

        # Delete all elements from tabwidgets
        keys = list(self.tabwidgets.keys())
        for key in keys:
            del self.tabwidgets[key]

        # Delete progress checkboxes
        if hasattr(self, 'progress_overview'):
            self.progress_overview.set_steps([])

    def setCursorWait(self):
        Qt.QApplication.setOverrideCursor(Qt.QCursor(QtCore.Qt.WaitCursor))
        Qt.QApplication.processEvents()

    def setCursorNormal(self):
        Qt.QApplication.restoreOverrideCursor()

    def initMain(self):
        """
        Initialize main window. Is executed when a new project is opened or
        a project is loaded.
        """

        self.setCursorWait()
        self.clearMain()

        # Tabframe
        self.tabsframe = widgets.CustomTabWidget(self)

        # Progresstree
        self.progress_overview = widgets.ProgressBoxes(self)
        progressbox = QtWidgets.QVBoxLayout()
        progressbox.addWidget(self.progress_overview)

        # Buttons
        self.buttonbox = QtWidgets.QHBoxLayout()

        self.next_button = QtWidgets.QPushButton("Volgende")
        self.next_button.clicked.connect(self.next_step)

        self.next_shortcut = QtWidgets.QShortcut(QtGui.QKeySequence(">"), self)
        self.next_shortcut.activated.connect(self.next_step)

        self.previous_button = QtWidgets.QPushButton("Vorige")
        self.previous_button.clicked.connect(self.previous_step)

        self.previous_shortcut = QtWidgets.QShortcut(QtGui.QKeySequence("<"), self)
        self.previous_shortcut.activated.connect(self.previous_step)

        self.buttonbox.addWidget(self.previous_button)
        self.buttonbox.addWidget(self.next_button)

        progressbox.addLayout(self.buttonbox)

        self.mainhbox.addLayout(progressbox)
        self.mainhbox.addWidget(self.tabsframe)

        self.setCursorNormal()

        # Initialize the overview
        self.tabwidgets['Overzicht'] = self.init_tab('Overzicht')
        self.tabsframe._open_or_focus('Overzicht')

        # Initialize the overview
        self.tabwidgets['Datavisualisatie'] = self.init_tab('Datavisualisatie')
        self.tabsframe._open_or_focus('Datavisualisatie')

        # Fill the progressbox
        calc_method = self.mainmodel.project.settings['calculation_method']['method']
        self.set_calculation_method(calc_method)
        
        # Initialize the first step
        self._go_to_step(self.step)

    def init_tab(self, name):
        """
        Method to initialize a tab
        """

        if name == 'Welkom':
            tab = tabs.WelcomeTab(self)
        elif name == 'Overzicht':
            tab = OverviewMapTab(self)
        elif name == 'Datavisualisatie':
            tab = DataViewerTab(self)
        elif name == 'Schematisering':
            tab = tabs.SchematisationTab(self)
        elif name == 'Hydraulische belasting steunpunt':
            tab = HydraulicLoadTab(self)
        elif name == 'Keuze methode':
            tab = tabs.CalculationMethodTab(self)
        elif name == 'Rekenen':
            tab = tabs.SimpleCalculationTab(self)
        elif name == 'Resultaten':
            tab = tabs.SimpleCalculationResultTab(self)
        elif name == 'SWAN - Condities modelrand - Stap 1':
            tab = tabs.SwanCalculationTab('I1', self)
        elif name == 'SWAN - Condities modelrand - Stap 2':
            tab = tabs.SwanCalculationTab('I2', self)
        elif name == 'SWAN - Condities modelrand - Stap 3':
            tab = tabs.SwanCalculationTab('I3', self)
        elif name == 'SWAN - Rekenen - D':
            tab = tabs.SwanCalculationTab('D', self)
        elif name == 'SWAN - Rekenen - TR':
            tab = tabs.SwanCalculationTab('TR', self)
        elif name == 'SWAN - Rekenen - W':
            tab = tabs.SwanCalculationTab('W', self)
        elif name == 'SWAN - Totaal':
            tab = tabs.AdvancedCalculationSwanResultTab(self)
        elif name == 'PHAROS - Initialiseren':
            tab = tabs.PharosInitializeTab(self)
        elif name == 'PHAROS - Rekenen':
            tab = tabs.PharosCalculationTab(self)
        elif name == 'Gecombineerd - Resultaten':
            tab = tabs.AdvancedCombinedResultsTab(self)
            
        # Toegevoegd Svasek 03/10/18 - Selecteer de Hares tab
        elif name == 'HARES - Rekenen':
            tab = tabs.HaresCalculationTab(self)

        elif name == 'Modelonzekerheden':
            tab = tabs.ModelUncertaintyTab(self)
        elif name == 'Exporteren':
            tab = tabs.ExportToDatabaseTab(self)
        else:
            raise ValueError('Tab name "{}" not recognized. Cannot open tab'.format(name))

        return tab


    def init_ui(self):

        self.init_menubar()

        self.mainhbox = QtWidgets.QHBoxLayout()

        self.setCentralWidget(QtWidgets.QWidget(self))
        self.centralWidget().setLayout(self.mainhbox)

        # self.statusBar().showMessage('Ready')

        self.setGeometry(400, 200, 1100, 700)

        self.show()

    def getIcon(self, iconname):
        """
        Retrieve Icon from data dir and set to GUI
        """
        if getattr(sys, 'frozen', False):
            # If the application is run as a bundle, the pyInstaller bootloader
            # extends the sys module by a flag frozen=True and sets the app 
            # path into variable _MEIPASS'.
            application_path = sys._MEIPASS
            datadir = os.path.join(application_path, 'data')
            
        else:
            application_path = os.path.dirname(os.path.abspath(__file__))
            datadir = os.path.join(application_path, '..', 'data')
            
        iconpath = os.path.join(datadir, 'icons', iconname)
        if not os.path.exists(iconpath):
            raise OSError('Icon in path: "{}" not found'.format(iconpath))

        return QtGui.QIcon(iconpath)

    def init_menubar(self):
        """
        Method to construct menu bar.
        """

        menubar = self.menuBar()

        new_action = QtWidgets.QAction(self.getIcon('new.png'), 'Nieuw', self)
        new_action.setShortcut(QtGui.QKeySequence.New)
        new_action.setStatusTip('Creeer een nieuw project.')
        new_action.triggered.connect(self.new_project)

        openAction = QtWidgets.QAction(self.getIcon('open.png'), 'Openen...', self)
        openAction.setStatusTip('Open bestaand project')
        openAction.setShortcut(QtGui.QKeySequence.Open)
        openAction.triggered.connect(self.open_project)

        saveAction = QtWidgets.QAction(self.getIcon('save.png'),'Opslaan',self)
        saveAction.setStatusTip('Bewaar project')
        saveAction.setShortcut(QtGui.QKeySequence.Save)
        saveAction.triggered.connect(self.save_project)

        saveasAction = QtWidgets.QAction(self.getIcon('saveas.png'),'Opslaan als...',self)
        saveasAction.setStatusTip('Bewaar project als')
        saveasAction.setShortcut(QtGui.QKeySequence.SaveAs)
        saveasAction.triggered.connect(self.save_project_as)

        exitAction = QtWidgets.QAction(self.getIcon('exit.png'), 'Afsluiten', self)
        exitAction.setShortcut('Ctrl+Q')
        exitAction.setStatusTip('HB Havens afsluiten')
        exitAction.triggered.connect(self.exit_hbhavens)

        manualAction = QtWidgets.QAction(self.getIcon('info.png'), 'Gebruikershandleiding...', self)
        exitAction.setShortcut('Ctrl+M')
        manualAction.setStatusTip('Gebruikershandleiding')
        manualAction.triggered.connect(self.open_manual)

        aboutAction = QtWidgets.QAction(self.getIcon('info.png'),'Over...',self)
        aboutAction.setShortcut(QtGui.QKeySequence.HelpContents)
        aboutAction.setStatusTip('Over HB Havens')
        aboutAction.triggered.connect(self.about)

        file_menu = menubar.addMenu('&Bestand')
        file_menu.addAction(new_action)
        file_menu.addAction(openAction)
        file_menu.addSeparator()
        file_menu.addAction(saveAction)
        file_menu.addAction(saveasAction)
        file_menu.addSeparator()
        file_menu.addAction(exitAction)

        help_menu = menubar.addMenu('&Help')
        help_menu.addAction(manualAction)
        file_menu.addSeparator()
        help_menu.addAction(aboutAction)

    def open_manual(self):

        handleiding = os.path.join(os.path.dirname(__file__), '..', '..', 'doc', 'Gebruikershandleiding', f"Gebruikershandleiding HB Havens - versie {self.mainmodel.appVersion}.pdf")
        handleiding = os.path.abspath(handleiding.lower())#.replace('/', '\\\\')
        url = QtCore.QUrl('file:///'+handleiding)
        if not QtGui.QDesktopServices.openUrl(url):
            QtWidgets.QMessageBox.warning(self, 'Openen gebruikershandleiding', f'Kan gebruikershandleiding niet openen vanaf: {handleiding}')
        
    def exit_hbhavens(self):
        """
        Check if project is saved before quit
        """
        if self.mainmodel.project.dirty:
            if not self.ok_to_continue():
                return None
        QtWidgets.qApp.quit()

    def closeEvent(self, event):
        if self.mainmodel.project.dirty:
            cont = self.ok_to_continue()
            if not cont and cont is not None:
                event.ignore()
                
    def open_project(self, *args, fname=None):
        """
        Method that loads project file and builds gui
        """
        if hasattr(self, 'mainmodel'):
            if not self.ok_to_continue():
                return None

        # Create main class
        self.mainmodel = MainModel()

        # Set open file dialog settings
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog

        # Set current dir
        currentdir = '.'
        if self.appsettings.value('currentdir'):
            currentdir = self.appsettings.value('currentdir')

        if fname is None:
            fname, _ = QtWidgets.QFileDialog.getOpenFileName(self, self.mainmodel.appName + ' - Open project', currentdir, "(*.json)", options=options)
        
        if fname == "":
            return None

        self.setCursorWait()
        # Load project
        self.mainmodel.project.open_from_file(fname)
        # save current dir
        self.appsettings.setValue('currentdir', os.path.dirname(fname))

        self.setCursorNormal()
        self.updateStatus('' )
        
        # Set progress step
        self.step = self.mainmodel.project.settings['project']['progress']

        # Initialize UI
        self.initMain()

        # Zoom map
        self.tabwidgets['Overzicht'].mapwidget.toolbar.home()
        self.tabwidgets['Overzicht'].mapwidget.toolbar.pan()

        # Add data to mapwidget and dataviewer
        if self.mainmodel.project.settings['simple']['finished']:
            # Add simple results to mapwidget and datawidget
            logger.info('Adding simple calculation results to visualisation tabs on loading.')
            self.tabwidgets['Overzicht'].add_table(method='simple')
            self.tabwidgets['Datavisualisatie'].add_table(method='simple')

        # Check if swan iterations are done
        if self.mainmodel.swan.calculation_results.initialized and not self.mainmodel.swan.iteration_results.empty:
            # Add swan iterations to mapwidget and datawidget
            logger.info('Adding SWAN iterations to visualisation tabs on loading.')
            self.tabwidgets['Datavisualisatie'].add_table(method='swan_iterations')

        # Check if swan calculations are done
        if not self.mainmodel.swan.calculation_results.empty and not self.mainmodel.swan.calculation_results['Wave direction swan'].isnull().all():
            # Add swan calculations to mapwidget and datawidget
            logger.info('Adding SWAN calculations to visualisation tabs on loading.')
            self.tabwidgets['Overzicht'].add_table(method='swan_calculations')
            self.tabwidgets['Datavisualisatie'].add_table(method='swan_calculations')

        # Check if pharos is done
        if not self.mainmodel.pharos.calculation_results.empty and not self.mainmodel.pharos.calculation_results['Wave direction totaal'].isnull().all():
            logger.info('Adding PHAROS calculations to visualisation tabs on loading.')
            self.tabwidgets['Overzicht'].add_table(method='pharos_calculations')
            self.tabwidgets['Datavisualisatie'].add_table(method='pharos_calculations')

        # Check if hares is done
        # if not self.mainmodel.hares.calculation_results.empty and not self.mainmodel.hares.calculation_results['Wave direction totaal'].isnull().all():
        #     logger.info('Adding HARES calculations to visualisation tabs on loading.')
        #     self.tabwidgets['Overzicht'].add_table(method='hares_calculations')
        #     self.tabwidgets['Datavisualisatie'].add_table(method='hares_calculations')


        self.clearDirty()
        self.updateStatus('')

    def new_project(self):
        """
        """
        if hasattr(self, 'mainmodel'):
            if not self.ok_to_continue():
                return

        # Create new main class
        self.mainmodel = MainModel()

        self.appsettings.setValue('name', self.mainmodel.appName)
        self.appsettings.setValue('version', self.mainmodel.appVersion)
        self.appsettings.setValue('date', self.mainmodel.appDate)

        # Initialize settings
        self.mainmodel.project.initSettings()
        # Reset step
        self.set_calculation_method('general')
        self.step = 0
        # Initialize main window
        self.initMain()
        self.updateStatus()

    def save_project_as(self):
        """
        """
        # Set dialog options
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog
        # Get file name from dialog
        fname, _ = QtWidgets.QFileDialog.getSaveFileName(self, self.mainmodel.appName + ' - Save project', '.', '(*.json)', options=options)
        # If a file name is given (not '')
        if fname:
            # Append extension if not given
            if not fname.endswith('.json'):
                fname += ".json"

            self.mainmodel.project.save_as(fname)
            self.mainmodel.save_tables()
            self.clearDirty()
            self.updateStatus()

    def save_project(self):
        """
        """

        # Wanneer er geen projectnaam bekend is, open opslaan als.
        if self.mainmodel.project.name is None:
            self.save_project_as()
            return None

        # Als er nog geen map bestaat met de projectnaam, open opslaan als
        if not os.path.exists(self.mainmodel.project.name):
            self.save_project_as()
            return None

        self.mainmodel.project.save()
        self.mainmodel.save_tables()
        self.clearDirty()
        self.updateStatus()

    def _go_to_step(self, newstep):
        """
        Method to continue to a given step.

        Parameters
        ----------
        newstep : int
            Index of the step to navigate to.

        Does the following:
        1. Open new tab
        2. Enable and disable navigation buttons
        3. Close old tab
        4. Increase step count (and progress overview)
        The order is important, since steps depend on each other.

        """
        # Open new tabs
        #----------------------------------------------------------------
        # Get name of the new tab
        tabname = self.steps[newstep]

        # Create the tab
        if tabname in self.tabwidgets.keys():
            # Focus on the tab
            logger.info('Focussing on tab {}'.format(tabname))
            self.tabsframe._open_or_focus(tabname)
        else:
            logger.info('Creating tab {}'.format(tabname))
            # Create the tab
            tab = self.init_tab(tabname)
            
            self.tabwidgets[tabname] = tab
            # Open and focus on tab
            self.tabsframe._open_or_focus(tabname)

        # Check step number with range and enable buttons
        #----------------------------------------------------------------
        self.previous_button.setEnabled(True)

        if newstep == 0:
            self.previous_button.setEnabled(False)

        # Close old tabs
        #----------------------------------------------------------------
        if self.step != newstep:
            self.tabsframe.close_by_name(self.steps[self.step])

        # Increase step count
        #----------------------------------------------------------------
        self.step = newstep
        self.mainmodel.project.settings['project']['progress'] = self.step
        self.progress_overview.set_current_step(self.step)

    def next_step(self):
        """
        Method to continue to next step
        """
        # Check if next step button is enabled
        currentTabname = self.steps[self.step]
        if self.next_button.isEnabled():
            self._go_to_step(self.step + 1)

    def previous_step(self):
        """
        Method to continue to previous step
        """
        # Check if previous step button is enabled
        if self.previous_button.isEnabled():
            self._go_to_step(self.step - 1)

    def about(self):

        
        text = f"""
        <h1>{self.mainmodel.appName}</h1><br>
        versie : {self.mainmodel.appVersion}<br>
        {self.mainmodel.appDate}<br>
        <br>
        &copy;2017 <a href="https://www.rijkswaterstaat.nl/over-ons/onze-organisatie/organisatiestructuur/water-verkeer-en-leefomgeving/index.aspx">Rijkwaterstaat-WVL</a><br>
        <br>
        ontwikkeld door : <br>
        &nbsp;&nbsp;<a href="https://www.hkv.nl">HKV lijn in water</a><br>
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&<br>
        &nbsp;&nbsp;<a href="http://www.aktishydraulics.com/">Aktis Hydraulics</a><br>
        """
        
        #Icon made by Alfredo Hernandez from www.flaticon.com
        dialogs.QuestionDialog.about(self,"Over " + self.mainmodel.appName, text)

    def updateStatus(self, message=None):
        """
        Update status
        """
        if self.mainmodel.project.name is not None:
            flbase = os.path.basename(self.mainmodel.project.name)
            self.setWindowTitle(self.mainmodel.appName + " - " + flbase + "[*]")
        else:
            self.setWindowTitle(self.mainmodel.appName + " [*]")
        self.setWindowModified(self.mainmodel.project.dirty)

    def ok_to_continue(self):
        """
        """
        if self.mainmodel.project.dirty:

            reply = QtWidgets.QMessageBox.question(
                self,
                "HB Havens - Opslaan wijzigingen",
                "Wijzigingen opslaan",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No | QtWidgets.QMessageBox.Cancel
            )
            if reply == QtWidgets.QMessageBox.Cancel:
                return False
            elif reply == QtWidgets.QMessageBox.Yes:
                return self.save_project()
            elif (reply == QtWidgets.QMessageBox.No):
                return True
        else:
            return True

    def setDirty(self):
        """
        Set project dirty status
        """
        self.mainmodel.project.dirty = True
        self.updateStatus()

    def clearDirty(self):
        """
        Clear project dirty status
        """
        self.mainmodel.project.dirty = False

    def resizeEvent(self, event):
        """
        Reload overview tab when window resizes
        """
        if hasattr(self, 'tabsframe'):
            if self.tabsframe.currentIndex() == self.tabsframe.tabindex['Overzicht']:
                self.tabsframe.tabwidgets['Overzicht'].mapwidget._update_background(None)
