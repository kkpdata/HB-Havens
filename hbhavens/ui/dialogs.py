import numpy as np
import pandas as pd
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
# Toegevoegd Svasek 31/10/2018 - Ander gebruik van figure, waardoor er in Spyder geen extra figuur opent
from matplotlib.figure import Figure
from PyQt5 import Qt, QtCore, QtGui, QtWidgets

from hbhavens import io
from hbhavens.ui import widgets

class GetPharosParametersDialog(QtWidgets.QDialog):
    """
    Dialog to get parameters with which the pharos table can be initialized
    """

    def __init__(self, parent=None):
        """
        Constructor
        """

        super(GetPharosParametersDialog, self).__init__(parent)

        # Refer elements from parent
        self.mainwindow = parent.mainwindow
        # Get all settings
        self.settings = parent.project.settings

        self.setWindowTitle('HB Havens: PHAROS')
        self.setWindowFlags(self.windowFlags() & ~QtCore.Qt.WindowContextHelpButtonHint)

        self.succeeded = False

        self.harbor_bedlevel = self.schematisation.bedlevel

        self.waterlevels = parent.hydraulic_loads[parent.hydraulic_loads.wlevcol].unique().tolist()
        
        # Get max Tp and max Hs
        self.f_range = parent.pharos.get_frequency_range()
        self.Hs_max = parent.hydraulic_loads['Hs'].max()

        # Built UI
        self.input_elements = {}
        self._init_ui()

        self.valid = False

        self.load_from_settings()

    def _init_ui(self):
        """
        Set up UI design
        """
        # Create GUI elements, set them in dict structure
        inf_symbol = u'\u221E'
        gamma_symbol = u'\u03B3'
        unicode_squared = u'\u00B9'
        labelwidth=175

        if len(self.waterlevels) > 100:
            raise NotImplementedError('More than 100 water levels where discovered in the hydraulic loads. The method with PHAROS is not implemented for this number of loads. Recalculate the wave conditions at given water levels, or pick a method without PHAROS.')

        self.input_elements['hydraulic loads'] = {
            'Hs_max': widgets.ParameterLabel(
                label='Max. significante golfhoogte:',
                labelwidth=labelwidth,
                value='{:.3f}'.format(self.Hs_max),
                unit='m'
            ),
            # 'Tp_max': widgets.ParameterLabel(
            #     label='Maximale piekperiode',
            #     labelwidth=150
            # ),
            'factor Tm Tp': widgets.ParameterInputLine(
                label='Factor Tm naar Tp:',
                labelwidth=labelwidth,
                validator=QtGui.QDoubleValidator(0.01, 99.99, 20),
            ),
            'water depth for wave length': widgets.ParameterInputLine(
                label='Waterdiepte voor golflengte:',
                labelwidth=labelwidth,
                unitlabel='m',
                validator=QtGui.QDoubleValidator(0.00, np.inf, 20),
            ),
        }

        self.input_elements['wave directions'] = {
            'lowest': widgets.ParameterInputLine(
                label='Laagste waarde [0-360]:',
                labelwidth=labelwidth,
                unitlabel='graden (nautisch)',
                validator=QtGui.QDoubleValidator(0.00, 360.00, 20),
            ),
            'highest': widgets.ParameterInputLine(
                label='Hoogste waarde [0-360]:',
                labelwidth=labelwidth,
                unitlabel='graden (nautisch)',
                validator=QtGui.QDoubleValidator(0.00, 360.00, 20),
            ),
            'bin size': widgets.ParameterInputLine(
                label='Klassegrootte [1-360]:',
                labelwidth=labelwidth,
                unitlabel='graden',
                validator=QtGui.QDoubleValidator(1.00, 360.00, 20),
            )
        }

        self.input_elements['frequencies'] = {
            'lowest': widgets.ParameterInputLine(
                label='Ondergrens [{:.3f} - {:.3f}]:'.format(*self.f_range),
                labelwidth=labelwidth,
                unitlabel='Hz',
                validator=QtGui.QDoubleValidator(self.f_range[0] - 0.01, self.f_range[1] + 0.01, 20),
            ),
            'highest': widgets.ParameterInputLine(
                label='Bovengrens [{:.3f} - {:.3f}]:'.format(*self.f_range),
                labelwidth=labelwidth,
                unitlabel='Hz',
                validator=QtGui.QDoubleValidator(self.f_range[0] - 0.01, self.f_range[1] + 0.01, 20),
            ),
            'number of bins': widgets.ParameterInputLine(
                label='Aantal klassen [1-50]:',
                labelwidth=labelwidth,
                validator=QtGui.QIntValidator(1, 50),
            ),
            'scale': widgets.ComboboxInputLine(
                label='Frequentie schaal:',
                labelwidth=labelwidth,
                items=['lineair', 'logaritmisch'],
            )
        }

        self.input_elements['2d wave spectrum'] = {
            'spread': widgets.ParameterInputLine(
                label='Spreiding [10-70]:',
                labelwidth=labelwidth,
                unitlabel='graden',
                validator=QtGui.QDoubleValidator(10.0, 70.0, 20),
            ),
            'gamma': widgets.ParameterInputLine(
                label='JONSWAP peak\nenhancement factor {} [1-7]:'.format(gamma_symbol),
                labelwidth=labelwidth,
                unitlabel='',
                validator=QtGui.QDoubleValidator(1.00, 7.00, 20),
            ),
            'min energy': widgets.ParameterInputLine(
                label='Signaleringswaarde energie [0-{}]:'.format(inf_symbol),
                labelwidth=labelwidth,
                unitlabel='m{}s/degree'.format(unicode_squared),
                validator=QtGui.QDoubleValidator(0.00, 2.00, 20),
            )
        }

        self.input_elements['paths'] = {
            'pharos folder': widgets.ExtendedLineEdit(
                label='Uitvoermap:',
                labelwidth=labelwidth,
                browsebutton=QtWidgets.QPushButton(
                    '...',
                    clicked=self._load_pharos_folder
                )
            ),
            'schematisation folder': widgets.ExtendedLineEdit(
                label='Schematisatiemap:',
                labelwidth=labelwidth,
                browsebutton=QtWidgets.QPushButton(
                    '...',
                    clicked=self._load_schematisations_folder
                )
            )
        }

        self.input_elements['water levels'] = {
            'checked': widgets.CheckBoxInput(
                labels=self.waterlevels,
                nrows=max(2, len(self.waterlevels) // 20),
                unitlabel='m + NAP'
            )
        }

        delta = u'\u0394'
        self.input_elements['transformation'] = {
            'dx': widgets.ParameterInputLine(
                label='{}x [RD + {}x = lokaal]:'.format(delta, delta),
                labelwidth=labelwidth,
            ),
            'dy': widgets.ParameterInputLine(
                label='{}y [RD + {}y = lokaal]:'.format(delta, delta),
                labelwidth=labelwidth,
            )
        }


        # Define titles for groups
        titles = {
            'hydraulic loads': 'Hydraulische belastingen',
            'wave directions': 'Golfrichtingen',
            'frequencies': 'Frequenties',
            '2d wave spectrum': '2D golfspectrum',
            'paths': 'Paden',
            'water levels': 'Te simuleren waterstanden',
            'transformation': 'Transformatie voor coordinatenstelsel'
        }


        # Create base layout
        self.setLayout(QtWidgets.QVBoxLayout())
        # self.layout().setSpacing(10)

        for tag, title in titles.items():
            if tag in self.input_elements:
                group_layout = QtWidgets.QVBoxLayout()
                for _, item in self.input_elements[tag].items():
                    group_layout.addWidget(item)

            # Add groupbox with title
            groupbox = QtWidgets.QGroupBox(title)
            groupbox.setLayout(group_layout)
            self.layout().addWidget(groupbox)


        line = QtWidgets.QFrame()
        line.setFrameShape(QtWidgets.QFrame.HLine)
        line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.layout().addWidget(line)

        # OK and Cancel buttons

        self.generate_button = QtWidgets.QPushButton('Genereer tabel')
        self.generate_button.setDefault(True)
        # self.generate_button.setEnabled(False)
        self.generate_button.clicked.connect(self.generate)

        self.cancel_button = QtWidgets.QPushButton('Annuleren')
        self.cancel_button.setAutoDefault(False)
        self.cancel_button.clicked.connect(self.cancel)

        button_box = QtWidgets.QDialogButtonBox(QtCore.Qt.Horizontal, self)
        button_box.addButton(self.generate_button, QtWidgets.QDialogButtonBox.ActionRole)
        button_box.addButton(self.cancel_button, QtWidgets.QDialogButtonBox.RejectRole)

        button_box.accepted.connect(QtWidgets.QDialog.accept)
        # button_box.rejected.connect(QtWidgets.QDialog.reject)

        self.layout().addWidget(button_box)


    def generate(self):
        """
        User pressed Generate button
        """
        # Validate the input
        self.validate()

        # Return none if validation failed
        if not self.valid:
            return None

        # Save to settings
        self.save_to_settings()

        # Set project dirty
        self.mainwindow.setDirty()

        # Set project succeeded
        self.succeeded = True
        self.accept()

    def cancel(self):
        """
        User pressed Cancel button
        """
        self.succeeded = False
        self.reject()

    def validate_parameter(self, val):
        """
        Validate parameter based on 3 conditions:
        1. is not nan
        2. is not None
        3. is not ''
        """
        valid = True
        # Lists are not checked
        if isinstance(val, list):
            return valid

        if isinstance(val, (float, int, np.integer, np.floating)):
            if np.isnan(val):
                valid = False
        if (val is None) or (val == ''):
            valid = False

        return valid

    def validate(self, check_empty=True):
        """
        Test for correct input in this dialog

        Parameters
        ----------
        check_empty : boolean
            Whether to check the empty parameters. This is not done when the table
            is loaded from settings, since it then does not matter if there
            are empty cells.
        """

        # Collect invalid parameters
        invalid = []
        parameters = []

        # Loop trough input elements and validate
        for group in self.input_elements.values():
            for widget in group.values():
                val = widget.get_value()
                # If not checking the empty parameters, if a parameter is empty, continue
                if not check_empty:
                    if not self.validate_parameter(val):
                        continue

                # Check if item is filled
                if not self.validate_parameter(val):
                    invalid.append(widget.label.replace(':', ''))
                    parameters.append(val)

                # If the item has a validator, check if the value is valid
                elif hasattr(widget, 'validator'):
                    if widget.validator is not None:
                        if widget.validator.validate(val, 1)[0] != 2:
                            invalid.append(widget.label.replace(':', ''))
                            parameters.append(val)

        if len(invalid) == 1:
            NotificationDialog('Er is geen geldige waarde voor "{}" ingevuld: {}'.format(invalid[0], parameters))
            self.valid = False
        elif len(invalid) > 1:
            NotificationDialog('Er zijn geen geldige waarden voor "{}" ingevuld: {}'.format('", "'.join(invalid), parameters))
            self.valid = False
        else:
            # If no error
            self.valid = True

    def save_to_settings(self):
        """
        Save parameters to settings. The settings have the same layout as the input_elements dict
        """
        for group_name, group in self.input_elements.items():
            for param, widget in group.items():
                val = widget.get_value()
                # Convert value to integer of float
                try:
                    val = float(val)
                    if val.is_integer():
                        val = int(val)
                except:
                    pass
                self.settings['pharos'][group_name][param] = val

    def load_from_settings(self):
        """
        Save parameters to settings. The settings have the same layout as the input_elements dict
        """
        for group_name, group in self.settings['pharos'].items():
            # Some settings do not have a GUI element, continue if encountered
            if group_name not in self.input_elements.keys():
                continue

            for param, value in group.items():
                # Some settings do not have a GUI element, continue if encountered
                if param not in self.input_elements[group_name].keys():
                    continue
                # Check if parameter is not empty before filling in
                if self.validate_parameter(value):
                    self.input_elements[group_name][param].set_value(value)

        # Validate
        self.validate(check_empty=False)

    def _load_schematisations_folder(self):
        """
        Get SWAN output folder
        """
        # Get path
        path = QtWidgets.QFileDialog().getExistingDirectory(self, 'PHAROS schematisaties follder')
        if not path:
            return None

        self.input_elements['paths']['schematisation folder'].set_value(path)

    def _load_pharos_folder(self):
        """
        Get PHAROS output folder
        """
        # Get path
        path = QtWidgets.QFileDialog().getExistingDirectory(self, 'PHAROS werkfolder')
        if not path:
            return None

        self.input_elements['paths']['pharos folder'].set_value(path)

class ChooseFloodDefenceWindow(QtWidgets.QDialog):
    """
    Dialog to add one or more flood defences to the harbor schematisation and
    GUI
    """

    def __init__(self, parent=None):
        """
        Constructor

        Parameters
        ----------
        parent : SchematisationTab
            Parent class, the schematisation widget tab
        """


        super(ChooseFloodDefenceWindow, self).__init__(parent)
        self.parent = parent
        self.datadir = parent.mainmodel.datadir

        self._init_ui()

    def _init_ui(self):
        """
        Set up UI design
        """

        hlayout = QtWidgets.QHBoxLayout()

        label = QtWidgets.QLabel('Kies een normtraject:')

        hlayout.addWidget(label)

        self.section_combobox = QtWidgets.QComboBox()
        self.section_ids = sorted([''] + [x.split('-')[0].zfill(2) + '-' + x.split('-')[1] if '-' in x else x for x in io.geometry.import_section_ids(self.datadir)])
        self.section_combobox.addItems(self.section_ids)

        hlayout.addWidget(self.section_combobox)

        self.add_button = QtWidgets.QPushButton('Toevoegen', clicked=self._add_flooddefence)

        hlayout.addWidget(self.add_button)

        vlayout = QtWidgets.QVBoxLayout()
        vlayout.addLayout(hlayout)

        line = QtWidgets.QFrame()
        line.setFrameShape(QtWidgets.QFrame.HLine)
        line.setFrameShadow(QtWidgets.QFrame.Sunken)

        vlayout.addWidget(line)

        self.close_button = QtWidgets.QPushButton('Sluiten', clicked=self.close)
        vlayout.addWidget(self.close_button, 0, QtCore.Qt.AlignRight)

        self.setLayout(vlayout)

        self.setWindowTitle("HB Havens: normtrajecten")
        self.setWindowFlags(self.windowFlags() & ~QtCore.Qt.WindowContextHelpButtonHint)

    def _add_flooddefence(self):
        """
        Add a flood defence to the schematisation, and add it to the GUI label
        """
        section_id = self.section_combobox.currentText()
        if not section_id:
            return None
        if section_id in self.parent.settings['schematisation']['flooddefence_ids']:
            return None
        # Add to data model
        self.parent.schematisation.add_flooddefence(section_id)

        # Adjust GUI
        self.parent.set_flood_defence_ids()
        self.parent.mainwindow.setDirty()
        self.parent.overview_tab.mapwidget.set_visible('flooddefence')

class RemoveFloodDefenceWindow(QtWidgets.QDialog):

    def __init__(self, parent=None):

        super(RemoveFloodDefenceWindow, self).__init__(parent)
        self.parent = parent
        self.settings = parent.settings
        self.section_ids = []
        self._init_ui()

    def _init_ui(self):
        """
        Set up UI design
        """
        hlayout = QtWidgets.QHBoxLayout()

        hlayout.addWidget(QtWidgets.QLabel('Kies een normtraject:'))

        self.section_combobox = QtWidgets.QComboBox()
        self._update_combobox()

        hlayout.addWidget(self.section_combobox)

        self.remove_button = QtWidgets.QPushButton('Verwijderen', clicked=self._del_flooddefence)
        hlayout.addWidget(self.remove_button)

        vlayout = QtWidgets.QVBoxLayout()
        vlayout.addLayout(hlayout)

        line = QtWidgets.QFrame()
        line.setFrameShape(QtWidgets.QFrame.HLine)
        line.setFrameShadow(QtWidgets.QFrame.Sunken)

        vlayout.addWidget(line)

        self.close_button = QtWidgets.QPushButton('Sluiten', clicked=self.close)
        vlayout.addWidget(self.close_button, 0, QtCore.Qt.AlignRight)

        self.setLayout(vlayout)

        self.setWindowTitle("HB Havens: normtrajecten")
        self.setWindowFlags(self.windowFlags() & ~QtCore.Qt.WindowContextHelpButtonHint)

    def _update_combobox(self):
        """
        Method to update the combobox with flooddefences, on initializing and
        when a section is removed
        """
        self.section_combobox.clear()
        self.section_ids = sorted(self.settings['schematisation']['flooddefence_ids'])
        self.section_combobox.addItems(self.section_ids)
        self.section_combobox.setCurrentText('')

    def _del_flooddefence(self):
        """
        Remove a flood defence from the schematisation, and GUI
        """
        section_id = self.section_combobox.currentText()
        if section_id:
            # Remove the data from model, and add it again if left
            self.parent.schematisation.del_flooddefence(section_id)

            # Adjust GUI
            self.parent.set_flood_defence_ids()
            self.parent.mainwindow.setDirty()
            self.parent.overview_tab.mapwidget.set_visible('flooddefence')
            # Update combobox
            self._update_combobox()

class NotificationDialog(QtWidgets.QMessageBox):

    def __init__(self, text, severity='information', details=''):
        """
        Create a notification dialog with a given text, severity level
        and if needed some details.
        
        Parameters
        ----------
        text : str
            Message text
        severity : str, optional
            Severity level, warning, critical or information (default) 
        details : str, optional
            Optional details, by default ''
        """

        super().__init__()

        self.setText(text)

        if severity == 'warning':
            icon = self.style().standardIcon(QtWidgets.QStyle.SP_MessageBoxWarning)
            self.setWindowTitle("Waarschuwing")
        elif severity == 'critical':
            icon = self.style().standardIcon(QtWidgets.QStyle.SP_MessageBoxCritical)
            self.setWindowTitle("Foutmelding")
        elif severity == 'information':
            icon = self.style().standardIcon(QtWidgets.QStyle.SP_MessageBoxInformation)
            self.setWindowTitle("Mededeling")

        self.setIconPixmap(icon.pixmap(icon.actualSize(QtCore.QSize(36, 36))))

        # self.setWindowIcon(get_icon())

        if details:
            self.setDetailedText("Details:\n{}".format(details))

        self.setStandardButtons(QtWidgets.QMessageBox.Ok)
        self.resize(200, 400)

        self.exec_()

    def resizeEvent(self, event):
        """
        Method called when the details are opened, and the window is enlarged.
        """

        result = super(NotificationDialog, self).resizeEvent(event)

        details_box = self.findChild(QtWidgets.QTextEdit)
        # 'is not' is better style than '!=' for None
        if details_box is not None:
            details_box.setFixedSize(500, 250)

        return result

def show_dialog(text, severity='information', details=''):
    msg = QtWidgets.QMessageBox()

    if len(text) < 100:
        text = text+' '*(100-len(text))

    msg.setText(text)

    if severity == 'warning':
        msg.setIcon(QtWidgets.QMessageBox.Warning)
        msg.setWindowTitle("HB Havens: Waarschuwing")
    elif severity == 'critical':
        msg.setIcon(QtWidgets.QMessageBox.Critical)
        msg.setWindowTitle("HB Havens: Foutmelding")
    elif severity == 'information':
        msg.setIcon(QtWidgets.QMessageBox.Information)
        msg.setWindowTitle("HB Havens: Mededeling")


    if details:
        msg.setDetailedText("Details:\n{}".format(details))

    # msg.findChild(QtWidgets.QGridLayout).setColumnMinimumWidth(1, 250)
    msg.setStandardButtons(QtWidgets.QMessageBox.Ok)

    msg.exec_()

class GetSwanParametersDialog(QtWidgets.QDialog):
    """
    Dialog window to get:
    1. path to master template
    2. path to swan folder tree
    and generate the swan folder tree on button click.
    """

    def __init__(self, parent=None):

        super(GetSwanParametersDialog, self).__init__(parent)

        self.parent_tab = parent
        self.swan = parent.swan
        self.hydraulic_loads = parent.hydraulic_loads
        self.mainwindow = parent.mainwindow
        self.settings = parent.settings
        self.setWindowTitle('HB Havens: SWAN')
        self.setWindowFlags(self.windowFlags() & ~QtCore.Qt.WindowContextHelpButtonHint)

        self.succeeded = False

        self.input_elements = {}
        self.init_ui()
        self.valid = False
        self.load_from_settings()

    def load_from_settings(self):
        """
        Save parameters to settings. The settings have the same layout as the input_elements dict
        """
        for param, value in self.settings['swan'].items():
            # Some settings do not have a GUI element, continue if encountered
            if param not in self.input_elements.keys():
                continue

            # Check if parameter is not empty before filling in
            if self.validate_parameter(value):
                self.input_elements[param].set_value(value)

        # Validate
        self.validate(check_empty=False)

    def init_ui(self):
        """
        Set up UI design
        """
        # Create GUI elements, set them in dict structure
        labelwidth = 150

        # Add parameter line edit for Factor Tm to Tp
        self.input_elements['factor Tm Tp'] = widgets.ParameterInputLine(
            label='Factor Tm naar Tp:',
            labelwidth=labelwidth,
            unitlabel='(NVT: Tp aanwezig)' if 'Tp' in self.hydraulic_loads.columns else '',
            validator=QtGui.QDoubleValidator(0.01, 99.99, 20),
        )

        if 'Tp' in self.hydraulic_loads.columns or self.parent_tab.step != 'I1':
            self.input_elements['factor Tm Tp'].set_enabled(False)

        # Add line edit with browsebutton for Master template
        self.input_elements['mastertemplate'] = widgets.ExtendedLineEdit(
            label='Master template bestand:',
            labelwidth=labelwidth,
            browsebutton=QtWidgets.QPushButton('...', clicked=self.select_master_template)
        )

        # Add line edit with browsebutton for depth file
        self.input_elements['depthfile'] = widgets.ExtendedLineEdit(
            label='Bathymetry bestand:',
            labelwidth=labelwidth,
            browsebutton=QtWidgets.QPushButton('...', clicked=self.select_bathymetry_file)
        )

        # Add line edit with browsebutton for swan result folder
        self.input_elements['swanfolder'] = widgets.ExtendedLineEdit(
            label='SWAN uitvoer folder:',
            labelwidth=labelwidth,
            browsebutton=QtWidgets.QPushButton('...', clicked=self.select_swan_folder)
        )


        self.setLayout(QtWidgets.QVBoxLayout())
        self.layout().setSpacing(10)

        for _, item in self.input_elements.items():
            self.layout().addWidget(item)

        line = QtWidgets.QFrame()
        line.setFrameShape(QtWidgets.QFrame.HLine)
        line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.layout().addWidget(line)

        # OK and Cancel buttons
        self.generateButton = QtWidgets.QPushButton('Genereer invoer')
        self.generateButton.setDefault(True)
        self.generateButton.clicked.connect(self.generate)

        self.cancelButton = QtWidgets.QPushButton('Annuleren')
        self.cancelButton.setAutoDefault(False)
        self.cancelButton.clicked.connect(self.cancel)

        button_box = QtWidgets.QDialogButtonBox(QtCore.Qt.Horizontal, self)
        button_box.addButton(self.generateButton, QtWidgets.QDialogButtonBox.ActionRole)
        button_box.addButton(self.cancelButton, QtWidgets.QDialogButtonBox.RejectRole)
        button_box.accepted.connect(QtWidgets.QDialog.accept)

        self.layout().addWidget(button_box)

    def generate(self):
        """
        User pressed Generate button
        """
        # Validate the input
        self.validate()

        # Return none if validation failed
        if not self.valid:
            return None

        # Save to settings
        self.save_to_settings()

        # Set project dirty
        self.mainwindow.setDirty()

        # Set project succeeded
        self.succeeded = True
        self.accept()

    def cancel(self):
        """
        User pressed Cancel button
        """
        self.succeeded = False
        self.reject()

    def validate_parameter(self, val):
        """
        Validate parameter based on 3 conditions:
        1. is not nan
        2. is not None
        3. is not ''
        """
        valid = True
        # Lists are not checked
        if isinstance(val, list):
            return valid

        if isinstance(val, (float, int, np.integer, np.floating)):
            if np.isnan(val):
                valid = False
        if (val is None) or (val == ''):
            valid = False

        return valid


    def validate(self, check_empty=True):
        """
        Test for correct input in this dialog

        Parameters
        ----------
        check_empty : boolean
            Whether to check the empty parameters. This is not done when the table
            is loaded from settings, since it then does not matter if there
            are empty cells.
        """

        # Collect invalid parameters
        invalid = []

        # Loop trough input elements and validate
        for widget in self.input_elements.values():
            val = widget.get_value()
            # If not checking the empty parameters, if a parameter is empty, continue
            if not check_empty:
                if not self.validate_parameter(val):
                    continue

            # Check if item is filled
            if not self.validate_parameter(val):
                invalid.append(widget.label.replace(':', ''))

            # If the item has a validator, check if the value is valid
            elif hasattr(widget, 'validator'):
                if widget.validator is not None:
                    if widget.validator.validate(val, 0)[0] != 2:
                        invalid.append(widget.label.replace(':', ''))

        if len(invalid) == 1:
            NotificationDialog('Er is geen geldige waarde voor "{}" ingevuld.'.format(invalid[0]))
            self.valid = False

        elif len(invalid) > 1:
            NotificationDialog('Er zijn geen geldige waarden voor "{}" ingevuld.'.format('", "'.join(invalid)))
            self.valid = False
        else:
            # If no error
            self.valid = True

    def save_to_settings(self):
        """
        Save parameters to settings. The settings have the same layout as the input_elements dict
        """
        for param, widget in self.input_elements.items():
            val = widget.get_value()
            # Convert value to integer of float
            try:
                val = float(val)
                if val.is_integer():
                    val = int(val)
            except:
                pass
            self.settings['swan'][param] = val


    def select_master_template(self):
        """
        Get master template for this project
        """
        # Get path
        path, file_type = QtWidgets.QFileDialog.getOpenFileName(self, 'Open bestand', '', "SWN bestand (*.swn)")
        if not path:
            return None

        # Save path to project structure
        self.input_elements['mastertemplate'].set_value(path)

    def select_bathymetry_file(self):
        """
        Get bathymetry file for this project
        """
        # Get path
        path, file_type = QtWidgets.QFileDialog.getOpenFileName(self, 'Open bestand', '', "DEP bestand (*.dep)")
        if not path:
            return None

        # Save path to project structure
        self.input_elements['depthfile'].set_value(path)

    def select_swan_folder(self):
        """
        Get SWAN output folder
        """
        # Get path
        path = QtWidgets.QFileDialog().getExistingDirectory(self, 'SWAN werkfolder')
        if not path:
            return None

        self.input_elements['swanfolder'].set_value(path)

# Toegevoegd Svasek 08/10/18 - Scherm dat vraagt om de locatie van de Hares uitvoer
class GetHaresFolderDialog(QtWidgets.QDialog):
    """
    Dialog window to get:
    1. path to Hares folder
    """

    def __init__(self, parent=None):

        super(GetHaresFolderDialog, self).__init__(parent)

        self.parent_tab = parent
        self.hares = parent.hares
        self.mainwindow = parent.mainwindow
        self.settings = parent.settings
        self.setWindowTitle('HB Havens: HARES')
        self.setWindowFlags(self.windowFlags() & ~QtCore.Qt.WindowContextHelpButtonHint)

        self.succeeded = False

        self.input_elements = {}
        self.init_ui()
        self.valid = False
        self.load_from_settings()

    def load_from_settings(self):
        """
        Save parameters to settings. The settings have the same layout as the input_elements dict
        """
        for param, value in self.settings['hares'].items():
            # Some settings do not have a GUI element, continue if encountered
            if param not in self.input_elements.keys():
                continue

            # Check if parameter is not empty before filling in
            if self.validate_parameter(value):
                self.input_elements[param].set_value(value)

        # Validate
        self.validate(check_empty=False)

    def init_ui(self):
        """
        Set up UI design
        """
        # Create GUI elements, set them in dict structure
        labelwidth = 150

        # Add parameter line edit for Factor Tm to Tp

        # Add line edit with browsebutton for swan result folder
        self.input_elements['hares folder'] = widgets.ExtendedLineEdit(
            label='HARES uitvoerbestanden folder:',
            labelwidth=labelwidth,
            browsebutton=QtWidgets.QPushButton('...', clicked=self.select_hares_folder)
        )


        self.setLayout(QtWidgets.QVBoxLayout())
        self.layout().setSpacing(10)

        for _, item in self.input_elements.items():
            self.layout().addWidget(item)

        line = QtWidgets.QFrame()
        line.setFrameShape(QtWidgets.QFrame.HLine)
        line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.layout().addWidget(line)

        # OK and Cancel buttons
        self.generateButton = QtWidgets.QPushButton('Start lezen uitvoerbestanden')
        self.generateButton.setDefault(True)
        self.generateButton.clicked.connect(self.generate)

        self.cancelButton = QtWidgets.QPushButton('Annuleren')
        self.cancelButton.setAutoDefault(False)
        self.cancelButton.clicked.connect(self.cancel)

        button_box = QtWidgets.QDialogButtonBox(QtCore.Qt.Horizontal, self)
        button_box.addButton(self.generateButton, QtWidgets.QDialogButtonBox.ActionRole)
        button_box.addButton(self.cancelButton, QtWidgets.QDialogButtonBox.RejectRole)
        button_box.accepted.connect(QtWidgets.QDialog.accept)

        self.layout().addWidget(button_box)

    def generate(self):
        """
        User pressed Generate button
        """
        # Validate the input
        self.validate()

        # Return none if validation failed
        if not self.valid:
            return None

        # Save to settings
        self.save_to_settings()

        # Set project dirty
        self.mainwindow.setDirty()

        # Set project succeeded
        self.succeeded = True
        self.accept()

    def cancel(self):
        """
        User pressed Cancel button
        """
        self.succeeded = False
        self.reject()

    def validate_parameter(self, val):
        """
        Validate parameter based on 3 conditions:
        1. is not nan
        2. is not None
        3. is not ''
        """
        valid = True
        # Lists are not checked
        if isinstance(val, list):
            return valid

        if isinstance(val, (float, int, np.integer, np.floating)):
            if np.isnan(val):
                valid = False
        if (val is None) or (val == ''):
            valid = False

        return valid


    def validate(self, check_empty=True):
        """
        Test for correct input in this dialog

        Parameters
        ----------
        check_empty : boolean
            Whether to check the empty parameters. This is not done when the table
            is loaded from settings, since it then does not matter if there
            are empty cells.
        """

        # Collect invalid parameters
        invalid = []

        # Loop trough input elements and validate
        for widget in self.input_elements.values():
            val = widget.get_value()
            # If not checking the empty parameters, if a parameter is empty, continue
            if not check_empty:
                if not self.validate_parameter(val):
                    continue

            # Check if item is filled
            if not self.validate_parameter(val):
                invalid.append(widget.label.replace(':', ''))

            # If the item has a validator, check if the value is valid
            elif hasattr(widget, 'validator'):
                if widget.validator is not None:
                    if widget.validator.validate(val, 0)[0] != 2:
                        invalid.append(widget.label.replace(':', ''))

        if len(invalid) == 1:
            NotificationDialog('Er is geen geldige waarde voor "{}" ingevuld.'.format(invalid[0]))
            self.valid = False

        elif len(invalid) > 1:
            NotificationDialog('Er zijn geen geldige waarden voor "{}" ingevuld.'.format('", "'.join(invalid)))
            self.valid = False
        else:
            # If no error
            self.valid = True

    def save_to_settings(self):
        """
        Save parameters to settings. The settings have the same layout as the input_elements dict
        """
        for param, widget in self.input_elements.items():
            val = widget.get_value()
            # Convert value to integer of float
            try:
                val = float(val)
                if val.is_integer():
                    val = int(val)
            except:
                pass
            self.settings['hares'][param] = val


    def select_hares_folder(self):
        """
        Get HARES output folder
        """
        # Get path
        path = QtWidgets.QFileDialog().getExistingDirectory(self, 'HARES uitvoerbestanden folder')
        if not path:
            return None

        self.input_elements['hares folder'].set_value(path)

class GetPharosSchematisationParametersDialog(QtWidgets.QDialog):

    def __init__(self, parent=None):

        super(GetPharosSchematisationParametersDialog, self).__init__(parent)

        # Get parent classes
        self.main = parent

        self.mainmodel = self.main.mainmodel
        self.pharos = self.mainmodel.pharos

        self.waterlevels = self.mainmodel.hydraulic_loads[self.mainmodel.hydraulic_loads.wlevcol].unique()

        self.methodAdvancedSettings = self.main.mainmodel.project.getGroupSettings('advanced')
        self.pharos_settings = {}
        self.setWindowTitle('HB Havens: PHAROS')
        self.setWindowFlags(self.windowFlags() & ~QtCore.Qt.WindowContextHelpButtonHint)

        self.succeeded = False

        # Get PHAROS settings and fill controls
        self.pharos_settings = self.methodAdvancedSettings['PHAROS']
        self._initUI(self.pharos_settings['schematisationsfolder'])

    def _initUI(self, folder):
        """
        Set up UI design
        """

        vlayout = QtWidgets.QVBoxLayout()

        # Pharos schematisaties
        #----------------------------------------------------------------

        schematisationsFrame = QtWidgets.QGroupBox()
        schematisationsFrame.setTitle('Kies Pharos schematisaties')

        self.listSchematisations = QtWidgets.QListView()

        # Create an empty model for the list's data
        schematisationModel = QtGui.QStandardItemModel(self.listSchematisations)

        schematisations = self.pharos.pharosio.getSchematisations(folder)
        for schematisation in schematisations:
            # create an item with a caption
            item = QtGui.QStandardItem(schematisation)
            # Add a checkbox to it
            item.setCheckable(True)
            # Set default true
            item.setCheckState(QtCore.Qt.Checked)
            # Add the item to the model
            schematisationModel.appendRow(item)

        # Apply the model to the list view
        self.listSchematisations.setModel(schematisationModel)

        ## Show the window and run the app
        vlayout.addWidget(self.listSchematisations)

        schematisationsFrame.setLayout(vlayout)


        # Water levels
        #----------------------------------------------------------------
        waterlevelsframe = QtWidgets.QGroupBox()
        waterlevelsframe.setTitle('Kies waterstanden')

        vlayout = QtWidgets.QVBoxLayout()
        self.listWaterlevels = QtWidgets.QListView()
        # Create an empty model for the list's data
        waterlevelsModel = QtGui.QStandardItemModel(self.listWaterlevels)

        for waterlevel in self.waterlevels:
            # create an item with a caption
            item = QtGui.QStandardItem(str(waterlevel.round(2)))
            # Add a checkbox to it
            item.setCheckable(True)
            # Set default true
            item.setCheckState(QtCore.Qt.Checked)
            # Add the item to the model
            waterlevelsModel.appendRow(item)

        # Apply the model to the list view
        self.listWaterlevels.setModel(waterlevelsModel)

        vlayout.addWidget(self.listWaterlevels)

        waterlevelsframe.setLayout(vlayout)

        vbox = QtWidgets.QVBoxLayout()
        vbox.addWidget(schematisationsFrame)
        vbox.addWidget(waterlevelsframe)

        self.setLayout(vbox)

        # OK and Cancel buttons

        self.generateButton = QtWidgets.QPushButton('OK')
        self.generateButton.setDefault(True)
        self.generateButton.clicked.connect(self.ok)

        self.cancelButton = QtWidgets.QPushButton('Annuleren')
        self.cancelButton.setAutoDefault(False)
        self.cancelButton.clicked.connect(self.cancel)

        button_box = QtWidgets.QDialogButtonBox(QtCore.Qt.Horizontal, self)
        button_box.addButton(self.generateButton, QtWidgets.QDialogButtonBox.ActionRole)
        button_box.addButton(self.cancelButton, QtWidgets.QDialogButtonBox.RejectRole)

        button_box.accepted.connect(QtWidgets.QDialog.accept)
        # button_box.rejected.connect(QtWidgets.QDialog.reject)

        self.layout().addWidget(button_box)

    def ok(self):
        """
        User pressed OK button
        """
        # Save parameters
        self.runschematisations = ['geen']
        schematisationModel = self.listSchematisations.model()
        for index in range(schematisationModel.rowCount()):
            item = schematisationModel.item(index)
            if item.isCheckable() and item.checkState() == QtCore.Qt.Checked:
                self.runschematisations.append(item.text())

        self.runwaterlevels = []
        waterlevelsModel = self.listWaterlevels.model()
        for index in range(waterlevelsModel.rowCount()):
            item = waterlevelsModel.item(index)
            if item.isCheckable() and item.checkState() == QtCore.Qt.Checked:
                self.runwaterlevels.append(item.text())

        self.succeeded = True
        self.accept()

    def cancel(self):
        """
        User pressed Cancel button
        """
        self.succeeded = False
        self.reject()


class DefineUncertaintiesDialog(QtWidgets.QDialog):

    def __init__(self, parent=None):

        super(DefineUncertaintiesDialog, self).__init__(parent)

        self.modeluncertainties = parent.modeluncertainties
        self.supportloc_unc = self.modeluncertainties.supportloc_unc
        self.harbor_unc = self.modeluncertainties.harbor_unc
        self.combined_unc = self.modeluncertainties.combined_unc

        self.location_names = self.modeluncertainties.table['Naam'].tolist()
        self.parent = parent
        self._initUI()

        self.succeeded = False

    def _initUI(self):
        """
        Set up UI design
        """

        self.setWindowTitle("HB Havens: onzekerheden")
        self.setWindowFlags(self.windowFlags() & ~QtCore.Qt.WindowContextHelpButtonHint)

        hlayout = QtWidgets.QHBoxLayout()

        vlayout = QtWidgets.QVBoxLayout()

        # Radio buttons
        #----------------------------------------------------------------
        self.button1 = QtWidgets.QRadioButton('Onzekerheden uit steunpunt overnemen')
        self.button2 = QtWidgets.QRadioButton('Onzekerheden uit havenmodel overnemen')
        self.button3 = QtWidgets.QRadioButton('Combinatie van bovenstaande gebruiken')

        vlayout.addWidget(self.button1)
        vlayout.addWidget(self.button2)
        vlayout.addWidget(self.button3)
        vlayout.addItem(QtWidgets.QSpacerItem(0, 0, QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Expanding))

        hlayout.addLayout(vlayout)

        vlayout = QtWidgets.QVBoxLayout()
        # Model uncertainties support location
        #----------------------------------------------------------------
        label = QtWidgets.QLabel()
        label.setText('Modelonzekerheden in steunpunt:')
        vlayout.addWidget(label)

        self.supportloc_unc_table = widgets.DataFrameWidget(self.supportloc_unc)
        self.supportloc_unc_table.fixed_fit_to_content(90)
        vlayout.addWidget(self.supportloc_unc_table)

        label = QtWidgets.QLabel()
        label.setText('Modelonzekerheden in havenmodel (zelf invullen):')
        vlayout.addWidget(label)

        self.harbor_unc_table = widgets.DataFrameWidget(self.harbor_unc, editing_enabled=True)
        self.harbor_unc_table.fixed_fit_to_content(90)
        vlayout.addWidget(self.harbor_unc_table)

        label = QtWidgets.QLabel()
        label.setText('Gecombineerde modelonzekerheid (berekenen):')
        vlayout.addWidget(label)

        calc_button = QtWidgets.QPushButton('Berekenen')
        calc_button.clicked.connect(self._calc_combined_uncertainty)
        vlayout.addWidget(calc_button)

        self.combined_unc_table = widgets.DataFrameWidget(self.combined_unc)
        self.combined_unc_table.fixed_fit_to_content(90)
        vlayout.addWidget(self.combined_unc_table)

        for table in [self.supportloc_unc_table, self.harbor_unc_table, self.combined_unc_table]:
            table.setShowGrid(True)
            table.setAlternatingRowColors(False)

        hlayout.addLayout(vlayout)

        vlayout = QtWidgets.QVBoxLayout()
        vlayout.addLayout(hlayout)
        line = QtWidgets.QFrame()
        line.setFrameShape(QtWidgets.QFrame.HLine)
        line.setFrameShadow(QtWidgets.QFrame.Sunken)
        vlayout.addWidget(line)


        # Buttons
        #----------------------------------------------------------------
        hbox = QtWidgets.QHBoxLayout()
        hbox.addItem(QtWidgets.QSpacerItem(0, 0, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Maximum))
        # Add ok/close
        self.closebutton = QtWidgets.QPushButton('Sluiten')
        self.closebutton.clicked.connect(self.close)
        hbox.addWidget(self.closebutton)
        # Add ok/close
        self.savebutton = QtWidgets.QPushButton('Opslaan')
        self.savebutton.clicked.connect(self._save)
        hbox.addWidget(self.savebutton)

        vlayout.addLayout(hbox)

        # Add layout to widget
        self.setLayout(vlayout)
        self.layout().setSpacing(10)

    def _calc_combined_uncertainty(self):
        """
        Method to calculate combined uncertainties.
        The actual calculation is done in the core ModelUncertainties class.
        """
        # Check if the harboruncertainties are filled:
        if pd.isnull(self.harbor_unc_table.model._data).any().any():
            raise ValueError('Niet alle modelonzekerheden voor het havenmodel zijn ingevuld.')

        # Calculate combined uncertainties
        self.combined_unc_table.model.layoutAboutToBeChanged.emit()
        self.modeluncertainties.calculate_combined_uncertainty()
        self.combined_unc_table.model.layoutChanged.emit()

    def _save(self):
        """
        Saved uncertainties to dataframe based on selected radio button
        """

        # Get option
        if self.button1.isChecked():
            option = 'Steunpunt'
            uncertainties = self.supportloc_unc_table
        elif self.button2.isChecked():
            # Check if the harboruncertainties are filled:
            if pd.isnull(self.harbor_unc_table.model._data).any().any():
                raise ValueError('Niet alle modelonzekerheden voor het havenmodel zijn ingevuld.')
            option = 'Havenmodel'
            uncertainties = self.harbor_unc_table
        elif self.button3.isChecked():
            if pd.isnull(self.combined_unc_table.model._data).any().any():
                raise ValueError('De gecombineerde modelonzekerheden zijn nog niet berekend.')
            option = 'Combinatie'
            uncertainties = self.combined_unc_table
        else:
            raise ValueError('Selecteer een optie voor de te gebruiken onzekerheid')

        self.parent.adjust_selection(uncertainties=uncertainties, option=option)

        # Toegevoegd Svasek 31/10/2018 - Sluit het onzekerheden input scherm als er op opslaan gedrukt wordt
        self.close()

class ResultScatterDialog(QtWidgets.QDialog):

    def __init__(self, modelunctab=None):

        super(ResultScatterDialog, self).__init__(modelunctab)

        self.modelunctab = modelunctab
        self.result_locations_df = self.modelunctab.mainmodel.schematisation.result_locations
        self.result_locations = self.result_locations_df['Naam'].tolist()
        self.hydraulic_loads = self.modelunctab.mainmodel.hydraulic_loads

        # Get results from parent
        calculation_settings = modelunctab.settings['calculation_method']
        # Simple method
        if calculation_settings['method'] == 'simple':
            self.results = modelunctab.simple_calculation.combinedresults.output
            self.result_parameters = self.modelunctab.simple_calculation.result_parameters

        # Advanced method
        elif calculation_settings['method'] == 'advanced':
            # With pharos
            if calculation_settings['include_pharos']:
                self.results = modelunctab.pharos.calculation_results
                self.result_parameters = modelunctab.swan.result_parameters

            # Toegevoegd Svasek 04/10/18 - Gebruik Hares resultaten
            # With HARES
            if calculation_settings['include_hares']:
                self.results = modelunctab.hares.calculation_results
                self.result_parameters = modelunctab.hares.result_parameters

            # Without pharos or hares
            else:
                self.results = modelunctab.swan.calculation_results
                self.result_parameters = modelunctab.swan.result_parameters
        else:
            raise ValueError('Calculation type (simple/advanced) unknown. Got "{}"'.format(calculation_settings['method']))

        self.scatter = None
        self.oneoneline = None

        # Determine initial location from selection
        self._get_initial_location()

        # Build UI
        self._init_ui()


    def _init_ui(self):
        """
        Set up UI design
        """
        self.setWindowTitle("HB Havens: resultaten")
        self.setWindowFlags(self.windowFlags() & ~QtCore.Qt.WindowContextHelpButtonHint)

        self.setLayout(QtWidgets.QVBoxLayout())

        # Create figure
        self.figure = Figure(figsize=(4,4))
        self.ax = self.figure.add_subplot()

        self.ax.grid()
        self.ax.spines['right'].set_visible(False)
        self.ax.spines['top'].set_visible(False)
        self.ax.tick_params(axis='y', color='0.75')
        self.ax.tick_params(axis='x', color='0.75')
        self.ax.set_aspect(1)

        # Add canvas
        self.canvas = FigureCanvasQTAgg(self.figure)

        # this is the Navigation widget
        # it takes the Canvas widget and a parent
        self.layout().addWidget(self.canvas)

        # Add location selection
        hbox = QtWidgets.QHBoxLayout()
        label = QtWidgets.QLabel('Locatie:')
        label.setFixedWidth(80)
        hbox.addWidget(label)
        self.location_combobox = QtWidgets.QComboBox()
        self.location_combobox.addItems(self.result_locations)
        self.location_combobox.setCurrentIndex(self.locid)
        self.location_combobox.currentIndexChanged.connect(self._set_location)
        hbox.addWidget(self.location_combobox)
        self.layout().addLayout(hbox)

        # Add parameter selection
        hbox = QtWidgets.QHBoxLayout()
        label = QtWidgets.QLabel('Parameter:')
        label.setFixedWidth(80)
        hbox.addWidget(label)
        self.parameter_combobox = QtWidgets.QComboBox()
        self.input_parameters = self.modelunctab.mainmodel.hydraulic_loads.result_columns[:]
        self.parameter_combobox.addItems(self.input_parameters)
        self.parameter_combobox.currentIndexChanged.connect(self._set_parameter)
        self.parameter_combobox.setCurrentIndex(0)
        self._set_parameter()
        self.figure.tight_layout()
        hbox.addWidget(self.parameter_combobox)
        self.layout().addLayout(hbox)

        # Line
        line = QtWidgets.QFrame()
        line.setFrameShape(QtWidgets.QFrame.HLine)
        line.setFrameShadow(QtWidgets.QFrame.Sunken)

        self.layout().addWidget(line)

        # Add ok/close
        self.closebutton = QtWidgets.QPushButton('Sluiten')
        self.closebutton.clicked.connect(self.close)
        self.layout().addWidget(self.closebutton, 0, QtCore.Qt.AlignRight)

        self.layout().setSizeConstraint(QtWidgets.QLayout.SetFixedSize)

    def _get_initial_location(self):
        """
        Get initial location from selection or list
        """
        for i in self.modelunctab.tableview.selectedIndexes():
            self.locid = i.row()
            break
        else:
            self.locid = 0

        self.locnaam = self.result_locations[self.locid]

    def _set_data(self):
        """
        Method to set data for result scatterplot.

        1. Determines the location and parameter to be visualized
        2. Get data from dataframe
        3. Set data to plot
        """

        # Remove old elements from plot
        if self.scatter is not None:
            self.scatter.remove()
        if self.oneoneline is not None:
            self.oneoneline.remove()

        # Get new data and plot
        self.slice = self.results.set_index('Location').loc[self.locnaam, [self.input_parameter, self.result_parameter]].values.T
        self.scatter = self.ax.scatter(*self.slice, s=5, alpha=0.7, color='C0')

        # Determine axes limits
        lowerlim, upperlim = self.slice.min(), self.slice.max()
        span = (upperlim - lowerlim)
        lowerlim = max(0, lowerlim - 0.05 * span)
        upperlim = upperlim + 0.05 * span

        # Plot a diagonal 1:1 line
        self.oneoneline, = self.ax.plot([lowerlim, upperlim], [lowerlim, upperlim], color='grey', dashes=(4, 3), lw=1.0)

        # Set the axes limits
        self.ax.set_xlim(lowerlim, upperlim)
        self.ax.set_ylim(lowerlim, upperlim)
        self.canvas.draw()


    def _set_location(self):
        """
        Changes the location for which the data is visualized.
        """
        # Get selected text
        self.locnaam = self.location_combobox.currentText()

        self._set_data()

    def _set_parameter(self):
        """
        Sets the parameter for which the data is visualized.
        """
        # Get parameter keys
        self.input_parameter = self.parameter_combobox.currentText()
        self.result_parameter = self.result_parameters[self.input_parameter]
        # Adjust axes labels
        self.ax.set_xlabel('{} steunpunt'.format(self.input_parameter))
        self.ax.set_ylabel('{} uitvoerlocatie'.format(self.input_parameter))
        # Set data
        self._set_data()


class DefineExportNameAndDatabase(QtWidgets.QDialog):

    def __init__(self, parent=None):

        super(DefineExportNameAndDatabase, self).__init__(parent)

        self.table = parent.model._data
        self.name = parent.selectedname
        self._initUI()

    def _initUI(self):
        """
        Set up UI design
        """

        vlayout = QtWidgets.QVBoxLayout()

        # Description
        #----------------------------------------------------------------
        hlayout = QtWidgets.QHBoxLayout()

        label = QtWidgets.QLabel()
        label.setText('Locatie:')
        label.setFixedWidth(100)
        hlayout.addWidget(label)

        label = QtWidgets.QLabel()
        label.setText(self.name)
        hlayout.addWidget(label)
        hlayout.setSpacing(10)

        vlayout.addLayout(hlayout)

        # Exportnaam
        #----------------------------------------------------------------
        self.exportname = ParameterInputLine(label='Exportnaam:', labelwidth=100)
        self.exportname.LineEdit.setMinimumWidth(200)
        vlayout.addLayout(self.exportname.layout)

        # Exportdatabase
        #----------------------------------------------------------------
        self.exportpath = ExtendedLineEdit(label='SQLite-database:', labelwidth=100, browsebutton=True)
        self.exportpath.BrowseButton.clicked.connect(self._get_path_database)
        vlayout.addLayout(self.exportpath.layout)

        # Line
        line = QtWidgets.QFrame()
        line.setFrameShape(QtWidgets.QFrame.HLine)
        line.setFrameShadow(QtWidgets.QFrame.Sunken)

        vlayout.addWidget(line)

        # Buttons
        #----------------------------------------------------------------
        hbox = QtWidgets.QHBoxLayout()
        hbox.addItem(QtWidgets.QSpacerItem(0, 0, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Maximum))
        # Add ok/close
        self.closebutton = QtWidgets.QPushButton('Sluiten')
        self.closebutton.clicked.connect(self.close)
        hbox.addWidget(self.closebutton)
        # Add ok/close
        self.savebutton = QtWidgets.QPushButton('Opslaan')
        self.savebutton.clicked.connect(self._save)
        hbox.addWidget(self.savebutton)

        vlayout.addLayout(hbox)

        # Add layout to widget
        self.setLayout(vlayout)

    def _get_path_database(self):
        """
        Method to get breakwater path
        """
        # Get path with dialog
        path = QtWidgets.QFileDialog.getOpenFileName(self.exportpath.LineEdit, 'Selecteer SQLite om resultaten toe te voegen', '', "SQLite (*.sqlite)")[0]
        # Save path to project structure
        self.exportpath.LineEdit.setText(path)


    def _save(self):
        """
        Save input
        """
        # Get selected names (index)
        idx = self.table.where(self.table['Naam'] == self.name).dropna(how='all').index.values[0]
        self.table.at[idx, 'Exportnaam'] = self.exportname.LineEdit.text()
        self.table.at[idx, 'SQLite database'] = self.exportpath.LineEdit.text()
        self.close()

class QuestionDialog(Qt.QDialog):

    def __init__(self, i_parent=None):
        super(HBHDialogs, self).__init__(parent)

    @staticmethod
    def about(self, title, text):
        Qt.QMessageBox.about(self, title, text)

    @staticmethod
    def question(self, title, text):
        """
        """
        reply = Qt.QMessageBox.question(self, title, text, Qt.QMessageBox.Yes | Qt.QMessageBox.No)
        if reply == Qt.QMessageBox.Yes:
            return True
        elif reply == Qt.QMessageBox.No:
            return False
        else:
            return False
