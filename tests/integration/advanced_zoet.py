import os
import shutil
import unittest

from hbhavens.core.models import MainModel

import logging
from hbhavens.core.logger import initialize_logger

import common

initialize_logger()
logger = logging.getLogger()

datadir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'heliushaven')

paths = {
    'harborarea': os.path.join(datadir, 'shapes', 'haventerrein.shp'),
    'breakwater': os.path.join(datadir, 'shapes', 'havendammen.shp'),
    'result_locations_save': os.path.join(datadir, 'shapes', 'uitvoerlocaties.shp'),
    'hrd': os.path.join(datadir, 'databases', 'steunpunt_heliushaven.sqlite')
}

class TestAdvancedZoetSwan(unittest.TestCase):

    def setUp(self):
        logger.info('SETUP TestAdvancedZoetSwan')
        self.mainmodel = MainModel()
        prjjson = 'temp/heliushaven.json'
        if os.path.exists(prjjson) and not self._testMethodName[0:5] == 'test1':
            self.mainmodel.project.open_from_file(prjjson)

    def tearDown(self):
        logger.info('TEARDOWN TestAdvancedZoetSwan')
        if os.path.exists('temp'):
            self.mainmodel.project.save_as('temp/heliushaven.json', overwrite_all=True)
            self.mainmodel.save_tables()

    def test1_calculation_swan(self):
        """
        Test preparing Heliushaven schematisation
        """
        logger.info('TEST1_SCHEMATISATION_HELIUSHAVEN')
        # If the directory already exists, remove it.
        if os.path.exists('temp'):
            shutil.rmtree('temp')
        os.mkdir('temp')

        supportloc = "HV_1_20-4_dk_00087"
        bedlevel = -10.0
        trajecten = ['20-4']

        common.add_schematisation(self.mainmodel, paths=paths, supportlocationname=supportloc, trajecten=trajecten, bedlevel=bedlevel)
        # Test flood defence

        # Stel in voor geavanceerde methode
        self.mainmodel.project.settings['calculation_method']['method'] = 'advanced'
        self.mainmodel.project.settings['calculation_method']['include_pharos'] = False
        self.mainmodel.project.settings['calculation_method']['include_hares'] = False
        self.mainmodel.project.settings['swan']['swanfolder'] = 'swandummy'
        self.mainmodel.project.settings['swan']['mastertemplate'] = f'dummy.swn'
        self.mainmodel.project.settings['swan']['depthfile'] = f'dummy.dep'
        self.mainmodel.project.settings['swan']['use_incoming_wave_factors'] = True

        # Copy support loc to iteration results
        cols = {
            'Hs': ['Hs steunpunt 1', 'Hs rand 2', 'Hs steunpunt 2', 'Hs rand 3', 'Hs steunpunt 3'],
            'Tp': ['Tp,s steunpunt 1', 'Tp,s steunpunt 2', 'Tp,s steunpunt 3', 'Tp,s rand 2', 'Tp,s rand 3']
        }

        # Set the iteration results.
        for var, columns in cols.items():
            for col in columns:
                self.mainmodel.swan.iteration_results.loc[:, col] = self.mainmodel.swan.iteration_results.loc[:, var]
        self.mainmodel.swan.iteration_finished()

        # Set the calculation results
        for step, factor in zip(['D', 'TR', 'W'], [0.1, 0.3, 0.2]):
            inp_cols = ['Hs', 'Tp', 'Tm-1,0']
            out_cols = ['Hm0_' + step, 'Tp_' + step, 'Tmm10_' + step]
            self.mainmodel.swan.calculation_results[out_cols] = self.mainmodel.swan.calculation_results[inp_cols] * factor
            self.mainmodel.swan.calculation_results['Theta0_' + step] = self.mainmodel.swan.calculation_results['Wave direction']
        self.mainmodel.swan.calculation_results.calculate_total_energy()

        # Model uncertainties
        common.set_model_uncertainties(self.mainmodel)

        # Exporteren naar hrd
        #========================================
        # In deze stap exporteren we de resultaten naar de HRD. We maken hiervoor eerst een kopie van de originele database.
        # Maak een kopie van de hrd, config en hlcd
        tmp_paths = common.create_tmp_db_copies(paths, ext='swan')
        common.randomize_table(self.mainmodel.schematisation.result_locations)
        common.export_to_database(self.mainmodel, tmp_paths, export_hlcd_config=False)

        # Verwijder
        shutil.rmtree('temp')

class TestAdvancedZoetSwanPharos(unittest.TestCase):

    def setUp(self):
        logger.info('SETUP TestAdvancedZoetSwanPharos')
        self.mainmodel = MainModel()
        prjjson = 'temp/Heliushaven.json'
        if os.path.exists(prjjson):
            self.mainmodel.project.open_from_file(prjjson)

    def tearDown(self):
        logger.info('TEARDOWN TestAdvancedZoetSwanPharos')
        if os.path.exists('temp'):
            self.mainmodel.project.save_as('temp/Heliushaven.json', overwrite_all=True)
            self.mainmodel.save_tables()

    def test2_calculation_swan_pharos(self):
        """
        Test preparing Heliushaven schematisation
        """
        logger.info('TEST1_SCHEMATISATION_HELIUSHAVEN')
        # If the directory already exists, remove it.
        if os.path.exists('temp'):
            shutil.rmtree('temp')
        os.mkdir('temp')

        supportloc = "HV_1_20-4_dk_00087"
        bedlevel = -10.0
        trajecten = ['20-4']

        common.add_schematisation(self.mainmodel, paths=paths, supportlocationname=supportloc, trajecten=trajecten, bedlevel=bedlevel)
        # Test flood defence

        # Stel in voor geavanceerde methode
        self.mainmodel.project.settings['calculation_method']['method'] = 'advanced'
        self.mainmodel.project.settings['calculation_method']['include_pharos'] = True
        self.mainmodel.project.settings['calculation_method']['include_hares'] = False
        
        # Copy support loc to iteration results
        cols = {
            'Hs': ['Hs steunpunt 1', 'Hs rand 2', 'Hs steunpunt 2', 'Hs rand 3', 'Hs steunpunt 3'],
            'Tp': ['Tp,s steunpunt 1', 'Tp,s steunpunt 2', 'Tp,s steunpunt 3', 'Tp,s rand 2', 'Tp,s rand 3']
        }

        # Set the iteration results.
        for var, columns in cols.items():
            for col in columns:
                self.mainmodel.swan.iteration_results.loc[:, col] = self.mainmodel.swan.iteration_results.loc[:, var]
        self.mainmodel.swan.iteration_finished()

        # Set the calculation results
        for step, factor in zip(['D', 'TR', 'W'], [0.1, 0.3, 0.2]):
            inp_cols = ['Hs', 'Tp', 'Tm-1,0']
            out_cols = ['Hm0_' + step, 'Tp_' + step, 'Tmm10_' + step]
            self.mainmodel.swan.calculation_results[out_cols] = self.mainmodel.swan.calculation_results[inp_cols] * factor
            self.mainmodel.swan.calculation_results['Theta0_' + step] = self.mainmodel.swan.calculation_results['Wave direction']
        self.mainmodel.swan.calculation_results.calculate_total_energy()

        # Set the pharos results
        cols = ['Hs', 'Tp', 'Tm-1,0', 'Wave direction']
        for col in cols:
            self.mainmodel.pharos.calculation_results[col+' pharos'] = self.mainmodel.pharos.calculation_results[col] * 0.05
        self.mainmodel.pharos.combine_with_swan()

        # Model uncertainties
        common.set_model_uncertainties(self.mainmodel)

        # Exporteren naar hrd
        #========================================
        # In deze stap exporteren we de resultaten naar de HRD. We maken hiervoor eerst een kopie van de originele database.
        # Maak een kopie van de hrd, config en hlcd
        tmp_paths = common.create_tmp_db_copies(paths, ext='swan_pharos')
        common.randomize_table(self.mainmodel.schematisation.result_locations)
        common.export_to_database(self.mainmodel, tmp_paths, export_hlcd_config=False)

        # Verwijder
        shutil.rmtree('temp')

class TestAdvancedZoetSwanHares(unittest.TestCase):

    def setUp(self):
        logger.info('SETUP TestAdvancedZoetSwanHares')
        self.mainmodel = MainModel()
        prjjson = 'temp/Heliushaven.json'
        if os.path.exists(prjjson):
            self.mainmodel.project.open_from_file(prjjson)

    def tearDown(self):
        logger.info('TEARDOWN TestAdvancedZoetSwanHares')
        if os.path.exists('temp'):
            self.mainmodel.project.save_as('temp/Heliushaven.json', overwrite_all=True)
            self.mainmodel.save_tables()

    def test3_calculation_swan_hares(self):
        logger.info('TEST1_SCHEMATISATION_HELIUSHAVEN')
        # If the directory already exists, remove it.
        if os.path.exists('temp'):
            shutil.rmtree('temp')
        os.mkdir('temp')

        supportloc = "HV_1_20-4_dk_00087"
        bedlevel = -10.0
        trajecten = ['20-4']

        common.add_schematisation(self.mainmodel, paths=paths, supportlocationname=supportloc, trajecten=trajecten, bedlevel=bedlevel)
        # Test flood defence

        # Stel in voor geavanceerde methode
        self.mainmodel.project.settings['calculation_method']['method'] = 'advanced'
        self.mainmodel.project.settings['calculation_method']['include_pharos'] = False
        self.mainmodel.project.settings['calculation_method']['include_hares'] = True
        
        # Copy support loc to iteration results
        cols = {
            'Hs': ['Hs steunpunt 1', 'Hs rand 2', 'Hs steunpunt 2', 'Hs rand 3', 'Hs steunpunt 3'],
            'Tp': ['Tp,s steunpunt 1', 'Tp,s steunpunt 2', 'Tp,s steunpunt 3', 'Tp,s rand 2', 'Tp,s rand 3']
        }

        # Set the iteration results.
        for var, columns in cols.items():
            for col in columns:
                self.mainmodel.swan.iteration_results.loc[:, col] = self.mainmodel.swan.iteration_results.loc[:, var]
        self.mainmodel.swan.iteration_finished()

        # Set the calculation results
        for step, factor in zip(['D', 'TR', 'W'], [0.1, 0.3, 0.2]):
            inp_cols = ['Hs', 'Tp', 'Tm-1,0']
            out_cols = ['Hm0_' + step, 'Tp_' + step, 'Tmm10_' + step]
            self.mainmodel.swan.calculation_results[out_cols] = self.mainmodel.swan.calculation_results[inp_cols] * factor
            self.mainmodel.swan.calculation_results['Theta0_' + step] = self.mainmodel.swan.calculation_results['Wave direction']
        self.mainmodel.swan.calculation_results.calculate_total_energy()

        # Set the hares results
        cols = ['Hs', 'Tp', 'Tm-1,0', 'Wave direction']
        for col in cols:
            self.mainmodel.hares.calculation_results[col+' hares'] = self.mainmodel.hares.calculation_results[col]
        self.mainmodel.hares.combine_with_swan()

        # Model uncertainties
        common.set_model_uncertainties(self.mainmodel)

        # Exporteren naar hrd
        #========================================
        # In deze stap exporteren we de resultaten naar de HRD. We maken hiervoor eerst een kopie van de originele database.
        # Maak een kopie van de hrd, config en hlcd
        tmp_paths = common.create_tmp_db_copies(paths, ext='swan_hares')
        common.randomize_table(self.mainmodel.schematisation.result_locations)
        common.export_to_database(self.mainmodel, tmp_paths, export_hlcd_config=False)

        # Verwijder
        shutil.rmtree('temp')

if __name__ == "__main__":
    unittest.main(verbosity=2)
