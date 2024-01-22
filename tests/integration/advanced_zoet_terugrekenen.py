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

class TestAdvancedZoetSwanRecalculate(unittest.TestCase):

    def setUp(self):
        logger.info('SETUP TestAdvancedZoetSwanRecalculate')
        self.mainmodel = MainModel()
        prjjson = 'temp/heliushaven.json'
        if os.path.exists(prjjson) and not self._testMethodName[0:5] == 'test1':
            self.mainmodel.project.open_from_file(prjjson)

    def tearDown(self):
        logger.info('TEARDOWN TestAdvancedZoetSwanRecalculate')
        if os.path.exists('temp'):
            self.mainmodel.project.save_as('temp/heliushaven.json', overwrite_all=True)
            self.mainmodel.save_tables()

    def test1_swan(self):
        # If the directory already exists, remove it.
        if os.path.exists('temp'):
            shutil.rmtree('temp')
        os.mkdir('temp')

        supportloc = "HV_1_20-4_dk_00087"
        bedlevel = -10.0
        trajecten = ['20-4']

        common.add_schematisation(self.mainmodel, paths=paths, supportlocationname=supportloc, trajecten=trajecten, bedlevel=bedlevel)

        # test2_recalculate_water_levels
        common.randomize_table(self.mainmodel.hydraulic_loads)
        self.mainmodel.hydraulic_loads.detect_waterlevel_breaks()
        # Test breaks
        self.assertListEqual(
            self.mainmodel.project.settings['hydraulic_loads']['waterlevels'],
            [0.566, 1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 7.289]
        )

        # test3_recalculate_wave_conditions
        # Recalculate and adapt
        self.mainmodel.hydraulic_loads.calculate_waveconditions()
        # Shuffle recalculated loads
        common.randomize_table(self.mainmodel.hydraulic_loads.recalculated_loads)
        common.randomize_table(self.mainmodel.schematisation.result_locations)
        self.mainmodel.hydraulic_loads.adapt_interpolated()
        # Shuffle adapted hydraulic loads
        common.randomize_table(self.mainmodel.hydraulic_loads)

        # Test recalculated loads
        table_path = f'{datadir}/swan/recalculated_loads_heliushaven.gz'
        common.compare_or_save(self, table_path, self.mainmodel.hydraulic_loads)

        # test4_calculation(self):
        # Stel in voor geavanceerde methode
        self.mainmodel.project.settings['calculation_method']['method'] = 'advanced'
        self.mainmodel.project.settings['calculation_method']['include_pharos'] = False
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

        # Model uncertainties
        common.set_model_uncertainties(self.mainmodel)
        
        # test5_interpolate_wave_conditions(self):
        # Shuffle recalculated loads
        common.randomize_table(self.mainmodel.schematisation.result_locations)
        common.randomize_table(self.mainmodel.hydraulic_loads)

        # Get results and column mapping
        column_mapping = self.mainmodel.swan.calculation_results.result_parameters
        results=self.mainmodel.swan.calculation_results.reset_index()

        # Shuffle results
        common.randomize_table(results)
        # Interpolate wave conditions
        interpolated = self.mainmodel.hydraulic_loads.interpolate_wave_conditions(results, column_mapping).round(3)
        # Compare results
        table_path = f'{datadir}/swan/interpolated_wave_conditions_heliushaven.gz'
        common.compare_or_save(self, table_path, interpolated)

        # test6_exporteren:
        # Exporteren naar hrd
        #========================================
        # In deze stap exporteren we de resultaten naar de HRD. We maken hiervoor eerst een kopie van de originele database.
        # Maak een kopie van de hrd, config en hlcd
        tmp_paths = common.create_tmp_db_copies(paths, ext='swan')
        common.randomize_table(self.mainmodel.schematisation.result_locations)
        common.export_to_database(self.mainmodel, tmp_paths, export_hlcd_config=False)

        # Verwijder
        shutil.rmtree('temp')

class TestAdvancedZoetSwanPharosRecalculate(unittest.TestCase):

    def setUp(self):
        logger.info('SETUP TestAdvancedZoetSwanPharosRecalculate')
        self.mainmodel = MainModel()
        prjjson = 'temp/Heliushaven.json'
        if os.path.exists(prjjson):
            self.mainmodel.project.open_from_file(prjjson)

    def tearDown(self):
        logger.info('TEARDOWN TestAdvancedZoetSwanPharosRecalculate')
        if os.path.exists('temp'):
            self.mainmodel.project.save_as('temp/Heliushaven.json', overwrite_all=True)
            self.mainmodel.save_tables()

    def test2_swan_pharos(self):
        # If the directory already exists, remove it.
        if os.path.exists('temp'):
            shutil.rmtree('temp')
        os.mkdir('temp')

        supportloc = "HV_1_20-4_dk_00087"
        bedlevel = -10.0
        trajecten = ['20-4']

        common.add_schematisation(self.mainmodel, paths=paths, supportlocationname=supportloc, trajecten=trajecten, bedlevel=bedlevel)

        # test2_recalculate_water_levels
        common.randomize_table(self.mainmodel.hydraulic_loads)
        self.mainmodel.hydraulic_loads.detect_waterlevel_breaks()
        # Test breaks
        self.assertListEqual(
            self.mainmodel.project.settings['hydraulic_loads']['waterlevels'],
            [0.566, 1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 7.289]
        )

        # test3_recalculate_wave_conditions
        # Recalculate and adapt
        self.mainmodel.hydraulic_loads.calculate_waveconditions()
        # Shuffle recalculated loads
        common.randomize_table(self.mainmodel.hydraulic_loads.recalculated_loads)
        common.randomize_table(self.mainmodel.schematisation.result_locations)
        self.mainmodel.hydraulic_loads.adapt_interpolated()
        # Shuffle adapted hydraulic loads
        common.randomize_table(self.mainmodel.hydraulic_loads)

        # Test recalculated loads
        table_path = f'{datadir}/swan/recalculated_loads_heliushaven.gz'
        common.compare_or_save(self, table_path, self.mainmodel.hydraulic_loads)

        # test4_calculation(self):
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
        
        # test5_interpolate_wave_conditions(self):
        # Shuffle recalculated loads
        common.randomize_table(self.mainmodel.schematisation.result_locations)
        common.randomize_table(self.mainmodel.hydraulic_loads)

        # Get results and column mapping
        column_mapping = self.mainmodel.pharos.result_parameters
        results=self.mainmodel.pharos.calculation_results.reset_index()

        # Shuffle results
        common.randomize_table(results)
        # Interpolate wave conditions
        interpolated = self.mainmodel.hydraulic_loads.interpolate_wave_conditions(results, column_mapping).round(3)
        # Compare results
        table_path = f'{datadir}/pharos/interpolated_wave_conditions_heliushaven.gz'
        common.compare_or_save(self, table_path, interpolated)

        # test6_exporteren:
        # Exporteren naar hrd
        #========================================
        # In deze stap exporteren we de resultaten naar de HRD. We maken hiervoor eerst een kopie van de originele database.
        # Maak een kopie van de hrd, config en hlcd
        tmp_paths = common.create_tmp_db_copies(paths, ext='pharos')
        common.randomize_table(self.mainmodel.schematisation.result_locations)
        common.export_to_database(self.mainmodel, tmp_paths, export_hlcd_config=False)

        # Verwijder
        shutil.rmtree('temp')

class TestAdvancedZoetSwanHaresRecalculate(unittest.TestCase):

    def setUp(self):
        logger.info('SETUP TestAdvancedZoetSwanHaresRecalculate')
        self.mainmodel = MainModel()
        prjjson = 'temp/Heliushaven.json'
        if os.path.exists(prjjson):
            self.mainmodel.project.open_from_file(prjjson)

    def tearDown(self):
        logger.info('TEARDOWN TestAdvancedZoetSwanHaresRecalculate')
        if os.path.exists('temp'):
            self.mainmodel.project.save_as('temp/Heliushaven.json', overwrite_all=True)
            self.mainmodel.save_tables()

    def test3_swan_hares(self):
        # If the directory already exists, remove it.
        if os.path.exists('temp'):
            shutil.rmtree('temp')
        os.mkdir('temp')

        supportloc = "HV_1_20-4_dk_00087"
        bedlevel = -10.0
        trajecten = ['20-4']

        common.add_schematisation(self.mainmodel, paths=paths, supportlocationname=supportloc, trajecten=trajecten, bedlevel=bedlevel)

        # test2_recalculate_water_levels
        common.randomize_table(self.mainmodel.hydraulic_loads)
        self.mainmodel.hydraulic_loads.detect_waterlevel_breaks()
        # Test breaks
        self.assertListEqual(
            self.mainmodel.project.settings['hydraulic_loads']['waterlevels'],
            [0.566, 1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 7.289]
        )

        # test3_recalculate_wave_conditions
        # Recalculate and adapt
        self.mainmodel.hydraulic_loads.calculate_waveconditions()
        # Shuffle recalculated loads
        common.randomize_table(self.mainmodel.hydraulic_loads.recalculated_loads)
        common.randomize_table(self.mainmodel.schematisation.result_locations)
        self.mainmodel.hydraulic_loads.adapt_interpolated()
        # Shuffle adapted hydraulic loads
        common.randomize_table(self.mainmodel.hydraulic_loads)

        # Test recalculated loads
        table_path = f'{datadir}/swan/recalculated_loads_heliushaven.gz'
        common.compare_or_save(self, table_path, self.mainmodel.hydraulic_loads)

        # test4_calculation(self):
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
            self.mainmodel.hares.calculation_results[col+' hares'] = self.mainmodel.hares.calculation_results[col] * 0.05
        self.mainmodel.hares.combine_with_swan()

        # Model uncertainties
        common.set_model_uncertainties(self.mainmodel)
        
        # test5_interpolate_wave_conditions(self):
        # Shuffle recalculated loads
        common.randomize_table(self.mainmodel.schematisation.result_locations)
        common.randomize_table(self.mainmodel.hydraulic_loads)

        # Get results and column mapping
        column_mapping = self.mainmodel.hares.result_parameters
        results=self.mainmodel.hares.calculation_results.reset_index()

        # Shuffle results
        common.randomize_table(results)
        # Interpolate wave conditions
        interpolated = self.mainmodel.hydraulic_loads.interpolate_wave_conditions(results, column_mapping).round(3)
        # Compare results
        table_path = f'{datadir}/hares/interpolated_wave_conditions_heliushaven.gz'
        common.compare_or_save(self, table_path, interpolated)

        # test6_exporteren:
        # Exporteren naar hrd
        #========================================
        # In deze stap exporteren we de resultaten naar de HRD. We maken hiervoor eerst een kopie van de originele database.
        # Maak een kopie van de hrd, config en hlcd
        tmp_paths = common.create_tmp_db_copies(paths, ext='hares')
        common.randomize_table(self.mainmodel.schematisation.result_locations)
        common.export_to_database(self.mainmodel, tmp_paths, export_hlcd_config=False)

        # Verwijder
        shutil.rmtree('temp')

if __name__ == "__main__":
    unittest.main(verbosity=2)
