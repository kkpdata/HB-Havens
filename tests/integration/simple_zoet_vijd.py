import os
import shutil
import unittest

import sqlite3
import pandas as pd

from hbhavens.core.models import MainModel
from hbhavens.io.database import HRDio

import logging
from hbhavens.core.logger import initialize_logger

import common

initialize_logger()
logger = logging.getLogger()

datadir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'riette')

paths = {
    'harborarea': os.path.join(datadir, 'shapes', 'haventerrein.shp'),
    'breakwater': os.path.join(datadir, 'shapes', 'havendammen.shp'),
    'hrd': os.path.join(datadir, 'databases', 'WBI2017_IJsseldelta_11-2_v01_steunpunt.sqlite'),
    'config': os.path.join(datadir, 'databases', 'WBI2017_IJsseldelta_11-2_v01_steunpunt.config.sqlite'),
    'result_locations_save': os.path.join(datadir, 'shapes', 'uitvoerlocaties.shp'),
    'hlcd': os.path.join(datadir, '..', 'hlcd.sqlite'),
}

class TestSimpleZoetUrk(unittest.TestCase):

    def setUp(self):
        logger.info('SETUP TestSimpleZoetVIJD')
        self.mainmodel = MainModel()
        prjjson = 'temp/riette.json'
        if os.path.exists(prjjson) and not self._testMethodName[0:5] == 'test1':
            self.mainmodel.project.open_from_file(prjjson)

    def tearDown(self):
        logger.info('TEARDOWN TestSimpleZoetVIJD')
        if os.path.exists('temp'):
            self.mainmodel.project.save_as('temp/riette.json', overwrite_all=True)
            self.mainmodel.save_tables()

    def test1_schematisation_Riette(self):
        """
        Test preparing Riette schematisation
        """
        logger.info('TEST1_SCHEMATISATION_RIETTE')
        if not os.path.exists('temp'):
            os.mkdir('temp')

        supportloc = "IJ_1_11-2_dk_00210"
        bedlevel = -5.0
        trajecten = ["11-2"]
        common.add_schematisation(self.mainmodel, paths=paths, supportlocationname=supportloc, trajecten=trajecten, bedlevel=bedlevel)

        # Test flood defence
        self.assertListEqual(self.mainmodel.schematisation.flooddefence['traject-id'].tolist(), trajecten)
        # Test harbor area
        self.assertAlmostEqual(self.mainmodel.schematisation.inner.area, 15239.245259023473)
        # Test support location name
        self.assertEqual(self.mainmodel.schematisation.support_location['Name'], supportloc)

        # Geef eerst in de projectinstellingen aan dat de eenvoudige methode gebruikt moet worden
        self.mainmodel.project.settings['calculation_method']['method'] = 'simple'

    def test2_recalculate_wave_conditions(self):
        """Find water level breaks in database
        """
        logger.info('TEST2_RECALCULATE_WAVE_CONDITIONS')

        self.mainmodel.project.settings['hydraulic_loads']["recalculate_waterlevels"] = True
        self.mainmodel.project.settings['hydraulic_loads']["waterlevels"].extend([-0.5, 1.8, 1.95, 3.0, 3.5, 5.0, 6.5])
        self.mainmodel.project.settings['hydraulic_loads']['recalculate_method'] = 'interpolation'

        # Recalculate and adapt
        self.mainmodel.hydraulic_loads.calculate_waveconditions()
        # Shuffle recalculated loads
        common.randomize_table(self.mainmodel.hydraulic_loads.recalculated_loads)
        self.mainmodel.hydraulic_loads.adapt_interpolated()
        # Shuffle adapted hydraulic loads
        common.randomize_table(self.mainmodel.hydraulic_loads)
        
        # Test recalculated loads
        table_path = f'{datadir}/simple/recalculates_loads_riette.gz'
        common.compare_or_save(self, table_path, self.mainmodel.hydraulic_loads)
        
    def test3_calculation(self):

        logger.info('TEST3_SIMPLE_CALCULATION')
        # Shuffle hydraulic loads
        common.randomize_table(self.mainmodel.hydraulic_loads)

        # Run all
        processes = ['Transmissie', 'Lokale golfgroei', 'Golfbreking']
        names = ['transmission', 'wavegrowth', 'wavebreaking']
        
        self.mainmodel.simple_calculation.run_all(processes=processes)

        # Test the results
        for name in names:
            table_path = f'{datadir}/simple/{name}_results_riette.gz'
            table = getattr(self.mainmodel.simple_calculation, name).output.round(4)
            common.compare_or_save(self, table_path, table)
            
        # Model uncertainties
        common.set_model_uncertainties(self.mainmodel)
   
    def test4_interpolate_wave_conditions(self):

        logger.info('TEST4_INTERPOLATE_WAVE_CONDITIONS')
        # Shuffle recalculated loads
        common.randomize_table(self.mainmodel.hydraulic_loads)

        # Get results and column mapping
        column_mapping = self.mainmodel.simple_calculation.result_parameters
        results=self.mainmodel.simple_calculation.combinedresults.output.reset_index()
        
        # Shuffle results
        common.randomize_table(results)
        # Interpolate wave conditions
        interpolated = self.mainmodel.hydraulic_loads.interpolate_wave_conditions(results, column_mapping).round(3)
        # Compare results
        table_path = f'{datadir}/simple/interpolated_wave_conditions_riette.gz'
        common.compare_or_save(self, table_path, interpolated)

    def test5_exporteren(self):
        logger.info('TEST5_EXPORTEREN')
        
        # Exporteren naar hrd
        #========================================
        # In deze stap exporteren we de resultaten naar de HRD. We maken hiervoor eerst een kopie van de originele database.
        # Maak een kopie van de hrd, config en hlcd
        tmp_paths = common.create_tmp_db_copies(paths, ext='simple')
        common.export_to_database(self.mainmodel, tmp_paths, export_hlcd_config=False)

        # Modify input variables for checking
        load_vars = self.mainmodel.hydraulic_loads.input_columns
        load_vars.remove('Water level')
        load_vars.extend(['Discharge Olst', 'Water level IJssel lake'])

        # Count the number of results, grouped per input variable
        counts, supportlocid = common.get_counts(self.mainmodel, tmp_paths)
        # Test the number of entries per input column
        for col, count in counts.items():
            # Check where the counts do not match the count of the support location
            faulty = count.index[~(count == count.loc[supportlocid]).all(axis=1).values].tolist()
            # Check if the list is equal
            self.assertListEqual(faulty, [], msg=f'Wrong number of "{col} for {faulty}: {count.loc[faulty]} instead of {count.loc[supportlocid]}.')

        # Check if the names in the database match the Locations
        conn = sqlite3.connect(tmp_paths['hrd'])
        hrdlocations = pd.read_sql('select Name, HRDLocationId from HRDLocations', con=conn)
        conn.close()
        hrd_names = set(hrdlocations['Name'].tolist())
        hbh_names = self.mainmodel.export.export_dataframe['Exportnaam'].tolist()

        # Check if all hbh names are in hrd
        self.assertTrue(hrd_names.issuperset(set(hbh_names)))
        
        # Controleer de waarden uit database, voor de eerste, middelste en laatste locatie
        names = [hbh_names[0], hbh_names[len(hbh_names) // 2], hbh_names[-1]]
        hrdlocationids = [hrdlocations.set_index('Name').at[name, 'HRDLocationId'] for name in names]
        
        hrd = HRDio(tmp_paths['hrd'])
        self.assertEqual(hrd.dbformat, 'WBI2017')
        load_vars.append('ClosingSituationId')

        for locid, name in zip(hrdlocationids, names):
            # Get locationdata from hrd
            hrd_table = hrd.read_HydroDynamicData(hrdlocationid=locid)

            # Get table with loaddata from swan results
            # Determine hbh-location for hrdlocationids
            hbhname = self.mainmodel.export.export_dataframe.set_index('Exportnaam').at[name, 'Naam']
            
            # Get results
            simple_table = self.mainmodel.export.resulttable.set_index('HydraulicLoadId')
            simple_table = simple_table.loc[simple_table['Location'].eq(hbhname)].drop(['Location', 'HRDLocationId'], axis=1)
            
            # Remove white space for itertupels
            simple_table = simple_table.sort_index(axis=1).sort_values(by=load_vars).reset_index(drop=True)
            # Sort the tables and compare
            hrd_table = hrd_table.sort_index(axis=1).sort_values(by=load_vars).reset_index(drop=True)

            # Compare
            common.compare_dataframes(self, simple_table.round(4), hrd_table.round(4))

        # Verwijder
        shutil.rmtree('temp')
    
if __name__ == "__main__":
    unittest.main(verbosity=2)
