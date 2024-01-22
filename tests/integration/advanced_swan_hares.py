import os
import shutil
import sqlite3
import unittest

import pandas as pd

import common
from hbhavens.core.logger import initialize_logger
from hbhavens.core.models import MainModel
from hbhavens.io.database import HRDio

import logging

initialize_logger()
logger = logging.getLogger()

datadir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'ijmuiden')

paths = {
    'harborarea': os.path.join(datadir, 'shapes', 'haventerrein.shp'),
    'breakwater': os.path.join(datadir, 'shapes', 'havendammen.shp'),
    'hrd': os.path.join(datadir, 'databases', 'WBI2017_IJmuiden_input_selectie.sqlite'),
    'result_locations': os.path.join(datadir, 'shapes', 'uitvoerlocaties.shp'),
}

class TestSwanHares(unittest.TestCase):

    def setUp(self):
        logger.info('SETUP TestSwanHares')
        self.mainmodel = MainModel()
        prjjson = 'temp/IJmuiden.json'
        if os.path.exists(prjjson) and not self._testMethodName[0:5] == 'test1':
            self.mainmodel.project.open_from_file(prjjson)

    def tearDown(self):
        logger.info('TEARDOWN TestSwanHares')
        if os.path.exists('temp'):
            self.mainmodel.project.save_as('temp/IJmuiden.json', overwrite_all=True)
            self.mainmodel.save_tables()

    def test1_schematisation_IJmuiden(self):
        """
        Test preparing IJmuiden schematisation
        """
        logger.info('TEST1_SCHEMATISATION_IJMUIDEN')
        if not os.path.exists('temp'):
            os.mkdir('temp')

        supportloc = "Goede steunpunt IJmuiden"
        bedlevel = -20.0
        trajecten = ['44-3', '14-10', '13-1']
        common.add_schematisation(self.mainmodel, paths=paths, supportlocationname=supportloc, trajecten=trajecten, bedlevel=bedlevel)

        # Test flood defence
        self.assertListEqual(self.mainmodel.schematisation.flooddefence['traject-id'].tolist(), trajecten)
        # Test harbor area
        self.assertAlmostEqual(self.mainmodel.schematisation.inner.area, 8313731.074086675)
        # Test support location name
        self.assertEqual(self.mainmodel.schematisation.support_location['Name'], supportloc)

        # Geef eerst in de projectinstellingen aan dat de eenvoudige methode gebruikt moet worden
        self.mainmodel.project.settings['calculation_method']['method'] = 'simple'


    def test2_swan(self):

        logger.info('TEST2_SWAN')

        # Berekening geavanceerde methode
        #========================================
        swan_dst = 'temp/swan'
        if not os.path.exists(swan_dst):
            os.makedirs(swan_dst)
        
        # Stel in voor geavanceerde methode
        self.mainmodel.project.settings['calculation_method']['method'] = 'advanced'
        self.mainmodel.project.settings['calculation_method']['include_pharos'] = False
        self.mainmodel.project.settings['calculation_method']['include_hares'] = True
        self.mainmodel.project.settings['swan']['swanfolder'] = swan_dst
        self.mainmodel.project.settings['swan']['mastertemplate'] = f'{datadir}/swan/A2_dummy.swn'
        self.mainmodel.project.settings['swan']['depthfile'] = f'{datadir}/swan/A2.dep'
        self.mainmodel.project.settings['swan']['use_incoming_wave_factors'] = False

        # SWAN iterations and input generations is not tested
        self.mainmodel.swan.iteration_finished()
        # Add values to prevent skipping the output reading
        self.mainmodel.swan.iteration_results[['Hs rand 3', 'Tp,s rand 3']] = 1.0
        # Randomly sort the table before importing
        common.randomize_table(self.mainmodel.swan.iteration_results)
            
        # Genereer invoerbestanden
        for step in ['D', 'TR', 'W']:
            self.mainmodel.swan.generate(step=step)
            # Copy testresults
            common.unzip_swan_files(datadir, swan_dst, step)

            # Randomly sort the table before importing
            common.randomize_table(self.mainmodel.swan.calculation_results)
            
            # Read the results
            self.mainmodel.swan.calculation_results.read_results(step)
            
            # If there are no test results yet, save
            table_path = f'{datadir}/swan/{step}_results_table.gz'
            common.compare_or_save(self, table_path, self.mainmodel.swan.calculation_results.sort_values(by=['Location', 'HydraulicLoadId']), cols=1)

    def test3_hares(self):

        # Make output dir if it does not exist
        haresdir = f'temp/IJmuiden/hares'
        
        # Pharos settings
        self.mainmodel.project.settings['hares']['hares folder'] = haresdir

        # Copy hares results
        common.unzip_hares_files(datadir, haresdir)

        # Read calculation results
        common.randomize_table(self.mainmodel.hares.calculation_results)
        self.mainmodel.hares.read_calculation_results()

        # If there are no test results yet, save
        table_path = f'{datadir}/hares/calculation_results_table.gz'
        common.compare_or_save(self, table_path, self.mainmodel.hares.calculation_results.fillna(0.0).sort_values(by=['Location', 'HydraulicLoadId']), cols=1)
        
        # Definieer modelonzekerheden
        #========================================
        common.set_model_uncertainties(self.mainmodel)

    def test4_export(self):

        # Exporteren naar hrd
        #========================================
        # In deze stap exporteren we de resultaten naar de HRD. We maken hiervoor eerst een kopie van de originele database.
        # Maak een kopie van de hrd, config en hlcd
        tmp_paths = common.create_tmp_db_copies(paths, ext='swan_hares')
        common.export_to_database(self.mainmodel, tmp_paths, export_hlcd_config=False)

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
        hrd_names = set(hrdlocations['Name'].tolist())
        hbh_names = self.mainmodel.export.export_dataframe['Exportnaam'].tolist()

        # Check if all hbh names are in hrd
        self.assertTrue(hrd_names.issuperset(set(hbh_names)))
        
        # Controleer de waarden uit database, voor de eerste, middelste en laatste locatie
        names = [hbh_names[0], hbh_names[len(hbh_names) // 2], hbh_names[-1]]
        hrdlocationids = [hrdlocations.set_index('Name').at[name, 'HRDLocationId'] for name in names]

        for locid, name in zip(hrdlocationids, names):

            hrd = HRDio(tmp_paths['hrd'])
            hrd_table = hrd.read_HydroDynamicData(hrdlocationid=locid)
            self.assertEqual(hrd.dbformat, 'WBI2017')

            # Get table with loaddata from swan results
            # Determine hbh-location for hrdlocationids
            hbhname = self.mainmodel.export.export_dataframe.set_index('Exportnaam').at[name, 'Naam']

            # Get swan results
            hares_table = self.mainmodel.hares.calculation_results[['Location', 'HydraulicLoadId', 'Tm-1,0 totaal', 'Hs totaal', 'Wave direction totaal']].set_index('HydraulicLoadId')
            hares_table = hares_table.loc[hares_table['Location'].eq(hbhname)].drop('Location', axis=1)

            # Merge with hydraulic loads
            load_vars = ['ClosingSituationId', 'Wind direction', 'Wind speed', 'Water level']
            hares_table = hares_table.join(self.mainmodel.hydraulic_loads[load_vars]).sort_index(axis=1).sort_values(by=load_vars).reset_index(drop=True)
            hares_table.columns = [col.replace(' totaal', '') for col in hares_table.columns]
            
            # Sort the tables and compare
            hrd_table = hrd_table.sort_index(axis=1).sort_values(by=load_vars).reset_index(drop=True)

            # Compare
            common.compare_dataframes(self, hares_table.round(3), hrd_table.round(3))

        conn.close()

        # Verwijder
        shutil.rmtree('temp')        


if __name__ == "__main__":
    unittest.main(verbosity=2)
