import logging
import os
import sqlite3
import unittest

import numpy as np
import pandas as pd

import hbhavens
from hbhavens.core.logger import initialize_logger
from hbhavens.core.models import MainModel
from hbhavens.io.database import HRDio

import common


datadir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'colijnsplaat')
trunkdir = os.path.abspath(os.path.join(os.path.dirname(hbhavens.__file__),'../..'))
tempdir = os.path.join(trunkdir, 'temp')

paths = {
    'temp': os.path.join(tempdir, 'colijnsplaat'),
    'harborarea': os.path.join(datadir, 'shapes', 'haventerrein.shp'),
    'breakwater': os.path.join(datadir, 'shapes', 'havendammen.shp'),
    'hrd': os.path.join(datadir, 'databases', 'WBI2023_Oosterschelde_28-1_v06_selectie.sqlite'),
    'result_locations': os.path.join(datadir, 'shapes', 'uitvoerlocaties.shp'),
}

initialize_logger(filename=os.path.join(tempdir,'hbhavens.log'))
logger = logging.getLogger()
class TestSimpleOosterschelde(unittest.TestCase):
    def setUp(self):
        logger.info('SETUP TestSimpleOosterschelde')
        self.mainmodel = MainModel()
        prjjson = os.path.join(paths['temp'],'colijnsplaat.json')
        if os.path.exists(prjjson) and not self._testMethodName[0:5] == 'test1':
            self.mainmodel.project.open_from_file(prjjson)

    def tearDown(self):
        logger.info('TEARDOWN TestSimpleOosterschelde')
        if os.path.exists(paths['temp']):
            self.mainmodel.project.save_as(os.path.join(paths['temp'],'colijnsplaat.json'), overwrite_all=True)
            self.mainmodel.save_tables()

    def test1_schematisation_Colijnsplaat(self):
        '''
        Test preparing Colijnsplaat schematisation
        '''
        logger.info('TEST1_SCHEMATISATION_COLIJNSPLAAT')
        if not os.path.exists(paths['temp']):
            os.mkdir(paths['temp'])

        supportloc = 'OS_3_hy04-00000'
        bedlevel = -10.0
        trajecten = ['28-1']
        common.add_schematisation(
            self.mainmodel, paths=paths, supportlocationname=supportloc, trajecten=trajecten, bedlevel=bedlevel
        )

        # Test flood defence
        self.assertListEqual(self.mainmodel.schematisation.flooddefence['traject-id'].tolist(), trajecten)
        # Test harbor area
        self.assertAlmostEqual(self.mainmodel.schematisation.inner.area, 126398.86321246061)
        # Test support location name
        self.assertEqual(self.mainmodel.schematisation.support_location['Name'], supportloc)

        # Geef eerst in de projectinstellingen aan dat de eenvoudige methode gebruikt moet worden
        self.mainmodel.project.settings['calculation_method']['method'] = 'simple'
        self.mainmodel.schematisation.plot_schematisation(save=True, path=paths['temp'], close=True)

    def test2_calculation(self):
        logger.info('TEST2_CALCULATION')

        # Shuffle hydraulic loads
        common.randomize_table(self.mainmodel.hydraulic_loads)

        # # # Test the results
        # def test_process(name, cols=2):
        #     table_path = f'{datadir}/simple/{name}_results_colijnsplaat.gz'
        #     table = getattr(self.mainmodel.simple_calculation, name).output.round(5)
        #     # Check if the results are equal to a test table
        #     # test_table = pd.read_csv(table_path, sep=';', decimal='.', index_col=list(range(cols)))
        #     # Check the results
        #     common.compare_or_save(self, table_path, table)

        # self.mainmodel.simple_calculation.initialize()

        processes = ['Diffractie', 'Transmissie', 'Lokale golfgroei']
        names = ['diffraction', 'transmission', 'wavegrowth', 'combinedresults']

        self.mainmodel.simple_calculation.run_all(processes=processes)

        for name in names:
            table_path = f'{datadir}/simple/{name}_results_colijnsplaat.gz'
            table = getattr(self.mainmodel.simple_calculation, name).output
            common.compare_or_save(self, table_path, table)

        # Model uncertainties
        common.set_model_uncertainties(self.mainmodel)

    def test3_exporteren(self):
        logger.info('TEST3_EXPORTEREN')

        # Exporteren naar hrd
        # ========================================
        # In deze stap exporteren we de resultaten naar de HRD. We maken hiervoor eerst een kopie van de originele database.
        # Maak een kopie van de hrd, config en hlcd
        tmp_paths = common.create_tmp_db_copies(paths, ext='simple')

        self.mainmodel.project.settings['export']['export_HLCD_and_config'] = False
        self.mainmodel.project.settings['export']['remove_original_HRD_loc'] = True

        # Initialiseer exportmodel
        self.mainmodel.export.add_result_locations()
        # Vul exportnamen in
        self.mainmodel.export.export_dataframe['Exportnaam'] = self.mainmodel.export.export_dataframe['Naam'].values
        # Vul database in
        self.mainmodel.project.settings['export']['HRD'] = tmp_paths['hrd']
        # Exporteer
        self.mainmodel.export.export_output_to_database()

        # Get resultdata for first added location in HRD
        conn = sqlite3.connect(tmp_paths['hrd'])
        names = "','".join(self.mainmodel.schematisation.result_locations['Naam'].tolist())
        hrdlocationids = np.stack(
            conn.execute(f"SELECT HRDLocationId FROM HRDLocations WHERE Name IN ('{names}');").fetchall()
        ).squeeze()
        supportlocid = self.mainmodel.schematisation.support_location['HRDLocationId']

        # Test the number of entries per input column
        input_columns = ['Wind speed', 'Wind direction']

        # Counts from db
        for col in input_columns:
            # Get the counts for the column
            counts = common.get_db_count_os(conn, col, [supportlocid] + hrdlocationids.tolist())
            # Select the support location
            check_count = counts.loc[supportlocid]
            # Compare
            faulty = counts.index[~(counts == check_count).all(axis=1).values].tolist()
            # Check if equal
            msg = f"Wrong number of {col} for {faulty}: {counts.loc[faulty]} instead of {counts.loc[supportlocid]}.",
            self.assertListEqual(faulty, [], msg=msg)

        # Check if the names in the database match the Locations
        hrdlocations = pd.read_sql('select Name, HRDLocationId from HRDLocations', con=conn)
        hrd_names = set(hrdlocations['Name'].tolist())
        hbh_names = self.mainmodel.export.export_dataframe['Exportnaam'].tolist()

        # Check if all hbh names are in hrd
        self.assertTrue(hrd_names.issuperset(set(hbh_names)))

        # Check if only hbh + steuntpunt names are in hrd
        # check since self.mainmodel.project.settings['export']['remove_original_HRD_loc'] = True
        self.assertTrue(len(hrdlocations) == len(hbh_names) + 1)

        # Controleer de waarden uit database, voor de eerste, middelste en laatste locatie
        names = [hbh_names[0], hbh_names[len(hbh_names) // 2], hbh_names[-1]]
        hrdlocationids = [hrdlocations.set_index('Name').at[name, 'HRDLocationId'] for name in names]

        for locid, name in zip(hrdlocationids, names):

            hrd = HRDio(tmp_paths['hrd'])
            hrd_table = hrd.read_HydroDynamicData(hrdlocationid=locid)
            self.assertEqual(hrd.dbformat, 'OS2023')

            # Get table with loaddata from swan results
            # Determine hbh-location for hrdlocationids
            hbhname = self.mainmodel.export.export_dataframe.set_index('Exportnaam').at[name, 'Naam']
            result_parameters = {
                'Hs,out': 'Hs',
                'Tp': 'Tp',
                'Tm-1,0': 'Tm-1,0',
                'Combined wave direction': 'Wave direction',
            }

            # Get results
            simple_table = self.mainmodel.simple_calculation.combinedresults.output[
                list(result_parameters) + ['HydraulicLoadId', 'Location']
            ].set_index('HydraulicLoadId')
            simple_table = simple_table.loc[simple_table['Location'].eq(hbhname)].drop('Location', axis=1)
            # Merge with hydraulic loads
            load_vars = ['Wind direction', 'Wind speed', 'Water level']
            simple_table = simple_table.join(self.mainmodel.hydraulic_loads[load_vars])
            simple_table.rename(columns=result_parameters, inplace=True)

            # Remove white space for itertupels
            simple_table = simple_table.sort_index(axis=1).sort_values(by=load_vars).reset_index(drop=True)
            # Sort the tables and compare
            hrd_table = (
                hrd_table.dropna(how='all', axis=1).sort_index(axis=1).sort_values(by=load_vars).reset_index(drop=True)
            )

            # Compare
            common.compare_dataframes(self, simple_table, hrd_table, round=4)

        conn.close()

        # Verwijder
        # shutil.rmtree(paths['temp'])        

if __name__ == '__main__':
    unittest.main(verbosity=2)
