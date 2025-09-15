import os
import unittest
import sqlite3
import pandas as pd

import hbhavens
from hbhavens.core.models import MainModel
from hbhavens.io.database import HRDio

import logging
from hbhavens.core.logger import initialize_logger

import common


datadir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'urk')
trunkdir = os.path.abspath(os.path.join(os.path.dirname(hbhavens.__file__),'../..'))
tempdir = os.path.join(trunkdir, 'temp')

paths = {
    'temp': os.path.join(tempdir, 'urk'),
    'harborarea': os.path.join(datadir, 'shapes', 'haventerrein.shp'),
    'breakwater': os.path.join(datadir, 'shapes', 'havendammen.shp'),
    'hrd': os.path.join(datadir, 'databases', 'WBI2017_IJsselmeer_7-2_v02.sqlite'),
    'config': os.path.join(datadir, 'databases', 'WBI2017_IJsselmeer_7-2_v02.config.sqlite'),
    'result_locations_save': os.path.join(datadir, 'shapes', 'uitvoerlocaties.shp'),
    'hlcd': os.path.join(datadir, '..', 'hlcd.sqlite'),
}

initialize_logger(filename=os.path.join(tempdir,'hbhavens.log'))
logger = logging.getLogger()
class TestSimpleZoetUrk(unittest.TestCase):

    def setUp(self):
        logger.info('SETUP TestSimpleZoetUrk')
        self.mainmodel = MainModel()
        prjjson = os.path.join(paths['temp'],'urk.json')
        if os.path.exists(prjjson) and not self._testMethodName[0:5] == 'test1':
            self.mainmodel.project.open_from_file(prjjson)

    def tearDown(self):
        logger.info('TEARDOWN TestSimpleZoetUrk')
        if os.path.exists(paths['temp']):
            self.mainmodel.project.save_as(os.path.join(paths['temp'],'urk.json'), overwrite_all=True)
            self.mainmodel.save_tables()

    def test1_schematisation_Urk(self):
        '''
        Test preparing Urk schematisation
        '''
        logger.info('TEST1_SCHEMATISATION_URK')
        if not os.path.exists(paths['temp']):
            os.mkdir(paths['temp'])

        supportloc = 'YM_3_7-2_ha02_00001'
        bedlevel = -5.0
        trajecten = ['7-2']
        common.add_schematisation(self.mainmodel, paths=paths, supportlocationname=supportloc, trajecten=trajecten, bedlevel=bedlevel)

        # Test flood defence
        self.assertListEqual(self.mainmodel.schematisation.flooddefence['traject-id'].tolist(), trajecten)
        # Test harbor area
        self.assertAlmostEqual(self.mainmodel.schematisation.inner.area, 205769.77851501762)
        # Test support location name
        self.assertEqual(self.mainmodel.schematisation.support_location['Name'], supportloc)

        # Geef eerst in de projectinstellingen aan dat de eenvoudige methode gebruikt moet worden
        self.mainmodel.project.settings['calculation_method']['method'] = 'simple'
        self.mainmodel.schematisation.plot_schematisation(save=True, path=paths['temp'], close=True)

        fig, ax = self.mainmodel.schematisation.plot_schematisation(buffer=10)
        ax.legend(loc='upper left')
        # fig.show()
        fig.savefig(os.path.join(paths['temp'],'urk.png'), dpi=150)

    def test2_calculation(self):

        # Shuffle hydraulic loads
        logger.info('TEST2_CALCULATION')
        common.randomize_table(self.mainmodel.hydraulic_loads)

        # Run all
        processes = ['Diffractie', 'Transmissie', 'Lokale golfgroei', 'Golfbreking']
        names = ['diffraction', 'transmission', 'wavegrowth', 'wavebreaking']
        
        self.mainmodel.simple_calculation.run_all(processes=processes)

        # Test the results
        for name in names:
            table_path = f'{datadir}/simple/{name}_results_urk.gz'
            table = getattr(self.mainmodel.simple_calculation, name).output.round(3)
            common.compare_or_save(self, table_path, table)
            
        # Model uncertainties
        common.set_model_uncertainties(self.mainmodel)
    
    def test3_exporteren(self):
        logger.info('TEST3_EXPORTEREN')
        
        # Exporteren naar hrd
        #========================================
        # In deze stap exporteren we de resultaten naar de HRD. We maken hiervoor eerst een kopie van de originele database.
        # Maak een kopie van de hrd, config en hlcd
        tmp_paths = common.create_tmp_db_copies(paths, ext='simple')
        common.export_to_database(self.mainmodel, tmp_paths, export_hlcd_config=True)

        # Count the number of results, grouped per input variable
        counts, supportlocid = common.get_counts(self.mainmodel, tmp_paths)
        # Test the number of entries per input column
        for col, count in counts.items():
            # Check where the counts do not match the count of the support location
            faulty = count.index[~(count == count.loc[supportlocid]).all(axis=1).values].tolist()
            # Check if the list is equal
            self.assertListEqual(
                faulty,
                [],
                msg = f"Wrong number of {col} for {faulty}: {count.loc[faulty]} instead of {count.loc[supportlocid]}.",
            )

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
        
        for locid, name in zip(hrdlocationids, names):

            hrd_table = hrd.read_HydroDynamicData(hrdlocationid=locid)

            # Get table with loaddata from swan results
            # Determine hbh-location for hrdlocationids
            hbhname = self.mainmodel.export.export_dataframe.set_index('Exportnaam').at[name, 'Naam']
            result_parameters = self.mainmodel.simple_calculation.result_parameters
            
            # Get results
            hlcols = self.mainmodel.hydraulic_loads.columns.tolist()
            columns = [val for key, val in result_parameters.items() if key in hlcols]
            simple_table = self.mainmodel.simple_calculation.combinedresults.output[columns + ['HydraulicLoadId', 'Location']].set_index('HydraulicLoadId')
            simple_table = simple_table.loc[simple_table['Location'].eq(hbhname)].drop('Location', axis=1)
            # Merge with hydraulic loads
            load_vars = ['ClosingSituationId', 'Wind direction', 'Wind speed', 'Water level IJssel lake']
            simple_table = simple_table.join(self.mainmodel.hydraulic_loads[load_vars])
            simple_table.rename(columns={v: k for k, v in result_parameters.items()}, inplace=True)
            
            # Remove white space for itertupels
            simple_table = simple_table.sort_index(axis=1).sort_values(by=load_vars).reset_index(drop=True)
            # Sort the tables and compare
            hrd_table = hrd_table.sort_index(axis=1).sort_values(by=load_vars).reset_index(drop=True)
                        
            # Compare
            common.compare_dataframes(self, simple_table.round(4), hrd_table.round(4))

    def test4_exporteren(self):
        logger.info('TEST4_EXPORTEREN')
        
        # Exporteren naar hrd
        #========================================
        # In deze stap exporteren we de resultaten naar de HRD. We maken hiervoor eerst een kopie van de originele database.
        # Maak een kopie van de hrd, config en hlcd
        # In deze test verwijderen we de originele HRD locaties en controleren of deze daadwerkelijk ontbreken
        tmp_paths = common.create_tmp_db_copies(paths, ext='simple_removeHRD')
        self.mainmodel.export.settings['export']['remove_original_HRD_loc'] = True
        common.export_to_database(self.mainmodel, tmp_paths, export_hlcd_config=True)

        # Count the number of results, grouped per input variable
        counts, supportlocid = common.get_counts(self.mainmodel, tmp_paths)
        # Test the number of entries per input column
        for col, count in counts.items():
            # Check where the counts do not match the count of the support location
            faulty = count.index[~(count == count.loc[supportlocid]).all(axis=1).values].tolist()
            # Check if the list is equal
            self.assertListEqual(
                faulty,
                [],
                msg = f"Wrong number of {col} for {faulty}: {count.loc[faulty]} instead of {count.loc[supportlocid]}.",
            )

        # Check if the names in the database match the Locations
        conn = sqlite3.connect(tmp_paths['hrd'])
        hrdlocations = pd.read_sql('select Name, HRDLocationId from HRDLocations', con=conn)
        conn.close()
        hrd_names = set(hrdlocations['Name'].tolist())
        hbh_names = self.mainmodel.export.export_dataframe['Exportnaam'].tolist()
        conn = sqlite3.connect(paths['hrd'])
        orilocations = pd.read_sql('select Name, HRDLocationId from HRDLocations', con=conn)
        conn.close()
        ori_names = set(orilocations['Name'].tolist())
        steunpunt = self.mainmodel.schematisation.support_location['Name']

        # Check if all hbh names are in hrd
        self.assertTrue(hrd_names.issuperset(set(hbh_names)))
        # Check if steunpunt is in hrd
        self.assertTrue(steunpunt in hrd_names)
        # Check if original HRD are not in hrd (behalve steunpunt)
        self.assertTrue(all([loc not in hrd_names for loc in ori_names if loc != steunpunt]))
        
        # Controleer de waarden uit database, voor de eerste, middelste en laatste locatie
        names = [hbh_names[0], hbh_names[len(hbh_names) // 2], hbh_names[-1]]
        hrdlocationids = [hrdlocations.set_index('Name').at[name, 'HRDLocationId'] for name in names]
                
        hrd = HRDio(tmp_paths['hrd'])
        self.assertEqual(hrd.dbformat, 'WBI2017')
        
        for locid, name in zip(hrdlocationids, names):

            hrd_table = hrd.read_HydroDynamicData(hrdlocationid=locid)

            # Get table with loaddata from swan results
            # Determine hbh-location for hrdlocationids
            hbhname = self.mainmodel.export.export_dataframe.set_index('Exportnaam').at[name, 'Naam']
            result_parameters = self.mainmodel.simple_calculation.result_parameters
            
            # Get results
            hlcols = self.mainmodel.hydraulic_loads.columns.tolist()
            columns = [val for key, val in result_parameters.items() if key in hlcols]
            simple_table = self.mainmodel.simple_calculation.combinedresults.output[columns + ['HydraulicLoadId', 'Location']].set_index('HydraulicLoadId')
            simple_table = simple_table.loc[simple_table['Location'].eq(hbhname)].drop('Location', axis=1)
            # Merge with hydraulic loads
            load_vars = ['ClosingSituationId', 'Wind direction', 'Wind speed', 'Water level IJssel lake']
            simple_table = simple_table.join(self.mainmodel.hydraulic_loads[load_vars])
            simple_table.rename(columns={v: k for k, v in result_parameters.items()}, inplace=True)
            
            # Remove white space for itertupels
            simple_table = simple_table.sort_index(axis=1).sort_values(by=load_vars).reset_index(drop=True)
            # Sort the tables and compare
            hrd_table = hrd_table.sort_index(axis=1).sort_values(by=load_vars).reset_index(drop=True)
                        
            # Compare
            common.compare_dataframes(self, simple_table.round(4), hrd_table.round(4))

        # Verwijder
        # shutil.rmtree(paths['temp'])
    
if __name__ == '__main__':
    unittest.main(verbosity=2)