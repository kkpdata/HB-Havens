import logging
import os
import shutil
import sqlite3
import unittest

import numpy as np
import pandas as pd

from hbhavens.core.logger import initialize_logger
from hbhavens.core.models import MainModel
from hbhavens.io.database import HRDio

import common

initialize_logger()
logger = logging.getLogger()

datadir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "colijnsplaat")

paths = {
    "harborarea": os.path.join(datadir, "shapes", "haventerrein.shp"),
    "breakwater": os.path.join(datadir, "shapes", "havendammen.shp"),
    "hrd": os.path.join(datadir, "databases", "WBI2023_Oosterschelde_28-1_v06_selectie.sqlite"),
    "result_locations": os.path.join(datadir, "shapes", "uitvoerlocaties.shp"),
}


class TestSimpleOosterschelde(unittest.TestCase):
    def setUp(self):
        logger.info("SETUP TestSimpleOosterschelde")
        self.mainmodel = MainModel()
        prjjson = "temp/colijnsplaat.json"
        if os.path.exists(prjjson) and not self._testMethodName[0:5] == 'test1':
            self.mainmodel.project.open_from_file(prjjson)

    def tearDown(self):
        logger.info("TEARDOWN TestSimpleOosterschelde")
        if os.path.exists("temp"):
            self.mainmodel.project.save_as("temp/colijnsplaat.json", overwrite_all=True)
            self.mainmodel.save_tables()

    def test1_schematisation_Colijnsplaat(self):
        """
        Test preparing Colijnsplaat schematisation
        """
        logger.info("TEST1_SCHEMATISATION_COLIJNSPLAAT")
        if not os.path.exists("temp"):
            os.mkdir("temp")

        supportloc = "OS_3_hy04-00000"
        bedlevel = -10.0
        trajecten = ["28-1"]
        common.add_schematisation(
            self.mainmodel, paths=paths, supportlocationname=supportloc, trajecten=trajecten, bedlevel=bedlevel
        )

        # Test flood defence
        self.assertListEqual(self.mainmodel.schematisation.flooddefence["traject-id"].tolist(), trajecten)
        # Test harbor area
        self.assertAlmostEqual(self.mainmodel.schematisation.inner.area, 126391.07426655896)
        # Test support location name
        self.assertEqual(self.mainmodel.schematisation.support_location["Name"], supportloc)

        # Geef eerst in de projectinstellingen aan dat de eenvoudige methode gebruikt moet worden
        self.mainmodel.project.settings["calculation_method"]["method"] = "simple"

    def test2_calculation(self):
        logger.info("TEST2_CALCULATION")

        # Shuffle hydraulic loads
        common.randomize_table(self.mainmodel.hydraulic_loads)

        def test_process(name, cols=2):
            table_path = f"{datadir}/simple/{name}_results_colijnsplaat.gz"
            table = getattr(self.mainmodel.simple_calculation, name).output.round(5)
            # Check if the results are equal to a test table
            test_table = pd.read_csv(table_path, sep=";", decimal=".", index_col=list(range(cols)))
            # Check the results
            common.compare_dataframes(self, test_table.sort_index(), table.sort_index(), round=3, fillna=0.0)

        self.mainmodel.simple_calculation.initialize()

        self.mainmodel.simple_calculation.diffraction.run()
        test_process("diffraction")

        self.mainmodel.simple_calculation.transmission.run()
        test_process("transmission")

        self.mainmodel.simple_calculation.wavegrowth.run()
        test_process("wavegrowth")

        self.mainmodel.simple_calculation.wavebreaking.run()
        test_process("wavebreaking")

        # Shuffle hydraulic loads
        common.randomize_table(self.mainmodel.simple_calculation.diffraction.output)
        common.randomize_table(self.mainmodel.simple_calculation.transmission.output)
        common.randomize_table(self.mainmodel.simple_calculation.wavegrowth.output)
        common.randomize_table(self.mainmodel.simple_calculation.wavebreaking.output)

        self.mainmodel.simple_calculation.combinedresults.run(
            ["Diffractie", "Transmissie", "Lokale golfgroei", "Golfbreking"]
        )

        test_process("combinedresults", cols=1)

        self.mainmodel.project.settings["simple"]["finished"] = True

        # Model uncertainties
        common.set_model_uncertainties(self.mainmodel)

    def test3_exporteren(self):
        logger.info("TEST3_EXPORTEREN")

        # Exporteren naar hrd
        # ========================================
        # In deze stap exporteren we de resultaten naar de HRD. We maken hiervoor eerst een kopie van de originele database.
        # Maak een kopie van de hrd, config en hlcd
        tmp_paths = common.create_tmp_db_copies(paths, ext="simple")

        self.mainmodel.project.settings["export"]["export_HLCD_and_config"] = False

        # Initialiseer exportmodel
        self.mainmodel.export.add_result_locations()
        # Vul exportnamen in
        self.mainmodel.export.export_dataframe["Exportnaam"] = self.mainmodel.export.export_dataframe["Naam"].values
        # Vul database in
        self.mainmodel.export.export_dataframe.loc[:, "SQLite-database"] = tmp_paths["hrd"]
        # Exporteer
        self.mainmodel.export.export_output_to_database()

        # Get resultdata for first added location in HRD
        conn = sqlite3.connect(tmp_paths["hrd"])
        names = '","'.join(self.mainmodel.schematisation.result_locations["Naam"].tolist())
        hrdlocationids = np.stack(
            conn.execute(f'SELECT HRDLocationId FROM HRDLocations WHERE Name IN ("{names}");').fetchall()
        ).squeeze()
        supportlocid = self.mainmodel.schematisation.support_location["HRDLocationId"]

        # Test the number of entries per input column
        input_columns = ["Wind speed", "Wind direction"]

        # Counts from db
        for col in input_columns:
            # Get the counts for the column
            counts = common.get_db_count_os(conn, col, [supportlocid] + hrdlocationids.tolist())
            # Select the support location
            check_count = counts.loc[supportlocid]
            # Compare
            faulty = counts.index[~(counts == check_count).all(axis=1).values].tolist()
            # Check if equal
            msg = f'Wrong number of "{col} for {faulty}: {counts.loc[faulty]} instead of {counts.loc[supportlocid]}.'
            self.assertListEqual(faulty, [], msg=msg)

        # Check if the names in the database match the Locations
        hrdlocations = pd.read_sql("select Name, HRDLocationId from HRDLocations", con=conn)
        hrd_names = set(hrdlocations["Name"].tolist())
        hbh_names = self.mainmodel.export.export_dataframe["Exportnaam"].tolist()

        # Check if all hbh names are in hrd
        self.assertTrue(hrd_names.issuperset(set(hbh_names)))

        # Controleer de waarden uit database, voor de eerste, middelste en laatste locatie
        names = [hbh_names[0], hbh_names[len(hbh_names) // 2], hbh_names[-1]]
        hrdlocationids = [hrdlocations.set_index("Name").at[name, "HRDLocationId"] for name in names]

        for locid, name in zip(hrdlocationids, names):

            hrd = HRDio(tmp_paths["hrd"])
            hrd_table = hrd.read_HydroDynamicData(hrdlocationid=locid)
            self.assertEqual(hrd.dbformat, "OS2023")

            # Get table with loaddata from swan results
            # Determine hbh-location for hrdlocationids
            hbhname = self.mainmodel.export.export_dataframe.set_index("Exportnaam").at[name, "Naam"]
            result_parameters = {
                "Hs,out": "Hs",
                "Tp": "Tp",
                "Tm-1,0": "Tm-1,0",
                "Combined wave direction": "Wave direction",
            }

            # Get results
            simple_table = self.mainmodel.simple_calculation.combinedresults.output[
                list(result_parameters) + ["HydraulicLoadId", "Location"]
            ].set_index("HydraulicLoadId")
            simple_table = simple_table.loc[simple_table["Location"].eq(hbhname)].drop("Location", axis=1)
            # Merge with hydraulic loads
            load_vars = ["Wind direction", "Wind speed", "Water level"]
            simple_table = simple_table.join(self.mainmodel.hydraulic_loads[load_vars])
            simple_table.rename(columns=result_parameters, inplace=True)

            # Remove white space for itertupels
            simple_table = simple_table.sort_index(axis=1).sort_values(by=load_vars).reset_index(drop=True)
            # Sort the tables and compare
            hrd_table = (
                hrd_table.dropna(how="all", axis=1).sort_index(axis=1).sort_values(by=load_vars).reset_index(drop=True)
            )

            # Compare
            common.compare_dataframes(self, simple_table, hrd_table, round=4)

        conn.close()

        # Verwijder
        shutil.rmtree("temp")


if __name__ == "__main__":
    unittest.main(verbosity=2)
