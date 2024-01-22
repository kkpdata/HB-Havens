import os
import shutil
import unittest

import sqlite3
import numpy as np
import pandas as pd

from hbhavens.core.models import MainModel
from hbhavens.io.database import HRDio
from hbhavens.core.logger import initialize_logger

import common

import logging

initialize_logger()
logger = logging.getLogger()

datadir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "eemshaven")

paths = {
    "harborarea": os.path.join(datadir, "shapes", "haventerrein.shp"),
    "breakwater": os.path.join(datadir, "shapes", "havendammen.shp"),
    "hrd": os.path.join(datadir, "databases", "WBI2017_Waddenzee_Oost_6-6_v03_selectie.sqlite"),
    "config": os.path.join(datadir, "databases", "WBI2017_Waddenzee_Oost_6-6_v03_selectie.config.sqlite"),
    "result_locations": os.path.join(datadir, "shapes", "uitvoerlocaties.shp"),
    "hlcd": os.path.join(datadir, "..", "hlcd.sqlite"),
}


class TestSwan(unittest.TestCase):
    def setUp(self):
        logger.info("SETUP TestSwan")
        self.mainmodel = MainModel()
        prjjson = "temp/eemshaven.json"
        if os.path.exists(prjjson) and not self._testMethodName[0:5] == 'test1':
            self.mainmodel.project.open_from_file(prjjson)

    def tearDown(self):
        logger.info("TEARDOWN TestSwan")
        if os.path.exists("temp"):
            self.mainmodel.project.save_as("temp/eemshaven.json", overwrite_all=True)
            self.mainmodel.save_tables()

    def test1_schematisation_Eemshaven(self):
        """
        Test preparing Eemshaven schematisation
        """
        if not os.path.exists("temp"):
            os.mkdir("temp")

        # Schematisatie
        # ========================================
        supportloc = "WZ_1_6-6_dk_00178"
        bedlevel = -10.0
        trajecten = ["6-6"]
        common.add_schematisation(
            self.mainmodel, paths=paths, supportlocationname=supportloc, trajecten=trajecten, bedlevel=bedlevel
        )

        # Test flood defence
        self.assertListEqual(self.mainmodel.schematisation.flooddefence["traject-id"].tolist(), trajecten)
        # Test harbor area
        self.assertAlmostEqual(self.mainmodel.schematisation.inner.area, 7230078.611028902)
        # Test support location name
        self.assertEqual(self.mainmodel.schematisation.support_location["Name"], supportloc)

    def test2_swan_iterations(self):
        """
        Test reading 2D spectrum.
        """
        # Berekening geavanceerde methode
        # ========================================
        swan_dst = "temp/swan"
        if not os.path.exists(swan_dst):
            os.makedirs(swan_dst)

        # Stel in voor geavanceerde methode
        self.mainmodel.project.settings["calculation_method"]["method"] = "advanced"
        self.mainmodel.project.settings["calculation_method"]["include_pharos"] = False
        self.mainmodel.project.settings["calculation_method"]["include_hares"] = False
        self.mainmodel.project.settings["swan"]["swanfolder"] = swan_dst
        self.mainmodel.project.settings["swan"]["mastertemplate"] = f"{datadir}/swan/A2_template_production2.swn"
        self.mainmodel.project.settings["swan"]["depthfile"] = f"{datadir}/swan/A2.dep"
        self.mainmodel.project.settings["swan"]["use_incoming_wave_factors"] = True

        # Genereer invoerbestanden
        for step in ["I1", "I2", "I3"]:
            self.mainmodel.swan.generate(step=step)
            # Copy testresults
            common.copy_swan_files(datadir, swan_dst, step)

            # Randomly sort the table before importing
            common.randomize_table(self.mainmodel.swan.iteration_results)
            common.randomize_table(self.mainmodel.schematisation.result_locations)

            # Read the results
            self.mainmodel.swan.iteration_results.read_results(step)

            # If there are no test results yet, save
            table_path = f"{datadir}/swan/{step}_results_table.gz"
            common.compare_or_save(self, table_path, self.mainmodel.swan.iteration_results.sort_index(), cols=1)

    def test3_swan_calculations(self):

        swan_dst = self.mainmodel.project.settings["swan"]["swanfolder"]

        # Check whether SWAN has to be taken into account or not
        include = self.mainmodel.project.settings["calculation_method"]
        only_swan = not include["include_pharos"] and not include["include_hares"]
        addition = "only_swan" if only_swan else "with_pharos"

        # Genereer invoerbestanden
        for step in ["D", "TR", "W"]:
            self.mainmodel.swan.generate(step=step)
            # Copy testresults
            common.unzip_swan_files(datadir, swan_dst, step)

            # Randomly sort the table before importing
            common.randomize_table(self.mainmodel.swan.calculation_results)
            common.randomize_table(self.mainmodel.schematisation.result_locations)

            # Read the results
            self.mainmodel.swan.calculation_results.read_results(step)

            # Compare or save
            if step == "W":
                table_path = f"{datadir}/swan/{step}_results_table_{addition}.gz"
            else:
                table_path = f"{datadir}/swan/{step}_results_table.gz"
            common.compare_or_save(
                self,
                table_path,
                self.mainmodel.swan.calculation_results.sort_values(by=["Location", "HydraulicLoadId"]),
                cols=1,
            )

    def test4_model_uncertainties(self):

        common.set_model_uncertainties(self.mainmodel)
        # TODO: Test waarden op verschillende wijzen berekend

    def test5_exporteren(self):

        # Exporteren naar hrd
        # ========================================
        # In deze stap exporteren we de resultaten naar de HRD. We maken hiervoor eerst een kopie van de originele database.
        # Maak een kopie van de hrd, config en hlcd
        tmp_paths = common.create_tmp_db_copies(paths, ext="swan")
        common.randomize_table(self.mainmodel.schematisation.result_locations)

        # Initialiseer exportmodel
        self.mainmodel.export.add_result_locations()
        self.mainmodel.export.add_HLCD(tmp_paths["hlcd"])
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
        for col in self.mainmodel.hydraulic_loads.input_columns:
            # Get the counts for the column
            counts = common.get_db_count(conn, col, [supportlocid] + hrdlocationids.tolist())
            # Select the support location
            check_count = counts.loc[supportlocid]
            # Compare
            faulty = counts.index[~(counts == check_count).all(axis=1).values].tolist()
            # Check if equal
            self.assertListEqual(
                faulty,
                [],
                msg=f'Wrong number of "{col} for {faulty}: {counts.loc[faulty]} instead of {counts.loc[supportlocid]}.',
            )

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
            self.assertEqual(hrd.dbformat, "WBI2017")

            # Get table with loaddata from swan results
            # Determine hbh-location for hrdlocationids
            hbhname = self.mainmodel.export.export_dataframe.set_index("Exportnaam").at[name, "Naam"]

            # Get swan results
            swan_table = self.mainmodel.swan.calculation_results[
                ["Location", "HydraulicLoadId", "Tm-1,0 swan", "Hm0 swan", "Wave direction swan"]
            ].set_index("HydraulicLoadId")
            swan_table = swan_table.loc[swan_table["Location"].eq(hbhname)].drop("Location", axis=1)
            # Merge with hydraulic loads
            load_vars = ["ClosingSituationId", "Wind direction", "Wind speed", "Water level"]
            swan_table = (
                swan_table.join(self.mainmodel.hydraulic_loads[load_vars])
                .sort_index(axis=1)
                .sort_values(by=load_vars)
                .reset_index(drop=True)
            )
            swan_table.columns = [col.replace(" swan", "").replace("Hm0", "Hs") for col in swan_table.columns]

            # Remove white space for itertupels
            swan_table.columns = [col.replace(" ", "_") for col in swan_table.columns]
            # Sort the tables and compare
            hrd_table = hrd_table.sort_index(axis=1).sort_values(by=load_vars).reset_index(drop=True)
            hrd_table.columns = [col.replace(" ", "_") for col in hrd_table.columns]

            # Compare
            common.compare_dataframes(self, swan_table.round(3), hrd_table.round(3))

        conn.close()

        # Verwijder
        shutil.rmtree("temp")


if __name__ == "__main__":
    unittest.main(verbosity=2)
