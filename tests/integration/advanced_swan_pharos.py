import os
import shutil
import sqlite3
import unittest
import zipfile

import pandas as pd

from hbhavens.core.logger import initialize_logger

# Specifieke hb havens modules
from hbhavens.core.models import MainModel
from hbhavens.io.database import HRDio

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


class TestSwanPharos(unittest.TestCase):
    def setUp(self):
        logger.info("SETUP TestSwanPharos")
        self.mainmodel = MainModel()
        prjjson = "temp/eemshaven.json"
        if os.path.exists(prjjson) and not self._testMethodName[0:5] == 'test1':
            self.mainmodel.project.open_from_file(prjjson)

    def tearDown(self):
        logger.info("TEARDOWN TestSwanPharos")
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

    def test2_swan(self):

        # Berekening geavanceerde methode
        # ========================================
        if os.path.exists("temp/pharos"):
            shutil.rmtree("temp/pharos")
        swan_dst = "temp/swan"
        if not os.path.exists(swan_dst):
            os.makedirs(swan_dst)

        # Stel in voor geavanceerde methode
        self.mainmodel.project.settings["calculation_method"]["method"] = "advanced"
        self.mainmodel.project.settings["calculation_method"]["include_pharos"] = True
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

            # Read the results
            self.mainmodel.swan.iteration_results.read_results(step)

            # If there are no test results yet, save
            table_path = f"{datadir}/swan/{step}_results_table.gz"
            if not os.path.exists(table_path):
                self.mainmodel.swan.iteration_results.to_csv(table_path, sep=";", decimal=".")

            # Check if the results are equal to a test table
            test_table = pd.read_csv(table_path, sep=";", decimal=".", index_col=0)

            # Check the results
            common.compare_dataframes(
                self, test_table.sort_index(), self.mainmodel.swan.iteration_results.sort_index(), fillna=0.0, round=3
            )

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
            common.randomize_table(self.mainmodel.schematisation.result_locations)
            common.randomize_table(self.mainmodel.swan.calculation_results)

            # Read the results
            self.mainmodel.swan.calculation_results.read_results(step)

            # If there are no test results yet, save
            if step == "W":
                table_path = f"{datadir}/swan/{step}_results_table_{addition}.gz"
            else:
                table_path = f"{datadir}/swan/{step}_results_table.gz"
            if not os.path.exists(table_path):
                self.mainmodel.swan.calculation_results.to_csv(table_path, sep=";", decimal=".")

            # Check if the results are equal to a test table
            test_table = pd.read_csv(table_path, sep=";", decimal=".", index_col=0)

            # Check the results
            common.compare_dataframes(
                self,
                test_table.sort_values(by=["Location", "HydraulicLoadId"]),
                self.mainmodel.swan.calculation_results.sort_values(by=["Location", "HydraulicLoadId"]),
                fillna=0.0,
                round=3,
            )

    def test3_pharos(self):

        pharos_dst = "temp/pharos"
        if not os.path.exists(pharos_dst):
            os.makedirs(pharos_dst)

        # Pharos settings
        self.mainmodel.project.settings["pharos"]["hydraulic loads"]["water depth for wave length"] = 10
        self.mainmodel.project.settings["pharos"]["hydraulic loads"]["Hs_max"] = 5.2

        self.mainmodel.project.settings["pharos"]["frequencies"]["lowest"] = 0.082
        self.mainmodel.project.settings["pharos"]["frequencies"]["highest"] = 0.25
        self.mainmodel.project.settings["pharos"]["frequencies"]["number of bins"] = 17
        self.mainmodel.project.settings["pharos"]["frequencies"]["scale"] = "logaritmisch"
        self.mainmodel.project.settings["pharos"]["frequencies"]["checked"] = [
            0.082,
            0.0879,
            0.0942,
            0.1010,
            0.1083,
            0.1161,
            0.1245,
            0.1335,
            0.1431,
            0.1535,
            0.1645,
            0.1764,
            0.1891,
            0.2028,
            0.2174,
            0.2331,
            0.25,
        ]

        self.mainmodel.project.settings["pharos"]["2d wave spectrum"]["gamma"] = 2
        self.mainmodel.project.settings["pharos"]["2d wave spectrum"]["min energy"] = 0.01
        self.mainmodel.project.settings["pharos"]["2d wave spectrum"]["spread"] = 30.0

        self.mainmodel.project.settings["pharos"]["paths"]["pharos folder"] = pharos_dst
        self.mainmodel.project.settings["pharos"]["paths"]["schematisation folder"] = f"{pharos_dst}/schematisations"

        self.mainmodel.project.settings["pharos"]["water levels"]["checked"] = [4.0]

        self.mainmodel.project.settings["pharos"]["transformation"]["dx"] = -240000
        self.mainmodel.project.settings["pharos"]["transformation"]["dy"] = -600000

        self.mainmodel.project.settings["pharos"]["schematisations"] = {
            "Grid_c02": [
                120.0,
                290.0,
                295.0,
                300.0,
                305.0,
                310.0,
                315.0,
                320.0,
                325.0,
                330.0,
                335.0,
                340.0,
                345.0,
                350.0,
                355.0,
                0.0,
                5.0,
                10.0,
                15.0,
                20.0,
                25.0,
                30.0,
                35.0,
                40.0,
                45.0,
                50.0,
                55.0,
                60.0,
                65.0,
                70.0,
                75.0,
                80.0,
                85.0,
                90.0,
                95.0,
                100.0,
                105.0,
                110.0,
                115.0,
            ]
        }

        # Copy schematisations
        shutil.copytree(f"{datadir}/pharos/schematisations", f"{pharos_dst}/schematisations")
        os.mkdir(f"{pharos_dst}/calculations")

        self.mainmodel.pharos.initialize()

        self.mainmodel.pharos.fill_spectrum_table()
        common.randomize_table(self.mainmodel.pharos.spectrum_table)

        self.mainmodel.pharos.generate()

        # Copy output
        with zipfile.ZipFile(f"{datadir}/pharos/json_output.zip", "r") as zip_ref:
            zip_ref.extractall(f"{pharos_dst}/calculations/Grid_c02/output")

        self.mainmodel.pharos.read_calculation_results()

        for key, waves in self.mainmodel.pharos.waves.items():
            # If there are no test results yet, save
            table_path = f"{datadir}/pharos/waves_{key}_table.gz"
            # common.compare_or_save(self, table_path, waves.sort_index(), cols=2)
            if not os.path.exists(table_path):
                waves.sort_index().to_csv(table_path, sep=";", decimal=".")
            # Check if the results are equal to a test table
            test_table = pd.read_csv(table_path, sep=";", decimal=".", index_col=[0, 1]).sort_index()
            # Check the results
            common.compare_dataframes(
                self, test_table.sort_index().reset_index().round(3), waves.sort_index().reset_index().round(3)
            )

        common.randomize_table(self.mainmodel.swan.calculation_results)
        common.randomize_table(self.mainmodel.pharos.calculation_results)
        self.mainmodel.pharos.assign_energies()

        # If there are no test results yet, save
        table_path = f"{datadir}/pharos/calculation_results_table.gz"
        common.compare_or_save(
            self,
            table_path,
            self.mainmodel.pharos.calculation_results.fillna(0.0).sort_values(by=["LocationId", "HydraulicLoadId"]),
            cols=1,
        )

        # Definieer modelonzekerheden
        # ========================================
        common.set_model_uncertainties(self.mainmodel)

    def test4_export(self):

        # Exporteren naar hrd
        # ========================================
        # In deze stap exporteren we de resultaten naar de HRD. We maken hiervoor eerst een kopie van de originele database.
        # Maak een kopie van de hrd, config en hlcd
        tmp_paths = common.create_tmp_db_copies(paths, ext="swan_pharos")
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
                msg=f'Wrong number of "{col} for {faulty}: {count.loc[faulty]} instead of {count.loc[supportlocid]}.',
            )

        # Check if the names in the database match the Locations
        conn = sqlite3.connect(tmp_paths["hrd"])
        hrdlocations = pd.read_sql("select Name, HRDLocationId from HRDLocations", con=conn)
        conn.close()
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
            pharos_table = self.mainmodel.pharos.calculation_results[
                ["Location", "HydraulicLoadId", "Tm-1,0 totaal", "Hs totaal", "Wave direction totaal"]
            ].set_index("HydraulicLoadId")
            pharos_table = pharos_table.loc[pharos_table["Location"].eq(hbhname)].drop("Location", axis=1)
            # Merge with hydraulic loads
            load_vars = ["ClosingSituationId", "Wind direction", "Wind speed", "Water level"]
            pharos_table = (
                pharos_table.join(self.mainmodel.hydraulic_loads[load_vars])
                .sort_index(axis=1)
                .sort_values(by=load_vars)
                .reset_index(drop=True)
            )
            pharos_table.columns = [col.replace(" totaal", "") for col in pharos_table.columns]

            # Sort the tables and compare
            hrd_table = hrd_table.sort_index(axis=1).sort_values(by=load_vars).reset_index(drop=True)

            # Compare
            common.compare_dataframes(self, pharos_table.round(3), hrd_table.round(3))

        # Verwijder
        shutil.rmtree("temp")


if __name__ == "__main__":
    unittest.main(verbosity=2)
