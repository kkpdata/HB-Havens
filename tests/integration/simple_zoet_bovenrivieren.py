import logging
import os
import shutil
import sqlite3
import unittest

import pandas as pd

from hbhavens.core.logger import initialize_logger
from hbhavens.core.models import MainModel
from hbhavens.io.database import HRDio

import common

initialize_logger()
logger = logging.getLogger()

datadir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "vluchthaven")

paths = {
    "harborarea": os.path.join(datadir, "shapes", "haventerrein.shp"),
    "breakwater": os.path.join(datadir, "shapes", "havendam.shp"),
    "result_locations": os.path.join(datadir, "shapes", "uitvoerlocaties.shp"),
    "hrd": os.path.join(datadir, "databases", "WBI2017_Bovenrijn_43-6_v04_steunpunt.sqlite"),
    "config": os.path.join(datadir, "databases", "WBI2017_Bovenrijn_43-6_v04_steunpunt.config.sqlite"),
    "hlcd": os.path.join(datadir, "..", "hlcd_nieuw.sqlite"),
}


class TestSimpleBovenrivieren(unittest.TestCase):
    def setUp(self):
        logger.info("SETUP TestSimpleZoetBovenrivieren")
        self.mainmodel = MainModel()
        prjjson = "temp/vluchthaven.json"
        if os.path.exists(prjjson) and not self._testMethodName[0:5] == 'test1':
            self.mainmodel.project.open_from_file(prjjson)

    def tearDown(self):
        logger.info("TEARDOWN TestSimpleZoetBovenrivieren")
        if os.path.exists("temp"):
            self.mainmodel.project.save_as("temp/vluchthaven.json", overwrite_all=True)
            self.mainmodel.save_tables()

    def test1_schematisation_Vluchthaven(self):
        """
        Test preparing Vluchthaven schematisation
        """
        logger.info("TEST1_SCHEMATISATION_VLUCHTHAVEN")
        if not os.path.exists("temp"):
            os.mkdir("temp")

        # Add entrance coordinate
        crd = (158800.0, 433050.0)
        self.mainmodel.schematisation.entrance = crd
        self.mainmodel.project.settings["schematisation"]["entrance_coordinate"] = crd

        # Add schematisation
        supportloc = "043-06_0023_1_WA_km0914"
        bedlevel = 0.0
        trajecten = ["43-6"]
        common.add_schematisation(
            self.mainmodel, paths=paths, supportlocationname=supportloc, trajecten=trajecten, bedlevel=bedlevel
        )

        # Test flood defence
        self.assertListEqual(self.mainmodel.schematisation.flooddefence["traject-id"].tolist(), trajecten)
        # Test harbor area
        self.assertAlmostEqual(self.mainmodel.schematisation.inner.area, 127205.93372931979)
        # Test support location name
        self.assertEqual(self.mainmodel.schematisation.support_location["Name"], supportloc)

        # Geef eerst in de projectinstellingen aan dat de eenvoudige methode gebruikt moet worden
        self.mainmodel.project.settings["calculation_method"]["method"] = "simple"

    def test2_calculation(self):

        # Shuffle hydraulic loads
        logger.info("TEST2_CALCULATION")
        common.randomize_table(self.mainmodel.hydraulic_loads)

        # Run all
        processes = ["Diffractie", "Transmissie", "Lokale golfgroei", "Golfbreking"]
        names = ["diffraction", "transmission", "wavegrowth", "wavebreaking"]

        common.randomize_table(self.mainmodel.schematisation.result_locations)
        self.mainmodel.simple_calculation.run_all(processes=processes)

        # Test the results
        for name in names:
            table_path = f"{datadir}/simple/{name}_results_vluchthaven.gz"
            table = getattr(self.mainmodel.simple_calculation, name).output
            common.compare_or_save(self, table_path, table)

        # Model uncertainties
        common.set_model_uncertainties(self.mainmodel)

    def test3_exporteren(self):
        logger.info("TEST3_EXPORTEREN")

        # Exporteren naar hrd
        # ========================================
        # In deze stap exporteren we de resultaten naar de HRD. We maken hiervoor eerst een kopie van de originele database.
        # Maak een kopie van de hrd, config en hlcd
        tmp_paths = common.create_tmp_db_copies(paths, ext="simple")
        common.randomize_table(self.mainmodel.schematisation.result_locations)
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

        hrd = HRDio(tmp_paths["hrd"])
        self.assertEqual(hrd.dbformat, "WBI2017")

        for locid, name in zip(hrdlocationids, names):

            hrd_table = hrd.read_HydroDynamicData(hrdlocationid=locid)

            # Get table with loaddata from swan results
            # Determine hbh-location for hrdlocationids
            hbhname = self.mainmodel.export.export_dataframe.set_index("Exportnaam").at[name, "Naam"]
            result_parameters = self.mainmodel.simple_calculation.result_parameters

            # Get results
            hlcols = self.mainmodel.hydraulic_loads.columns.tolist()
            columns = [val for key, val in result_parameters.items() if key in hlcols]
            simple_table = self.mainmodel.simple_calculation.combinedresults.output[
                columns + ["HydraulicLoadId", "Location"]
            ].set_index("HydraulicLoadId")
            simple_table = simple_table.loc[simple_table["Location"].eq(hbhname)].drop("Location", axis=1)
            # Merge with hydraulic loads
            load_vars = ["ClosingSituationId", "Wind direction", "Wind speed", "Discharge Lobith"]
            simple_table = simple_table.join(self.mainmodel.hydraulic_loads[load_vars])
            simple_table.rename(columns={v: k for k, v in result_parameters.items()}, inplace=True)

            # Remove white space for itertupels
            simple_table = simple_table.sort_index(axis=1).sort_values(by=load_vars).reset_index(drop=True)
            # Sort the tables and compare
            hrd_table = hrd_table.sort_index(axis=1).sort_values(by=load_vars).reset_index(drop=True)

            # Compare
            common.compare_dataframes(self, simple_table, hrd_table, round=4)

        # Verwijder
        shutil.rmtree("temp")


if __name__ == "__main__":
    unittest.main(verbosity=2)
