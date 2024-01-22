import os
import shutil
import zipfile
import sqlite3

import numpy as np
import pandas as pd

inputvariableids = {
    "Discharge Lobith": 1,
    "Discharge Lith": 2,
    "Discharge Borgharen": 3,
    "Discharge Olst": 4,
    "Discharge Dalfsen": 5,
    "Water level Maasmond": 6,
    "Water level IJssel lake": 7,
    "Water level Marker lake": 8,
    "Wind speed": 9,
    "Water level": 10,
    "Wave period": 11,
    "Sea water level": 12,
    "Wave height": 13,
    "Sea water level (u)": 14,
    "Uncertainty water level (u)": 15,
    "Storm surge duration": 16,
    "Time shift surge and tide": 17,
}


def add_schematisation(mainmodel, paths, supportlocationname, trajecten, bedlevel):

    # Add flooddefence
    for trajectid in trajecten:
        mainmodel.schematisation.add_flooddefence(trajectid)
    # Voeg haventerrein en havendammen toe
    mainmodel.schematisation.add_harborarea(paths["harborarea"])
    mainmodel.schematisation.add_breakwater(paths["breakwater"])
    # Bepaal het binnengaats gebied
    mainmodel.schematisation.generate_harbor_bound()
    mainmodel.schematisation.set_bedlevel(bedlevel)
    # Add database
    mainmodel.hydraulic_loads.add_HRD(paths["hrd"])
    # Genereer uitvoerlocaties
    if "result_locations" in paths:
        mainmodel.schematisation.add_result_locations(paths["result_locations"])
    else:
        mainmodel.schematisation.generate_result_locations(35, 70, 50)
        mainmodel.schematisation.result_locations.to_file(paths["result_locations_save"])
        mainmodel.project.settings["schematisation"]["result_locations_shape"] = paths["result_locations_save"]

    # Set support location
    mainmodel.schematisation.set_selected_support_location(supportlocationname)


def randomize_table(table):
    """
    Randomly sort a table
    """
    # Save old index
    index_names = table.index.names
    if len(index_names) > 1:
        for i, name in enumerate(index_names):
            table[f"before_index_{name}"] = table.index.get_level_values(i)
    else:
        table[f"before_index_{index_names[0]}"] = table.index.array

    # Randomize
    table.index = np.argsort(np.random.rand(len(table)))
    table.sort_index(inplace=True)

    # Restore index
    table.set_index([f"before_index_{name}" for name in index_names], inplace=True)
    table.index.names = index_names


def compare_dataframes(cls, df1, df2, repws=True, round=None, fillna=None):

    # Compare columns
    cols1, cols2 = sorted(df1.columns.tolist()), sorted(df2.columns.tolist())
    cls.assertListEqual(
        cols1, cols2, msg=f'DataFrame columns differ:\ndf1: {", ".join(cols1)}\nfd2: {", ".join(cols2)}'
    )

    # Replace white spaces in column names, since we're using itertuples
    if repws:
        for df in [df1, df2]:
            df.columns = [col.replace(" ", "_") for col in df.columns]

    # Check number of NaN's
    cls.assertEqual(df1.isnull().sum().sum(), df2.isnull().sum().sum())

    # Compare sizes
    cls.assertTupleEqual(df1.shape, df2.shape, msg=f"DataFrame sizes differ: {df1.shape} and {df2.shape}.")

    # Check dataframe content
    if fillna is not None:
        df1 = df1.fillna(fillna)
        df2 = df2.fillna(fillna)

    if round is not None:
        df1 = df1.round(round)
        df2 = df2.round(round)

    for row1, row2 in zip(df1.itertuples(), df2.itertuples()):
        msg = f"Found a difference at row: {row1.Index} (table 1) and {row2.Index} (table 2)\n\nRow 1: {row1}\n\nRow 2: {row2}"
        cls.assertTupleEqual(row1, row2, msg=msg)


def set_model_uncertainties(mainmodel):
    # Definieer modelonzekerheden
    # ========================================
    # Initialiseer, voeg uitvoerlocaties toe aan tabel
    mainmodel.modeluncertainties.add_result_locations()
    # Initialiseer, laad modelonzekerheden in uit database
    mainmodel.modeluncertainties.load_modeluncertainties()

    # Voeg de onzekerheden uit de steunpuntlocatie toe aan de tabel
    onzekerheden = mainmodel.modeluncertainties.supportloc_unc.unstack()
    onzekerheden.index = [" ".join(vals) for vals in onzekerheden.index.to_numpy()]
    mainmodel.modeluncertainties.table.loc[:, onzekerheden.index] = onzekerheden.values


def create_tmp_db_copies(paths, ext):

    dbkeys = ["hlcd", "hrd", "config"]

    tmp_paths = {}
    for key, path in paths.items():
        if key not in dbkeys:
            continue
        if key == "config":
            tmp_paths[key] = os.path.join(
                "temp", os.path.split(path)[-1].replace(".config.sqlite", f"_{ext}.config.sqlite")
            )
        else:
            tmp_paths[key] = os.path.join("temp", os.path.split(path)[-1].replace(".sqlite", f"_{ext}.sqlite"))

    for path in tmp_paths.values():
        if os.path.exists(path):
            os.remove(path)

    for key, path in paths.items():
        if key not in dbkeys:
            continue
        shutil.copy2(path, tmp_paths[key])

    return tmp_paths


def get_db_count_os(conn, parameter, locationids):

    hrdlocidsstr = ",".join([str(loc) for loc in locationids])

    if parameter == "Wind speed":
        windspeedid = conn.execute(
            "select HRDInputColumnId from HRDInputVariables where InputVariableId = 9"
        ).fetchall()[0][0]
        SQL = f"""
        select RD.HRDLocationId, ID.Value, Count(RD.HydraulicLoadId) from HydroDynamicResultData RD
        inner join HydroDynamicInputData ID on ID.HydraulicLoadId = RD.HydraulicLoadId
        where ID.HRDInputColumnId = {windspeedid} and RD.HRDLocationId IN ({hrdlocidsstr})
        group by RD.HRDLocationId, ID.Value"""

    elif parameter == "Wind direction":
        SQL = f"""
        select RD.HRDLocationId, WD.Direction as Value, Count(RD.HydraulicLoadId) from HydroDynamicData
        inner join HRDWindDirections WD on WD.HRDWindDirectionID = HydroDynamicData.HRDWindDirectionID
        inner join HydroDynamicResultData RD on HydroDynamicData.HydraulicLoadId = RD.HydraulicLoadId
        where HRDLocationId IN ({hrdlocidsstr})
        group by HRDLocationId, HydroDynamicData.HRDWindDirectionId"""

    counts = pd.read_sql(SQL, con=conn, index_col=["HRDLocationId", "Value"]).unstack()

    return counts


def get_db_count(conn, parameter, locationids):

    hrdlocidsstr = ",".join([str(loc) for loc in locationids])

    if parameter in inputvariableids:
        SQL = f"select HRDInputColumnId from HRDInputVariables where InputVariableId={inputvariableids[parameter]}"
        inputcolumnid = conn.execute(SQL).fetchone()
        if inputcolumnid is None:
            raise ValueError(parameter, inputvariableids[parameter])
        SQL = f"""
        select HRDLocationId, ID.Value, Count(HD.HydroDynamicDataId) from HydroDynamicData HD
        inner join HydroDynamicInputData ID on ID.HydroDynamicDataId = HD.HydroDynamicDataId
        where ID.HRDInputColumnId = {inputcolumnid[0]} and HRDLocationId IN ({hrdlocidsstr})
        group by HRDLocationId, ID.Value"""

    elif parameter == "Wind direction":
        SQL = f"""
        select HRDLocationId, Direction as Value, Count(HydroDynamicDataId) from HydroDynamicData
        inner join HRDWindDirections on HRDWindDirections.HRDWindDirectionID = HydroDynamicData.HRDWindDirectionID
        where HRDLocationId IN ({hrdlocidsstr})
        group by HRDLocationId, HydroDynamicData.HRDWindDirectionId"""

    else:
        raise NotImplementedError(parameter)
    # elif parameter == 'ClosingSituationId':
    #     SQL = f"""
    #     select HRDLocationId, Direction as Value, Count(HydroDynamicDataId) from HydroDynamicData
    #     inner join HRDWindDirections on HRDWindDirections.HRDWindDirectionID = HydroDynamicData.HRDWindDirectionID
    #     where HRDLocationId IN ({hrdlocidsstr})
    #     group by HRDLocationId, HydroDynamicData.HRDWindDirectionId"""

    counts = pd.read_sql(SQL, con=conn, index_col=["HRDLocationId", "Value"]).unstack()

    return counts


def copy_swan_files(datadir, swan_dst, step):
    """
    Copy SWAN files
    """

    if step.startswith("I"):
        # Get directories
        srcpath = f"{datadir}/swan/iterations/{step}/table"
        dstpath = f"{swan_dst}/iterations/{step}/table"

    elif step in ["D", "TR", "W"]:
        # Get directories
        srcpath = f"{datadir}/swan/calculations/{step}/spectra"
        dstpath = f"{swan_dst}/calculations/{step}/spectra"

    # Create destination directory if it does not exist
    if not os.path.exists(dstpath):
        os.makedirs(dstpath)

    # Copy testresults
    for file in os.listdir(srcpath):
        shutil.copy2(srcpath + "/" + file, dstpath + "/" + file)


def unzip_swan_files(datadir, swan_dst, step):
    """
    Unzip SWAN files
    """
    # Get directories
    srcpath = f"{datadir}/swan/calculations/{step}.zip"
    dstpath = f"{swan_dst}/calculations/{step}/spectra"

    # Create destination directory if it does not exist
    if not os.path.exists(dstpath):
        os.makedirs(dstpath)

    # Unzip testresults
    with zipfile.ZipFile(srcpath, "r") as zip_ref:
        zip_ref.extractall(dstpath)


def unzip_hares_files(datadir, hares_dst):
    """
    Unzip SWAN files
    """
    # Get directories
    srcpath = f"{datadir}/hares/results.zip"
    dstpath = f"{hares_dst}"

    # Create destination directory if it does not exist
    if not os.path.exists(dstpath):
        os.makedirs(dstpath)

    # Unzip testresults
    with zipfile.ZipFile(srcpath, "r") as zip_ref:
        zip_ref.extractall(dstpath)


def compare_or_save(cls, table_path, table, cols=None):

    if cols is None:
        cols = len(table.index.names)

    # Save if not exists
    if not os.path.exists(table_path):
        table.to_csv(table_path, sep=";", decimal=".")

    # Check if the results are equal to a test table
    test_table = pd.read_csv(table_path, sep=";", decimal=".", index_col=list(range(cols)))

    # Check the results
    compare_dataframes(cls, test_table.sort_index(), table.sort_index(), fillna=0.0, round=3)


def export_to_database(mainmodel, tmp_paths, export_hlcd_config=True):

    mainmodel.project.settings["export"]["export_HLCD_and_config"] = export_hlcd_config
    if export_hlcd_config:
        # Add HLCD
        mainmodel.export.add_HLCD(tmp_paths["hlcd"])
    # Initialiseer exportmodel
    mainmodel.export.add_result_locations()
    # Vul exportnamen in
    mainmodel.export.export_dataframe["Exportnaam"] = [
        f"export_{naam}" for naam in mainmodel.export.export_dataframe["Naam"].values
    ]
    # Vul database in
    mainmodel.export.export_dataframe.loc[:, "SQLite-database"] = tmp_paths["hrd"]
    # Exporteer
    mainmodel.export.export_output_to_database()


def get_counts(mainmodel, tmp_paths):

    conn = sqlite3.connect(tmp_paths["hrd"])
    names = '","'.join(mainmodel.export.export_dataframe["Exportnaam"].tolist())
    hrdlocationids = np.stack(
        conn.execute(f'SELECT HRDLocationId FROM HRDLocations WHERE Name IN ("{names}");').fetchall()
    ).squeeze()
    supportlocid = mainmodel.schematisation.support_location["HRDLocationId"]

    counts = {}
    # Test the number of entries per input column
    for col in mainmodel.hydraulic_loads.input_columns:
        # Get the counts for the column
        counts[col] = get_db_count(conn, col, [supportlocid] + hrdlocationids.tolist())

    conn.close()

    return counts, supportlocid
