# -*- coding: utf-8 -*-
"""
Created on  : Tue Jul 11 11:35:04 2017
Author      : Guus Rongen
Project     : PR3594.10.00
Description :
"""

import logging
import os
import sqlite3

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point

from hbhavens.core import datamodels

logger = logging.getLogger(__name__)


inputvariableids = {
    1  : "Discharge Lobith",
    2  : "Discharge Lith",
    3  : "Discharge Borgharen",
    4  : "Discharge Olst",
    5  : "Discharge Dalfsen",
    6  : "Water level Maasmond",
    7  : "Water level IJssel lake",
    8  : "Water level Marker lake",
    9  : "Wind speed",
    10 : "Water level",
    11 : "Wave period",
    12 : "Sea water level",
    13 : "Wave height",
    14 : "Sea water level (u)",
    15 : "Uncertainty water level (u)",
    16 : "Storm surge duration",
    17 : "Time shift surge and tide",
    18 : "Lake level VZM",
}

intputvarabr = {
    "Discharge Lobith": 'Q={:05.0f}',
    "Discharge Lith": 'Q={:05.0f}',
    "Discharge Borgharen": 'Q={:05.0f}',
    "Discharge Olst": 'Q={:05.0f}',
    "Discharge Dalfsen": 'Q={:05.0f}',
    "Water level Maasmond": 'M={:04.2f}',
    "Water level IJssel lake": 'M={:04.2f}',
    "Water level Marker lake": 'M={:04.2f}',
    "Wind speed": 'U={:02.0f}',
    "Water level": 'h={:03.1f}',
    "Sea water level": 'M={:04.2f}',
    "Storm surge duration": 'D={:03.0f}',
    "Time shift surge and tide": 'P={:03.0f}',
    "Wind direction": 'D={:05.1f}',
    "ClosingSituationId": 'K={:02d}',
    "Lake level VZM": 'M={:04.2f}',
}

resultvariableids = {
    1 : "h",
    2 : "Hs",
    3 : "Ts",
    4 : "Tp",
    5 : "Tpm",
    6 : "Tm-1,0",
    7 : "Wave direction",
    8 : "Storage situation VZM",
}

def add_to_table(tablename, dataframe, conn):

    # First check if table exists
    n = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?;", (tablename,)).fetchall()
    if not np.size(n):
        raise ValueError('Table "{}" does not exist'.format(tablename))

    # Secondly check if all columns are present in the table
    columns = [r[1] for r in conn.execute("PRAGMA TABLE_INFO('{}');".format(tablename)).fetchall()]
    try:
        dataframe = dataframe[columns]
    except ValueError as err:
        print('The input dataframe does not have the right columns:', err.args)

    # Voeg toe aan tabel
    dataframe.to_sql(tablename, conn, if_exists='append', chunksize=100000, index=False)

class ExportResultTable(datamodels.ExtendedDataFrame):

    _metadata = [
        'hydraulic_loads', 'result_columns', 'input_columns', 'sort_cols', 'settings'
    ]

    def __init__(self, mainmodel):
        super(ExportResultTable, self).__init__()
        
        # Links
        self.hydraulic_loads = mainmodel.hydraulic_loads
        self.settings = mainmodel.project.settings

        # Link to columns
        self.result_columns = mainmodel.hydraulic_loads.result_columns
        self.input_columns = mainmodel.hydraulic_loads.input_columns
        self.sort_cols = ['Location', 'HydraulicLoadId']

        # self.HRDLocations = pd.DataFrame(
        #     columns=['HRDLocationId', 'LocationTypeId', 'Name', 'XCoordinate', 'YCoordinate', 'WaterLevelCorrection']
        # )

        # self.UncertaintyModelFactor = pd.DataFrame(
        #     columns=['HRDLocationId', 'ClosingSituationId', 'HRDResultColumnId', 'Mean', 'Standarddeviation']
        # )

    def add_interpolated_results(self, results, column_mapping):
        """Method to interpolate results from recalculated water levels
        to original water levels, and add the tow the export dataframe class.
        
        Parameters
        ----------
        results : pandas.DataFrame
            DataFrame with results
        column_mapping : dict
            Dictionary with the translation form column name in dataframe to column name
            in output (database)
        """
        if not self.settings['hydraulic_loads']['recalculate_waterlevels']:
            raise ValueError('This function should only be used when waterlevels are recalculated.')
        
        # Check mapping
        if not set(list(column_mapping)).issuperset(set(self.result_columns)):
            raise KeyError(f'Did not get all required columns. Expected: {set(self.result_columns)}, got {set(list(column_mapping))}')

        # Interpolate wave conditions on water levels
        interp_results = self.hydraulic_loads.interpolate_wave_conditions(results, column_mapping)

        # Empty results
        self.delete_all()
        self.reindex_inplace(columns=interp_results.columns)
        self.set_data(interp_results)

        # Check sort columns presence
        for col in self.sort_cols:
            if col not in interp_results.columns.array:
                raise KeyError(f'Results are missing column "{col}"')

        # Check if there are NaNs in the resulttable
        if self.isnull().any().any():
            raise ValueError('NaN values in result table.')

    def add_results(self, results, column_mapping):
        """Method to add results from different calculation types to export dataframe.
        
        Parameters
        ----------
        results : pandas.DataFrame
            DataFrame with results
        column_mapping : dict
            Dictionary with the translation form column name in dataframe to column name
            in output (database)
        """

        # Recalculate water levels
        if self.settings['hydraulic_loads']['recalculate_waterlevels']:
            raise ValueError('This function should not be used when waterlevels are recalculated.')

        # Empty results
        self.delete_all()
        self.reindex_inplace(columns=self.sort_cols + self.input_columns + self.result_columns)

        # Check mapping
        if not set(list(column_mapping)).issuperset(set(self.result_columns)):
            raise KeyError(f'Did not get all required columns. Expected: {set(self.result_columns)}, got {set(list(column_mapping))}')

        for col in self.sort_cols:
            if col not in results.columns.array:
                raise KeyError(f'Results are missing column "{col}"')
        
        # Determine what columns in the table match with the required columns
        tablecols = [column_mapping[col] for col in self.result_columns]
        # Add Location and Load columns for sorting
        tablecols += self.sort_cols

        # Check if the needed columns are present in the table (check for nan on reindex)
        result_selection = results.reindex(columns=tablecols)
        if result_selection.isnull().any().any():
            columns = ', '.join(result_selection.columns.to_numpy()[result_selection.isnull().any(axis=0)].astype(str).tolist())
            index = result_selection.isnull().any(axis=1).sum()
            raise ValueError(f'NaN values in results. Columns: {columns}, Index: {index}/{len(result_selection)}')

        result_selection = result_selection.sort_values(
            by=self.sort_cols).rename(columns={v: k for k, v in column_mapping.items()})
        
        self[self.sort_cols + self.result_columns] = result_selection[self.sort_cols + self.result_columns].reset_index(drop=True)

        # 2. Add hydraulic loads
        # Only if not recalculated water levels. If recalculated water levels, the original hydraulic loads are used
        # and these are already present in the result from the function 'interpolate_wave_conditions'
        if not self.settings['hydraulic_loads']['recalculate_waterlevels']:
            nlocations = len(self['Location'].unique())
            self.loc[:, self.input_columns] = np.tile(self.hydraulic_loads.sort_index()[self.input_columns].values, (nlocations, 1))

        # Check if there are NaNs in the resulttable
        if self.isnull().any().any():
            raise ValueError('NaN values in result table.')

    def set_hrdlocationid(self, tab):
        """Add HRDLocationId
        
        Parameters
        ----------
        tab : pandas.DataFrame
            DataFrame with 'Naam' and 'HRDLocationId' columns
        """
        
        dct = {row.Naam: row.HRDLocationId for row in tab.itertuples()}
        self['HRDLocationId'] = [dct[row.Location] for row in self.itertuples()]

class HRDio:

    def __init__(self, path):

        self.path = path
        if not os.path.exists(path):
            raise OSError(f'Path "{path}" does not exist.')

        self.conn = None

        # Get database format
        self._connect()
        columns = [col[1] for col in self.conn.execute('PRAGMA table_info(HydroDynamicData);').fetchall()]
        self.dbformat = 'OS2023' if 'HydraulicLoadId' in columns else 'WBI2017'
        self._close()
        
    def _connect(self):
        self.conn = sqlite3.connect(self.path)

    def _close(self):
        self.conn.close()

    def remove_locations(self, polygon, exemption=''):
        """
        Remove locations and corresponding hydro dynamic data from db

        Parameters
        ----------
        polygon : shapely.geometry.Polygon
            Polygon within which the locations are removed
        
        Returns
        -------
        remove_ids : list
            List with the HRDLocationIds of the removed locations
        """

        if isinstance(exemption, str):
            exemption = [exemption]

        # Get locations
        self._connect()
        remove_ids = []
        sql = f'SELECT HRDLocationId, Name, XCoordinate, YCoordinate FROM HRDLocations;'
        locations = self.conn.execute(sql)
        for locid, name, x, y in locations:
            if polygon.intersects(Point(x, y)):
                if name in exemption:
                    logger.warning('The support location is in the harbor bounds, but will not be removed.')
                    continue
                remove_ids.append(locid)
        remove_id_str = ','.join(map(str, remove_ids))
        
        if self.dbformat == 'WBI2017':
            # Collect HydroDynamicDataIds
            sql = f'SELECT HydroDynamicDataId FROM HydroDynamicData WHERE HRDLocationId IN ({remove_id_str});'
            data_ids = self.conn.execute(sql).fetchall()
            data_id_str = ','.join(str(i[0]) for i in data_ids)

            # Delete from HRDLocations and UncertaintyModelFactor
            for table in ['HRDLocations', 'UncertaintyModelFactor']:
                self.conn.execute(f'DELETE FROM {table} WHERE HRDLocationId IN ({remove_id_str});')

            # Delete from HydroDynamicData, HydroDynamicInputData and HydroDynamicResultData
            for table in ['HydroDynamicData', 'HydroDynamicInputData', 'HydroDynamicResultData']:
                self.conn.execute(f'DELETE FROM {table} WHERE HydroDynamicDataId IN ({data_id_str});')

        if self.dbformat == 'OS2023':
            # Delete from HRDLocations, UncertaintyModelFactor and HydroDynamicResultData
            for table in ['HRDLocations', 'UncertaintyModelFactor', 'HydroDynamicResultData']:
                self.conn.execute(f'DELETE FROM {table} WHERE HRDLocationId IN ({remove_id_str});')

        self.conn.commit()
        self._close()

        return remove_ids

    def get_track_id(self):
        self._connect()
        systemid = self.conn.execute('SELECT TrackID FROM General;').fetchone()[0]
        self._close()
        return systemid

    def get_system_id(self):
        self._connect()
        systemid = self.conn.execute('SELECT GeneralId FROM General;').fetchone()[0]
        self._close()
        return systemid

    def get_type_of_hydraulic_load_id(self):
        """
        Get type of hydraulic data
        """
        self._connect()
        resultvariables = np.hstack(self.conn.execute('SELECT ResultVariableId FROM HRDResultVariables;').fetchall())
        self._close()

        # TypeOfHydraulicData Water level or Wave data
        if (1 in resultvariables) and (2 in resultvariables):
            # Wave and Water level
            TypeOfHydraulicDataId = 2
        elif (1 in resultvariables):
            # Water level
            TypeOfHydraulicDataId = 1
        else:
            # Wave data
            TypeOfHydraulicDataId = 3

        return TypeOfHydraulicDataId

    def get_max_hrdlocation_id(self):
        self._connect()
        maxhrdlocid = self.conn.execute('SELECT MAX(HRDLocationId) FROM HRDLocations;').fetchone()[0]
        self._close()
        return maxhrdlocid

    def add_hrd_locations(self, locations):
        """
        Function to prepare a HRDLocations table for export, from a geopandas
        GeoDataFrame

        Parameters
        ----------
        locations : geopandas.GeoDataFrame
            GeoDataFrame with data to export
        """

        self._connect()
        # Create empty dataframe with columns from db table
        columns = [col[1] for col in self.conn.execute('PRAGMA table_info(HRDLocations);').fetchall()]
        HRDLocations = pd.DataFrame(columns=columns)

        # Fill dataframe
        HRDLocations[['Name', 'HRDLocationId']] = locations[['Exportnaam', 'HRDLocationId']]
        HRDLocations[['XCoordinate', 'YCoordinate']] = [list(pt.coords[0]) for pt in locations['geometry']]
        HRDLocations[['XCoordinate', 'YCoordinate']] = HRDLocations[['XCoordinate', 'YCoordinate']].round(3)
        HRDLocations['WaterLevelCorrection'] = 0.0
        HRDLocations['LocationTypeId'] = 2

        if 'Acceptatie geometrie' in columns:
            HRDLocations['Acceptatie geometrie'] = 1

        # Add to database
        add_to_table('HRDLocations', HRDLocations, self.conn)

        self.conn.commit()
        self._close()
        
    def add_hydro_dynamic_data(self, resultdata, supportlocid=None):
        """
        Export HydroDynamicData to database

        Parameters
        ----------
        locations : geopandas.GeoDataFrame
            GeoDataFrame with data to export
        resultdata : pandas.DataFrame
            Resultdata to be transformed and exported
        """

        self._connect()
        if self.dbformat == 'WBI2017':
            # Add max value to HydroDynamicDataId
            max_hddid = self.conn.execute('SELECT MAX(HydroDynamicDataId) FROM HydroDynamicData;').fetchone()[0]
            resultdata.loc[:, 'HydroDynamicDataId'] = np.arange(len(resultdata)) + 1 + max_hddid

            # HydroDynamicData
            #-------------------------------------------------------------
            HydroDynamicData = resultdata.reindex(columns=['HydroDynamicDataId', 'ClosingSituationId', 'HRDLocationId', 'Wind direction'])
            # Transform wind directions
            winddirection_conv = {k: v for k, v in self.conn.execute('SELECT Direction, HRDWindDirectionId FROM HRDWindDirections').fetchall()}
            HydroDynamicData.loc[:, 'HRDWindDirectionId'] = [winddirection_conv[wd] for wd in HydroDynamicData['Wind direction'].array]
            # If ClosingSituationId not present in resultdata, set to 1.
            if pd.isnull(HydroDynamicData['ClosingSituationId']).all():
                HydroDynamicData.loc[:, 'ClosingSituationId'] = 1
            # Add to database
            add_to_table('HydroDynamicData', HydroDynamicData, self.conn)

            # HydroDynamicInputData
            #-------------------------------------------------------------
            HRDInputVariables = pd.read_sql('SELECT * FROM HRDInputVariables;', self.conn)
            HydroDynamicInputData = resultdata[['HydroDynamicDataId'] + [inputvariableids[ivid] for ivid in HRDInputVariables['InputVariableId'].array]]
            HydroDynamicInputData.columns = ['HydroDynamicDataId'] + HRDInputVariables['HRDInputColumnId'].tolist()
            HydroDynamicInputData.set_index('HydroDynamicDataId', inplace=True)
            HydroDynamicInputData = pd.DataFrame(HydroDynamicInputData.stack()).reset_index()
            HydroDynamicInputData.columns = ['HydroDynamicDataId', 'HRDInputColumnId', 'Value']
            # Add to database
            add_to_table('HydroDynamicInputData', HydroDynamicInputData, self.conn)

            # HydroDynamicResultData
            #-------------------------------------------------------------
            HRDResultVariables = pd.read_sql('SELECT * FROM HRDResultVariables;', self.conn)
            # Selecteer result data met algemene kolombenaming
            resultcolumns = ['HydroDynamicDataId'] + [resultvariableids[rvid] for rvid in HRDResultVariables['ResultVariableId'].array]
            #TODO: Make nice division bewteen zoet and zout
            if 'h' in resultcolumns and 'h' not in resultdata.columns:
                resultdata.rename(columns={'Water level': 'h'}, inplace=True)
            HydroDynamicResultData = resultdata[resultcolumns]

            # Converteer kolombenaming naar HRDResultColumnId
            HydroDynamicResultData.columns = ['HydroDynamicDataId'] + HRDResultVariables['HRDResultColumnId'].tolist()
            HydroDynamicResultData.set_index('HydroDynamicDataId', inplace=True)
            HydroDynamicResultData = pd.DataFrame(HydroDynamicResultData.stack()).reset_index()
            HydroDynamicResultData.columns = ['HydroDynamicDataId', 'HRDResultColumnId', 'Value']
            # Add to database
            add_to_table('HydroDynamicResultData', HydroDynamicResultData, self.conn)

        elif self.dbformat == 'OS2023':
            # For OS2023 only the HydroDynamicResultData table needs to be filled,
            # since the HydroDynamicData and HydroDynamicInputData are location independend
            HydroDynamicResultData = resultdata.set_index(['HRDLocationId', 'HydraulicLoadId'])
            
            # Convert resultvariables to id's
            result_var_conv = {v: k for k, v in self.conn.execute('SELECT HRDResultColumnId, ColumnName FROM HRDResultVariables;').fetchall() if v != 'ZWL'}
            HydroDynamicResultData = HydroDynamicResultData.loc[:, list(result_var_conv)].copy()
            HydroDynamicResultData.columns = [result_var_conv[var] for var in HydroDynamicResultData.columns]

            # Stack, add index to columns and rename columns
            HydroDynamicResultData = HydroDynamicResultData.stack().reset_index()
            HydroDynamicResultData.columns = ['HRDLocationId', 'HydraulicLoadId', 'HRDResultColumnId', 'Value']
            
            # Add to database
            add_to_table('HydroDynamicResultData', HydroDynamicResultData, self.conn)

            # Now the water levels (ZWL) have to be copied to the result locations
            zwl = pd.read_sql(f'SELECT HydraulicLoadId, HRDResultColumnId, Value FROM HydroDynamicResultData WHERE HRDLocationId={supportlocid} AND HRDResultColumnId=1;', con=self.conn)
            hrdlocationids = HydroDynamicResultData['HRDLocationId'].unique()
            zwl = pd.concat([zwl.assign(HRDLocationId=hrdid) for hrdid in hrdlocationids], ignore_index=True)
            add_to_table('HydroDynamicResultData', zwl, self.conn)


        else:
            raise ValueError('Format "{}" not understood.'.format(self.dbformat))

        self.conn.commit()
        self._close()

    def add_uncertainty_model_factor(self, locations, uncertainties):
        """
        Export UncertaintyModelFactor to database

        Parameters
        ----------
        locations : geopandas.GeoDataFrame
            GeoDataFrame with data to export
        uncertainties : pandas.DataFrame
            Uncertainty data to be transformed and exported
        conn : sqlite3.connection
            Connection to HRD
        """
        self._connect()
        # Select required locations from uncertainty data
        uncertainties = uncertainties.loc[locations['Naam'], :]
        # Replace name with HRDLocationId
        uncertainties.index = [locations.set_index('Naam').loc[name, 'HRDLocationId'] for name in uncertainties.index]
        # Convert column names
        uncertainties.columns = pd.MultiIndex.from_tuples([tuple(col.replace('mu', 'Mean').replace('sigma', 'Standarddeviation').split()) for col in uncertainties.columns])
        uncertainties = uncertainties.stack()
        # get closing situations id
        csids = np.hstack(self.conn.execute('SELECT DISTINCT(ClosingSituationId) FROM ClosingSituations;').fetchall()).tolist()
        # Stack uncertainties dataframe for each ClosingSituationId
        if len(csids) == 1:
            UncertaintyModelFactor = uncertainties.reset_index()
            UncertaintyModelFactor['ClosingSituationId'] = csids[0]
        else:
            UncertaintyModelFactor = pd.concat([uncertainties]*len(csids)).reset_index()
            UncertaintyModelFactor['ClosingSituationId'] = np.repeat(csids, len(uncertainties))
        UncertaintyModelFactor.columns = ['HRDLocationId', 'HRDResultColumnId'] + UncertaintyModelFactor.columns[2:].tolist()
        # Replace HLCD names with HLCD id (HRDResultColumnId now contains the ResultVariableIds)
        UncertaintyModelFactor['HRDResultColumnId'].replace({v: k for k, v in resultvariableids.items()}, inplace=True)
        # Replace result variable ids with column ids
        HRDResultVariables = pd.read_sql('SELECT * FROM HRDResultVariables;', self.conn)
        # Create a dictionary to convert Result Variable Id to Result Column Id
        hlcdid_hrdcol_dict = HRDResultVariables.set_index('ResultVariableId')['HRDResultColumnId'].to_dict()
        # Select only the values in the table that are present in the HRDResultVariables keys.
        # Note that this is only needed when different database set-ups are combines in HB Havens
        UncertaintyModelFactor = UncertaintyModelFactor.loc[np.in1d(UncertaintyModelFactor['HRDResultColumnId'], list(hlcdid_hrdcol_dict.keys()))]
        # Replace (HRDResultColumnId now contains the HRDResultColumnId)
        UncertaintyModelFactor['HRDResultColumnId'].replace(hlcdid_hrdcol_dict, inplace=True)
        # Sort values
        UncertaintyModelFactor.sort_values(by=['HRDLocationId', 'ClosingSituationId', 'HRDResultColumnId'], inplace=True)
        # Add to database
        add_to_table('UncertaintyModelFactor', UncertaintyModelFactor, self.conn)
        self.conn.commit()
        self._close()

    def read_HydroDynamicData(self, hrdlocationid):
        """
        Reads hydro dynamic data from database.
        Depending on the format of the database, this function uses for the
        function "read_HydroDynamicData_2017" for WBI2017 databases, and the
        function "read_HydroDynamicData_2023" for WBI2023 (pilot) databases. It
        thus first determines with which type we are dealing.

        Parameters
        ----------
        conn : sqlite3.connection
            Open connection to database
        hrdlocationid : integer
            integer with the locationid of the location
        """

        self._connect()

        if self.dbformat == 'OS2023':
            data = self.read_HydroDynamicData_2023(hrdlocationid)
        elif self.dbformat == 'WBI2017':
            data = self.read_HydroDynamicData_2017(hrdlocationid)
        else:
            raise ValueError(f'Database "{self.dbformat}" format not known.')

        self._close()

        return data


    def read_HydroDynamicData_2017(self, hrdlocationid):
        """
        Reads hydro dynamic data for a location (input and results)
        """

        if not isinstance(hrdlocationid, (int, np.integer)):
            raise TypeError('must be int, not {}'.format(type(hrdlocationid)))

        # First collect the dataids. Also replace wind direction ids with real ids
        SQL = """
        SELECT D.HydroDynamicDataId, D.ClosingSituationId, W.Direction AS "Wind direction"
        FROM
        HydroDynamicData D INNER JOIN HRDWindDirections W ON D.HRDWindDirectionId=W.HRDWindDirectionId
        WHERE HRDLocationId = {};""".format(int(hrdlocationid))

        dataids = pd.read_sql(SQL, self.conn, index_col='HydroDynamicDataId')
        dataidsstr = ','.join(dataids.index.values.astype(str).tolist())

        # Collect the result data. Replace HRDResultColumnId with variable id's
        SQL = """
        SELECT RD.HydroDynamicDataId, RV.ResultVariableId, RD.Value
        FROM
        HydroDynamicResultData RD INNER JOIN HRDResultVariables RV ON RD.HRDResultColumnId = RV.HRDResultColumnId
        WHERE HydroDynamicDataId IN ({});""".format(dataidsstr)

        resultdata = pd.read_sql(SQL, self.conn, index_col=['HydroDynamicDataId', 'ResultVariableId']).unstack()
        # Reduce columnindex to single level index (without 'Value')
        resultdata.columns = [resultvariableids[rid] for rid in resultdata.columns.get_level_values(1)]
        
        # Collect inputdata in a similar way
        SQL = """
        SELECT ID.HydroDynamicDataId, IV.InputVariableId, ID.Value
        FROM
        HydroDynamicInputData ID INNER JOIN HRDInputVariables IV ON ID.HRDInputColumnId = IV.HRDInputColumnId
        WHERE HydroDynamicDataId IN ({});""".format(dataidsstr)

        inputdata = pd.read_sql(SQL, self.conn, index_col=['HydroDynamicDataId', 'InputVariableId']).unstack()
        # Reduce columnindex to single level index (without 'Value')
        inputdata.columns = [inputvariableids[iid] for iid in inputdata.columns.get_level_values(1)]

        # Join data and sort values
        data = dataids.join(inputdata).join(resultdata)
        
        # Sort
        data.sort_values(by=dataids.columns.tolist() + inputdata.columns.tolist(), inplace=True)
        logger.info(f'Loaded hydraulic loads in WBI2017 format from {self.path}.')

        return data

    def read_HydroDynamicData_2023(self, hrdlocationid):
        """
        Reads hydro dynamic data for a location (input and results) from a
        database with the WBI2023 format.
        """

        if not isinstance(hrdlocationid, (int, np.integer)):
            raise TypeError('must be int, not {}'.format(type(hrdlocationid)))

        # First collect the dataids. Also replace wind direction ids with real ids
        SQL = """
        SELECT D.HydraulicLoadId, D.ClosingSituationId, W.Direction AS "Wind direction"
        FROM
        HydroDynamicData D INNER JOIN HRDWindDirections W ON D.HRDWindDirectionId=W.HRDWindDirectionId;"""


        dataids = pd.read_sql(SQL, self.conn, index_col='HydraulicLoadId')
        dataidsstr = ','.join(dataids.index.drop_duplicates().values.astype(str).tolist())

        # Collect the result data. Replace HRDResultColumnId with variable id's
        SQL = """
        SELECT RD.HydraulicLoadId, RV.ResultVariableId, RD.Value
        FROM
        HydroDynamicResultData RD INNER JOIN HRDResultVariables RV ON RD.HRDResultColumnId = RV.HRDResultColumnId
        WHERE HydraulicLoadId IN ({}) AND HRDLocationId = {};""".format(dataidsstr, hrdlocationid)

        resultdata = pd.read_sql(SQL, self.conn, index_col=['HydraulicLoadId', 'ResultVariableId']).unstack()
        # Reduce columnindex to single level index (without 'Value')
        resultdata.columns = [resultvariableids[rid] for rid in resultdata.columns.get_level_values(1)]

        # Collect inputdata in a similar way
        SQL = """
        SELECT ID.HydraulicLoadId, IV.InputVariableId, ID.Value
        FROM
        HydroDynamicInputData ID INNER JOIN HRDInputVariables IV ON ID.HRDInputColumnId = IV.HRDInputColumnId
        WHERE HydraulicLoadId IN ({});""".format(dataidsstr)

        inputdata = pd.read_sql(SQL, self.conn, index_col=['HydraulicLoadId', 'InputVariableId']).unstack()

        # Reduce columnindex to single level index (without 'Value')
        inputdata.columns = [inputvariableids[iid] for iid in inputdata.columns.get_level_values(1)]

        # Join data and sort values
        data = dataids.join(inputdata).join(resultdata).sort_values(by=['Wind direction', 'Wind speed', 'Water level'])

        # In the WBI2023 the water levels and waves are in the same table, but have different input variables
        # this gives empty columns in the loaded output, delete these.
        data.dropna(how='all', axis=1, inplace=True)
        
        # Drop entries without wave parameters
        data = data.loc[~np.isnan(resultdata['Hs'])]

        # Set the WBI2017 index name (HydroDynamicDataId) for consistency
        data.index.name = 'HydroDynamicDataId'
        logger.info(f'Loaded hydraulic loads in OS2023 format from {self.path}.')

        return data

        
    def read_HRDLocations(self):
        """
        Reads locations from HRD and converts to geopandas.GeoDataFrame
        """
        self._connect()

        # Retrieve locations
        loctable = pd.read_sql('SELECT HRDLocationId, XCoordinate, YCoordinate, Name FROM HRDLocations;', self.conn)
        # Create Point geometries
        ptgeometries = [Point(row.XCoordinate, row.YCoordinate) for row in loctable.itertuples()]
        # Construct new table
        locations = gpd.GeoDataFrame(loctable[['HRDLocationId', 'Name', 'XCoordinate', 'YCoordinate']], geometry=ptgeometries)
        locations.index = locations['HRDLocationId']

        self._close()

        return locations

    def read_UncertaintyModelFactor(self, hrdlocationid):
        """
        Reads model uncertainties for a location from a database

        Parameters
        ----------
        hrdlocationid : int
            Location id of the location where the uncertainties are exported from
        """

        if not isinstance(hrdlocationid, (int, np.integer)):
            raise TypeError('must be int, not {}'.format(type(hrdlocationid)))

        # First collect the dataids. Also replace wind direction ids with real ids
        SQL = """
        SELECT
        U.ClosingSituationId, RV.ResultVariableId, U.Mean AS mu, U.Standarddeviation AS sigma
        FROM
        UncertaintyModelFactor U
        INNER JOIN
        HRDResultVariables RV
        ON
        U.HRDResultColumnId = RV.HRDResultColumnId
        WHERE
        U.HRDLocationId = {};""".format(hrdlocationid)

        # Read from database
        self._connect()
        modeluncertainty = pd.read_sql(SQL, self.conn, index_col=['ClosingSituationId', 'ResultVariableId'])

        # It is possible that the uncertainties vary per closing situation id
        # At the moment the maximum values are used.
        modeluncertainty = modeluncertainty.groupby(level=1).max()

        # Replace HRD result column ids
        modeluncertainty.index = [resultvariableids[iid] for iid in modeluncertainty.index.array]

        self._close()

        return modeluncertainty

    
    def check_element_presence(self, elements, column, table):
        """
        Count the number of occurences of an element in a table
        """
        if not isinstance(elements, list):
            raise TypeError('Unexpected type. Expected elements to be list.')
        
        self._connect()
        elementstr = ','.join(map(str, elements))
        sql = f'SELECT COUNT(*) FROM {table} WHERE {column} IN ({elementstr})'
        count = self.conn.execute(sql).fetchone()[0]
        self._close()

        return count



class HLCDio:

    def __init__(self, path):

        self.path = path
        if not os.path.exists(path):
            raise OSError(f'Path "{path}" does not exist.')

        self.conn = None
        
    def _connect(self):
        self.conn = sqlite3.connect(self.path)

    def _close(self):
        self.conn.close()

    def remove_locations(self, remove_ids):
        """
        Remove locations from HLCD

        Parameters
        ----------
        remove_ids : list
            List with LocationIds to be removes from database, together with
            corresponding data.
        """
        self._connect()
        
        # Get LocationIds to remove
        remove_id_str = ",".join(map(str, remove_ids))
        sql = f'SELECT LocationId FROM Locations WHERE HRDLocationId IN ({remove_id_str});'
        remove_hlcd_ids = np.hstack(self.conn.execute(sql).fetchall())
        if not any(remove_hlcd_ids):
            self.conn.close()
            return None

        remove_id_str = ','.join(map(str, remove_hlcd_ids))
        sql = f'DELETE FROM Locations WHERE LocationId IN ({remove_id_str});'
        self.conn.execute(sql)
        self.conn.commit()
        self._close()

        return remove_hlcd_ids

    def get_max_hrdlocation_id(self, systemid=None, trackid=None):
        """
        Function to prepare a HRDLocations table for export, from a geopandas
        GeoDataFrame

        Parameters
        ----------
        systemid : int
            Id of the water system
        trackid : int
            Id of track
        """
        self._connect()

        if systemid is not None:
            # Determine the maximum HRDLocationId in the system
            maxlocid = self.conn.execute("""
            SELECT MAX(HRDLocationId) FROM Locations WHERE (LocationId > ?) AND (LocationId < ?);
            """, (systemid*100000, (systemid+1)*100000)).fetchone()[0]

            if maxlocid is not None:
                descriptive_id = False
                self._close()
                return maxlocid, descriptive_id

        if trackid is not None:
            # If not max location ID is found, try looking for the maximum given the trackid
            maxlocid = self.conn.execute('SELECT MAX(HRDLocationId) FROM Locations WHERE TrackId=?;', (trackid,)).fetchone()[0]
            if maxlocid is not None:
                descriptive_id = True
                self._close()
                return maxlocid, descriptive_id

        # Else, just find the highest location id in the database
        maxlocid = self.conn.execute('SELECT MAX(HRDLocationId) FROM Locations;').fetchone()[0]
        descriptive_id = False
        self._close()
        
        return maxlocid, descriptive_id

    def get_max_location_id(self):
        """
        Function to prepare a HRDLocations table for export, from a geopandas
        GeoDataFrame
        """
        self._connect()
        maxlocid = self.conn.execute('SELECT MAX(LocationId) FROM Locations;').fetchone()[0]
        self._close()
        
        return maxlocid

    def check_element_presence(self, table, column, elements):
        """
        Count the number of occurences of an element in a table
        """
        if not isinstance(elements, list):
            raise TypeError('Unexpected type. Expected elements to be list.')
        
        self._connect()
        elementstr = ','.join(map(str, elements))
        sql = f'SELECT COUNT(*) FROM {table} WHERE {column} IN ({elementstr})'
        count = self.conn.execute(sql).fetchone()[0]
        self._close()

        return count

    def add_locations(self, result_locations):
        """
        Modify the HLCD by removing old locations within the harbor and adding new
        locations. The HRD in needed for this function since it is used to determine
        the region.

        Parameters
        ----------
        result_locations : geopandas.GeoDataFrame
            GeoDataFrame with result locations
        conn : sqlite3.connection
            Connection to HLCD
        resultvariables : list
            List of result variables. Used for determining TypeOfHydraulicData
        """
        
        result_loc_cols = ['LocationId', 'TypeOfHydraulicDataId', 'TrackId', 'HRDLocationId', 'InterpolationSupportId']
        if not set(result_locations.columns.tolist()).issuperset(set(result_loc_cols)):
            raise KeyError('Not all columns present to fill Location table.')
        
        Locations = pd.DataFrame(columns=[
            'LocationId',
            'TypeOfHydraulicDataId',
            'TrackId',
            'HRDLocationId',
            'AreaNumber',
            'InterpolationSupportId',
            'ImplicInterpolationSupportId',
            'ImplicPerformanceLevelSupportId'
        ])

        # Location ids
        Locations[result_loc_cols] = result_locations[result_loc_cols]

        # add to database
        self._connect()
        add_to_table('Locations', Locations, self.conn)
        self.conn.commit()
        self._close()

    def get_interpolation_support_id(self, systemid, supportlocid):

        self._connect()

        # find InterpolationSupportId in HLCD
        sql = f'SELECT InterpolationSupportId FROM Locations WHERE LocationId={systemid:1d}{supportlocid:05d};'
        interpolation_support_id = self.conn.execute(sql).fetchone()
        
        if interpolation_support_id is not None:
            interpolation_support_id = interpolation_support_id[0]

        self._close()

        return interpolation_support_id
        
       

class Configio:

    def __init__(self, path):

        self.path = path
        if not os.path.exists(path):
            raise OSError(f'Path "{path}" does not exist.')

        self.conn = None
        
    def _connect(self):
        self.conn = sqlite3.connect(self.path)

    def _close(self):
        self.conn.close()

    def remove_locations(self, remove_ids):
        """
        Remove locations with corresponding calculation settings from config.

        Parameters
        ----------
        remove_ids : list
            List with LocationIds to be removes from database, together with
            corresponding data.
        """

        remove_ids = ','.join(map(str, remove_ids))
        self._connect()
        self.conn.execute(f'DELETE FROM NumericsSettings WHERE LocationId IN ({remove_ids});')
        self.conn.execute(f'DELETE FROM TimeIntegrationSettings WHERE LocationID IN ({remove_ids});')
        self.conn.commit()
        self._close()
        
    def add_numerical_settings(self, locations):
        """
        Add settings for numerics and time integration to config databasepath

        time integration
            calculationSchemeFBC    = 1
            calculationSchemeAPT    = 2
        --> calculationSchemeNTI    = 3
        numerics
            methodFORM                                    =  1
            methodCrudeMonteCarlo                         =  3
            methodDirectionalSampling                     =  4
            methodNumericalIntegration                    =  5
            methodImportanceSampling                      =  6
            methodFORMandDirSampling                      = 11
        --> methodDirSamplingWithFORMiterations           = 12
            methodCrudeMonteCarloWithFORMiterations       = 13
            methodImportanceSamplingWithFORMiterations    = 14

        Parameters
        ----------
        locations : geopandas.GeoDataFrame
            GeoDataFrame with data to export
        conn : sqlite3.connection
            Connection to config
        """

        self._connect()

        # Retrieve calculation settings for one location from database
        locationid = self.conn.execute('SELECT LocationId FROM NumericsSettings LIMIT 1;').fetchall()[0][0]
        NumericsSettings = pd.read_sql('SELECT * FROM NumericsSettings WHERE LocationId = {};'.format(locationid), self.conn)
        TimeIntegrationSettings = pd.read_sql('SELECT * FROM TimeIntegrationSettings WHERE LocationID = {};'.format(locationid), self.conn)

        # Set to heavy settings (methodDirSamplingWithFORMiterations and calculationSchemeNTI)
        NumericsSettings['CalculationMethod'] = 12
        TimeIntegrationSettings['TimeIntegrationSchemeID'] = 3

        # Stack for the number of locations to add
        length = len(NumericsSettings)
        NumericsSettings = pd.concat([NumericsSettings]*len(locations))
        NumericsSettings['LocationId'] = np.repeat(locations['LocationId'].squeeze().tolist(), length)
        add_to_table('NumericsSettings', NumericsSettings, self.conn)

        length = len(TimeIntegrationSettings)
        TimeIntegrationSettings = pd.concat([TimeIntegrationSettings]*len(locations))
        TimeIntegrationSettings['LocationID'] = np.repeat(locations['LocationId'].squeeze().tolist(), length)
        add_to_table('TimeIntegrationSettings', TimeIntegrationSettings, self.conn)

        self.conn.commit()

        self._close()
