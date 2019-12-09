import geopandas as gpd
import pandas as pd
import numpy as np
import re

def as_list(arg):
    if arg is None:
        return []
    elif isinstance(arg, (list, tuple)):
        return arg
    else:
        return [arg]

class ExtendedGeoDataFrame(gpd.GeoDataFrame):

    # normal properties
    _metadata = gpd.GeoDataFrame._metadata + ['required_columns', 'required_types', 'geotype']

    def __init__(self, geotype, required_columns=None, required_types=None, **kwargs):
        # Initialize with required column
        if 'columns' not in kwargs.keys():
            kwargs['columns'] = as_list(required_columns)
        else:
            kwargs['columns'].extend(as_list(required_columns))
        
        super(ExtendedGeoDataFrame, self).__init__(**kwargs)
        
        # Set required columns
        self.required_columns = as_list(required_columns)
        
        # Set required types        
        self.required_types = as_list(required_types)
        # Get geotype
        self.geotype = geotype

    def delete_all(self):
        """
        Empty the dataframe
        """
        if not self.empty:
            self.iloc[:, 0] = np.nan
            self.dropna(inplace=True)
    
    def read_file(self, path):
        """
        Import function, extended with type checks. Does not destroy reference to object.
        """

        self.delete_all()
        
        # Read file
        gdf = gpd.read_file(path)

        # Check number of entries
        if len(gdf) == 0:
            raise IOError('Imported shapefile contains no rows.')

        self.set_data(gdf)

    def set_data(self, gdf):

        # Copy content
        for col, values in gdf.iteritems():
            self[col] = values.values
        self.index = gdf.index
        self.index.name = gdf.index.name

        # Check columns and types
        self._check_columns()
        self._check_types()

        # Check geometry types
        self._check_geotype()

    def _check_columns(self):
        """
        Check presence of columns in geodataframe
        """
        for column in self.required_columns:
            if column not in self.columns.array:
                raise KeyError('Column "{}" not found. Expected at least {}'.format(column, ', '.join(self.required_columns)))

    def _check_types(self):
        """
        Check column types in dataframe
        """
        for (column, column_type) in zip(self.required_columns, self.required_types):
            error_msg = 'Column "{}" has datatype "{}". Expected "{}".'.format(column, type(column), column_type)
            
            if not isinstance(column_type, tuple):
                column_type = tuple([column_type])

            # Convert types to string
            if column_type[0] in [np.integer, np.floating]:
                valtype = np.dtype(type(self[column].values[0]))
                if not any(np.issubdtype(valtype, coltype) for coltype in column_type):
                    raise TypeError(error_msg)

            else:
                if not isinstance(self[column][0], column_type):
                    raise TypeError(error_msg)
    
    def _check_geotype(self):
        """
        Check geometry type
        """
        first_geo = self['geometry'].iloc[0]
        if not isinstance(first_geo, self.geotype):
            raise TypeError('Geometrietype "{}" vereist. De ingevoerde shapefile heeft geometrietype "{}".'.format(
                re.findall('([A-Z].*)\'', repr(self.geotype))[0],
                re.findall('([A-Z].*)\'', repr(type(first_geo)))[0],
            ))


class ExtendedDataFrame(pd.DataFrame):

    def __init__(self, *args, **kwargs):
        super(ExtendedDataFrame, self).__init__(*args, **kwargs)

    def delete_all(self):
        """
        Empty the dataframe keeping the dataframe id
        """
        if not self.empty:
            self.iloc[:, 0] = np.nan
            self.dropna(inplace=True)

    def reindex_inplace(self, columns=None, index=None, overwrite_existing=True):
        """
        Change the table columns keeping the dataframe id
        """
        if columns is not None:
            # Set all new columns to NaN
            for col in columns:
                if col in self.columns.array and not overwrite_existing:
                    continue
                else:
                    self[col] = np.nan
            # Remove all columns not in given columns
            self.drop(self.columns.difference(columns).tolist(), axis=1, inplace=True)
        
        if index is not None:
            # Set all new index to NaN
            for idx in index:
                if idx in self.index.array and not overwrite_existing:
                    continue
                else:
                    self.loc[idx] = np.nan
            # Remove all index not in given index
            self.drop(self.index.difference(index).tolist(), axis=0, inplace=True)
    # def reindex_inplace(self, columns=None, index=None, overwrite_existing=True):
    #     """
    #     Change the table columns keeping the dataframe id
    #     """
    #     if columns is not None:
    #         # Set all new columns to NaN
    #         if not overwrite_existing:
    #             for col in self.columns.difference(columns):
    #                 self[col] = np.nan
    #         else:
    #             for col in columns:
    #                 self[col] = np.nan
    #         # Remove all columns not in given columns
    #         self.drop(self.columns.difference(columns).tolist(), axis=1, inplace=True)
        
    #     if index is not None:
    #         # Set all new index to NaN
    #         if not overwrite_existing:
    #             for idx in self.index.difference(index):
    #                 self.loc[idx] = np.nan
    #         else:
    #             for idx in index:
    #                 self.loc[idx] = np.nan
    #         # Remove all index not in given index
    #         self.drop(self.index.difference(index).tolist(), axis=0, inplace=True)
    
    def set_data(self, dataframe):
        """
        Method to add data from other dataframe to current dataframe
        """
        # Copy content
        for col, values in dataframe.iteritems():
            self[col] = values.values
        self.index = dataframe.index
        self.index.name = dataframe.index.name

    def load_pickle(self, path, intersection=False):
        """
        Method to load (intersecting) data from pickle
        """
        # Load dataframe
        loaded = pd.read_pickle(path)
        if intersection:
            # Find intersection columns and rows
            idx, cols = loaded.index.intersection(self.index), loaded.columns.intersection(self.columns)
            # Set intersecting data
            self.loc[idx, cols] = loaded.loc[idx, cols]
        else:
            
            self.reindex_inplace(loaded.columns)
            self.set_data(loaded)
