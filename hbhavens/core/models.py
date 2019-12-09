# -*- coding: utf-8 -*-
"""
Created on  : Fri Aug 11 11:33:42 2017
Author      : Guus Rongen, HKV Lijn in Water
Project     : HB Havens
Description : General classes for HB Havens

"""

import json
import logging
import os
import re
import shutil
import sys
import time
from itertools import product, chain

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from descartes import PolygonPatch
from matplotlib import path
from scipy.interpolate import interp2d, interp1d
from shapely.geometry import (
    LineString, MultiLineString, MultiPolygon, Point, Polygon)
from shapely.ops import linemerge, polygonize, unary_union
from shapely.prepared import prep as _prep

import warnings
from scipy import optimize
warnings.filterwarnings("ignore", category=optimize.OptimizeWarning)
from scipy.ndimage import label

import hbhavens.core.geometry as geometry
from hbhavens import io
from hbhavens.core import datamodels
from hbhavens.core.hares import Hares
from hbhavens.core.pharos import Pharos
from hbhavens.core.simple import SimpleCalculation
from hbhavens.core.swan import Swan

logger = logging.getLogger(__name__)

__version__ = "2.0"
__date__ = "september 2019"

class MainModel:
    """
    Main class. Contains all sub models.
    """
    def __init__(self):
        """Constructor"""
        # Create project class
        self.appName = "HB Havens"
        self.appVersion = __version__
        self.appDate = __date__
        self.project = Project(self)

        if getattr(sys, 'frozen', False):
            # If the application is run as a bundle, the pyInstaller bootloader
            # extends the sys module by a flag frozen=True and sets the app
            # path into variable _MEIPASS'.
            application_path = sys._MEIPASS
            self.datadir = os.path.join(application_path, 'data')
        else:
            application_path = os.path.dirname(os.path.abspath(__file__))
            self.datadir = os.path.join(application_path, '..', 'data')

        # Create schematisation class
        self.schematisation = HarborSchematisation(self)
        # Create input databases class
        self.hydraulic_loads = HydraulicLoads(self)
        # Create simple calculation class
        self.simple_calculation = SimpleCalculation(self)
        # Create advanced calculation Swan class
        self.swan = Swan(self)
        # Create advanced calculation Pharos class
        self.pharos = Pharos(self)
        # Initialiseren van de Hares calculatie class
        self.hares = Hares(self)
        # Create model uncertainties class
        self.modeluncertainties = ModelUncertainties(self)
        # Create export database class
        self.export = Export(self)

    def load_from_project(self):
        """
        Load project from dictionary (json)
        """
        logger.info('Loading project from file...')
        harbor_settings = self.project.settings['schematisation']
        database_settings = self.project.settings['hydraulic_loads']
        simple_settings = self.project.settings['simple']
        simple_finished = self.project.settings['simple']['finished']
        export_settings = self.project.settings['export']

        # Load harbor geometries
        if harbor_settings['flooddefence_ids']:
            for sectionid in harbor_settings['flooddefence_ids']:
                self.schematisation.add_flooddefence(sectionid)
        else:
            logger.info('..Did not load any dike sections')

        if harbor_settings['harbor_area_shape']:
            self.schematisation.add_harborarea(harbor_settings['harbor_area_shape'])
            logger.info('..Loaded harbor area')
        else:
            logger.info('..Did not load harbor area')

        if harbor_settings['breakwater_shape']:
            self.schematisation.add_breakwater(harbor_settings['breakwater_shape'])
            logger.info('..Loaded break waters')

            if len(self.schematisation.breakwaters) == 2:
                self.schematisation.generate_harbor_bound()
                logger.info('..Generated harbor bound for two breakwaters.')

            elif harbor_settings['entrance_coordinate']:
                self.schematisation.entrance_coordinate = tuple(harbor_settings['entrance_coordinate'])
                self.schematisation.generate_harbor_bound()
                logger.info('..Generated harbor bound for single breakwater and entrance coordinate.')

            else:
                logger.warning(f'No entrance coordinate given: {harbor_settings["entrance_coordinate"]}, did not generate harbor bound.')
        else:
            logger.warning('Did not load break waters from project')


        if harbor_settings['representative_bedlevel']:
            self.schematisation.set_bedlevel(harbor_settings['representative_bedlevel'])

        # Load databases
        if database_settings['HRD']:
            self.hydraulic_loads.add_HRD(database_settings['HRD'])
            logger.info('Added HRD while loading project')
        else:
            logger.warning('Did not add HRD while loading project')

        # Set support location and load result locations
        if harbor_settings['support_location_name']:
            self.schematisation.set_selected_support_location(harbor_settings['support_location_name'])
            logger.info(f'Loaded hydraulic loads for location "{harbor_settings["support_location_name"]}" from project')

        if harbor_settings['harbor_area_shape'] and harbor_settings['breakwater_shape'] and harbor_settings['flooddefence_ids'] and  harbor_settings['result_locations_shape']:
            self.schematisation.add_result_locations(harbor_settings['result_locations_shape'])
            logger.info('Loaded result locations from project')
        else:
            logger.warning(f'Did not load result locations: "{harbor_settings["result_locations_shape"]}" from project')

        # Load simple calculation results
        if self.project.settings['calculation_method']['method'] == 'simple':
            # Finished is overwritten by importing the geometries, reset
            simple_settings['finished'] = simple_finished
            if simple_settings['finished']:
                logger.info('Loaded simple calculation results from project')
                self.simple_calculation.load_results()

        # Load Advanced classes Swan and Pharos
        # if self.project.settings['calculation_method']['method'] == 'advanced':
            # print('Loaded swan iteration and calculation results')
        self.swan.load_from_project()

            # print('Loaded pharos calculation results')
        self.pharos.load_from_project()

            # Toegevoegd Svasek 04/10/18 - Laden van Hares class
            # print('Loaded hares calculation results')
        self.hares.load_from_project()

        # Load modeluncertainties
        self.modeluncertainties.load_tables()

        # Load exportmodel
        self.export.load_table()
        if export_settings['HLCD']:
            self.export.add_HLCD(export_settings['HLCD'])

        logger.info('Finished loading!\n')

    def save_tables(self):
        """
        Most of the project settings are saved in the json file. The tables
        however are more easily saved as pickle (python binary format). This
        method saves these tables.
        """

        # Simple calculation
        if self.project.settings['calculation_method']['method'] == 'simple':
            if self.project.settings['simple']['finished']:
                self.simple_calculation.save_results()

        # Advanced calculation
        # if self.project.settings['calculation_method']['method'] == 'advanced':
            # print('Saving swan tables')
        self.swan.save_tables()
        self.pharos.save_tables()
        self.hares.save_tables()

        # Modeluncertainties
        self.modeluncertainties.save_tables()

        # Exportmodel
        self.export.save_table()

class HydraulicLoads(datamodels.ExtendedDataFrame):
    """Class for loading and modifying hydraulic loads
    
    Parameters
    ----------
    datamodels.ExtendedDataFrame : subclass of pandas DataFrame
        parent
    """

    _metadata = [
        'schematisation', 'settings', 'description_dict', 'hydraulicloadid_dict',
        'input_columns', 'result_columns', 'wlevcol', 'xp', 'recalculated_loads'
    ]

    def __init__(self, mainmodel):
        # Copy ExtendedDataFrame propoerties
        super(HydraulicLoads, self).__init__()

        # Links
        self.schematisation = mainmodel.schematisation
        self.settings = mainmodel.project.settings

        # Empty dicts and lists
        self.description_dict = {}
        self.hydraulicloadid_dict = {}
        self.input_columns = []
        self.result_columns = []

        # Empty dataframe for recalculated loads
        self.recalculated_loads = datamodels.ExtendedDataFrame()

    def add_HRD(self, path):
        """
        Add HRD-database. This methods add the path variable, and puts
        it in the settings.

        The hydraulic loads are not loaded in this function, but the
        result locations are (from the HRDLocations)
        """

        # Change settings
        self.settings['hydraulic_loads']['HRD'] = path
        
        # Read database
        hrd = io.database.HRDio(path)
        self.schematisation.add_support_locations(hrd.read_HRDLocations())
        logger.info(f'Added support locations from "{path}"')

    def load_hydraulic_loads(self, interpolated=False):
        """
        Load the hydraulic loads from the support location from the settings

        Parameters
        ----------
        interpolated: bool
            In case of interpolated loads, the dataframe interpolated_loads is used
        """

        if not interpolated:
            # Load Hydraulic Loads
            hrdio = io.database.HRDio(self.settings['hydraulic_loads']['HRD'])
            locationid = self.schematisation.support_location['HRDLocationId']
            loads = hrdio.read_HydroDynamicData(locationid)

        if interpolated:
            loads = self.recalculated_loads.copy()

        # Drop empty columns
        loads.dropna(axis=1, how='all', inplace=True)

        # Drop available content (we dont overwrite the variable since we want
        # dont want to destroy the reference)
        self.delete_all()

        del self.input_columns[:]
        del self.result_columns[:]

        for col in loads.columns:
            if col in list(io.database.resultvariableids.values()):
                self.result_columns.append(col)
            else:
                # If only 1 closingsituationid, ignore the column
                if col == 'ClosingSituationId' and len(loads['ClosingSituationId'].unique()) == 1:
                    continue
                self.input_columns.append(col)

        # Reindex the columns
        self.reindex_inplace(columns=loads.columns)

        # Add new content
        self[loads.columns] = loads

        # Sort hydraulic loads
        self.sort_values(by=self.input_columns, inplace=True)

        # Derive descriptions
        self.create_descriptions()

        # Create load combination names
        for idx in self.index:
            self.at[idx, 'Description'] = self.description_dict[idx]

        # Set index name
        self.index.name = 'HydraulicLoadId'

        # Get water level columns
        if 'Water level' in self.input_columns:
            self.wlevcol = 'Water level'
        elif 'h' in self.result_columns:
            self.wlevcol = 'h'
        else:
            raise KeyError('No column for water level in hydraulic loads.')

    def create_descriptions(self):
        """
        Method to create description of hydraulic load input variables.

        A dictionary the go from hydraulic load id to description
        is created, and vice versa.
        """

        # Create format string
        format_str = ' '.join([io.database.intputvarabr[inp] for inp in self.input_columns])

        # Add a description to the dictionary for each load
        self.description_dict.clear()
        for row in self[self.input_columns].itertuples():
            self.description_dict[row.Index] = format_str.format(*row[1:])

        # Create a reversed dictionary
        self.hydraulicloadid_dict.clear()
        for loadid, description in self.description_dict.items():
            self.hydraulicloadid_dict[description] = loadid

    def _get_breaks(self, xp, fp, nround=3):
        """Method to find gradient changes in a list of x, f(x) values.
        
        Parameters
        ----------
        xp : np.array
            X-coordinates of data
        fp : np.array
            Y-coordinates of data
        nround : int, optional
            Number of decimals to round, by default 3
        
        Returns
        -------
        list
            List with x-coordinates of breaks
        """
        # DEMO (<=0.0001; <=0.01)
        # WBI (<=0.01; <=0.1)
        atol = 0.00001
        rtol = 0.01
        nbreaks = np.inf
        
        while nbreaks > 12:
            atol *= 10
            if round(atol) == 1:
                raise ValueError('Did not succeed in finding breaks. Search manually.')
            
            breaks = []
            # Calculate gradients
            dydx = np.diff(fp) / np.diff(xp)
            # Get position of gradient change (absolute tolerance=0.01: 0.01 m Hs to 1 m h)
            # Not close if absolute difference is larger than 1 cm hs per 1 m h AND
            # relative difference is larger than a factor 2

            notclose = ~np.isclose(dydx[1:], dydx[:-1], atol=atol) & ~(np.isclose(dydx[1:], dydx[:-1], rtol=rtol) | np.isclose(dydx[:-1], dydx[1:], rtol=rtol))
            notclose = np.r_[False, notclose] & np.r_[notclose, False]
            # Calculate intersection at gradient change
            for i in np.where(notclose)[0]:
                p1, p2, p3, p4 = (xp[i-1], fp[i-1]), (xp[i], fp[i]), (xp[i+1], fp[i+1]), (xp[i+2], fp[i+2])
                isect = geometry.intersection_lines([p1, p2], [p3, p4])
                if isect is not None:
                    breaks.append(round(isect[0], nround))

            nbreaks = len(breaks)
            
        return breaks

    def detect_waterlevel_breaks(self):
        """
        Method to detect water level breaks in hydraulic load table.
        First the breaks are searched for each combination of wind speed and wind direction,
        from the collected set, only the breaks with more than one occurence are returned.
        """
        wlevs = []
        for comb, group in self.groupby(['Wind direction', 'Wind speed']):
            if (group['Hs'] > 0).any():
                hs, waterlevel = group[['Hs', self.wlevcol]].sort_values(by=self.wlevcol).drop_duplicates().values.T
                wlevs.extend(self._get_breaks(waterlevel, hs, 4))

        # Return all levels with more than one occurence
        unique, count = np.unique(wlevs, return_counts=True)
        wlevs = unique[count > 1].tolist()
        wlevs.insert(0, np.floor(self[self.wlevcol].min()*1000)/1000)
        wlevs.append(np.ceil(self[self.wlevcol].max()*1000)/1000)
        wlevs = np.array(wlevs)

        # Group levels
        labeled, n = label(np.isclose(wlevs[:, None], wlevs[None, :], atol=1e-2))
        h = []
        for i in range(1, n+1):
            idx = np.unique(np.where(labeled == i)[0])
            for decimals in [0, 1, 2, 3]:
                roundeq = wlevs[idx] == np.round(wlevs[idx], decimals)
                if any(roundeq):
                    h.append(wlevs[idx][roundeq].mean())
                    break
            else:
                h.append(wlevs[idx].mean())
        wlevs = h[:]

        # Add to settings
        del self.settings['hydraulic_loads']['waterlevels'][:]
        self.settings['hydraulic_loads']['waterlevels'].extend(wlevs)

        logger.info('Detected water level breaks at: {}'.format(', '.join(map(str, wlevs))))

    def _piecewise_linear_1(self, x, b, k1):
        k = [k1]
        condlist = [x <= self.xp[1]]
        funclist = [lambda x: b + k[0]*x]
        return np.piecewise(x, condlist, funclist)

    def _piecewise_linear_2(self, x, b, k1, k2):
        k = [k1, k2]
        condlist = [x <= self.xp[1], x >= self.xp[-2]]
        funclist = [
            lambda x: b + k[0]*x,
            lambda x: b + k[0]*x + sum([k[i+1]*(x - self.xp[i+1]) for i in range(1)]),
        ]
        return np.piecewise(x, condlist, funclist)

    def _piecewise_linear_3(self, x, b, k1, k2, k3):
        k = [k1, k2, k3]
        condlist = [x <= self.xp[1]] + [(x >= self.xp[i]) & (x <= self.xp[i+1]) for i in range(1, len(self.xp)-2)] + [x >= self.xp[-2]]
        funclist = [
            lambda x: b + k[0]*x,
            lambda x: b + k[0]*x + sum([k[i+1]*(x - self.xp[i+1]) for i in range(1)]),
            lambda x: b + k[0]*x + sum([k[i+1]*(x - self.xp[i+1]) for i in range(2)]),
        ]
        return np.piecewise(x, condlist, funclist)

    def _piecewise_linear_4(self, x, b, k1, k2, k3, k4):
        k = [k1, k2, k3, k4]
        condlist = [x <= self.xp[1]] + [(x >= self.xp[i]) & (x <= self.xp[i+1]) for i in range(1, len(self.xp)-2)] + [x >= self.xp[-2]]
        funclist = [
            lambda x: b + k[0]*x,
            lambda x: b + k[0]*x + sum([k[i+1]*(x - self.xp[i+1]) for i in range(1)]),
            lambda x: b + k[0]*x + sum([k[i+1]*(x - self.xp[i+1]) for i in range(2)]),
            lambda x: b + k[0]*x + sum([k[i+1]*(x - self.xp[i+1]) for i in range(3)]),
        ]
        return np.piecewise(x, condlist, funclist)

    def _piecewise_linear_5(self, x, b, k1, k2, k3, k4, k5):
        k = [k1, k2, k3, k4, k5]
        condlist = [x <= self.xp[1]] + [(x >= self.xp[i]) & (x <= self.xp[i+1]) for i in range(1, len(self.xp)-2)] + [x >= self.xp[-2]]
        funclist = [
            lambda x: b + k[0]*x,
            lambda x: b + k[0]*x + sum([k[i+1]*(x - self.xp[i+1]) for i in range(1)]),
            lambda x: b + k[0]*x + sum([k[i+1]*(x - self.xp[i+1]) for i in range(2)]),
            lambda x: b + k[0]*x + sum([k[i+1]*(x - self.xp[i+1]) for i in range(3)]),
            lambda x: b + k[0]*x + sum([k[i+1]*(x - self.xp[i+1]) for i in range(4)]),
        ]
        return np.piecewise(x, condlist, funclist)

    def _piecewise_linear_6(self, x, b, k1, k2, k3, k4, k5, k6):
        k = [k1, k2, k3, k4, k5, k6]
        condlist = [x <= self.xp[1]] + [(x >= self.xp[i]) & (x <= self.xp[i+1]) for i in range(1, len(self.xp)-2)] + [x >= self.xp[-2]]
        funclist = [
            lambda x: b + k[0]*x,
            lambda x: b + k[0]*x + sum([k[i+1]*(x - self.xp[i+1]) for i in range(1)]),
            lambda x: b + k[0]*x + sum([k[i+1]*(x - self.xp[i+1]) for i in range(2)]),
            lambda x: b + k[0]*x + sum([k[i+1]*(x - self.xp[i+1]) for i in range(3)]),
            lambda x: b + k[0]*x + sum([k[i+1]*(x - self.xp[i+1]) for i in range(4)]),
            lambda x: b + k[0]*x + sum([k[i+1]*(x - self.xp[i+1]) for i in range(5)]),
        ]
        return np.piecewise(x, condlist, funclist)

    def _piecewise_linear_7(self, x, b, k1, k2, k3, k4, k5, k6, k7):
        k = [k1, k2, k3, k4, k5, k6, k7]
        condlist = [x <= self.xp[1]] + [(x >= self.xp[i]) & (x <= self.xp[i+1]) for i in range(1, len(self.xp)-2)] + [x >= self.xp[-2]]
        funclist = [
            lambda x: b + k[0]*x,
            lambda x: b + k[0]*x + sum([k[i+1]*(x - self.xp[i+1]) for i in range(1)]),
            lambda x: b + k[0]*x + sum([k[i+1]*(x - self.xp[i+1]) for i in range(2)]),
            lambda x: b + k[0]*x + sum([k[i+1]*(x - self.xp[i+1]) for i in range(3)]),
            lambda x: b + k[0]*x + sum([k[i+1]*(x - self.xp[i+1]) for i in range(4)]),
            lambda x: b + k[0]*x + sum([k[i+1]*(x - self.xp[i+1]) for i in range(5)]),
            lambda x: b + k[0]*x + sum([k[i+1]*(x - self.xp[i+1]) for i in range(6)]),
        ]
        return np.piecewise(x, condlist, funclist)

    def _piecewise_linear_8(self, x, b, k1, k2, k3, k4, k5, k6, k7, k8):
        k = [k1, k2, k3, k4, k5, k6, k7, k8]
        condlist = [x <= self.xp[1]] + [(x >= self.xp[i]) & (x <= self.xp[i+1]) for i in range(1, len(self.xp)-2)] + [x >= self.xp[-2]]
        funclist = [
            lambda x: b + k[0]*x,
            lambda x: b + k[0]*x + sum([k[i+1]*(x - self.xp[i+1]) for i in range(1)]),
            lambda x: b + k[0]*x + sum([k[i+1]*(x - self.xp[i+1]) for i in range(2)]),
            lambda x: b + k[0]*x + sum([k[i+1]*(x - self.xp[i+1]) for i in range(3)]),
            lambda x: b + k[0]*x + sum([k[i+1]*(x - self.xp[i+1]) for i in range(4)]),
            lambda x: b + k[0]*x + sum([k[i+1]*(x - self.xp[i+1]) for i in range(5)]),
            lambda x: b + k[0]*x + sum([k[i+1]*(x - self.xp[i+1]) for i in range(6)]),
            lambda x: b + k[0]*x + sum([k[i+1]*(x - self.xp[i+1]) for i in range(7)]),
        ]
        return np.piecewise(x, condlist, funclist)

    def _piecewise_linear_9(self, x, b, k1, k2, k3, k4, k5, k6, k7, k8, k9):
        k = [k1, k2, k3, k4, k5, k6, k7, k8, k9]
        condlist = [x <= self.xp[1]] + [(x >= self.xp[i]) & (x <= self.xp[i+1]) for i in range(1, len(self.xp)-2)] + [x >= self.xp[-2]]
        funclist = [
            lambda x: b + k[0]*x,
            lambda x: b + k[0]*x + sum([k[i+1]*(x - self.xp[i+1]) for i in range(1)]),
            lambda x: b + k[0]*x + sum([k[i+1]*(x - self.xp[i+1]) for i in range(2)]),
            lambda x: b + k[0]*x + sum([k[i+1]*(x - self.xp[i+1]) for i in range(3)]),
            lambda x: b + k[0]*x + sum([k[i+1]*(x - self.xp[i+1]) for i in range(4)]),
            lambda x: b + k[0]*x + sum([k[i+1]*(x - self.xp[i+1]) for i in range(5)]),
            lambda x: b + k[0]*x + sum([k[i+1]*(x - self.xp[i+1]) for i in range(6)]),
            lambda x: b + k[0]*x + sum([k[i+1]*(x - self.xp[i+1]) for i in range(7)]),
            lambda x: b + k[0]*x + sum([k[i+1]*(x - self.xp[i+1]) for i in range(8)]),
        ]
        return np.piecewise(x, condlist, funclist)

    def _piecewise_linear_10(self, x, b, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10):
        k = [k1, k2, k3, k4, k5, k6, k7, k8, k9, k10]
        condlist = [x <= self.xp[1]] + [(x >= self.xp[i]) & (x <= self.xp[i+1]) for i in range(1, len(self.xp)-2)] + [x >= self.xp[-2]]
        funclist = [
            lambda x: b + k[0]*x,
            lambda x: b + k[0]*x + sum([k[i+1]*(x - self.xp[i+1]) for i in range(1)]),
            lambda x: b + k[0]*x + sum([k[i+1]*(x - self.xp[i+1]) for i in range(2)]),
            lambda x: b + k[0]*x + sum([k[i+1]*(x - self.xp[i+1]) for i in range(3)]),
            lambda x: b + k[0]*x + sum([k[i+1]*(x - self.xp[i+1]) for i in range(4)]),
            lambda x: b + k[0]*x + sum([k[i+1]*(x - self.xp[i+1]) for i in range(5)]),
            lambda x: b + k[0]*x + sum([k[i+1]*(x - self.xp[i+1]) for i in range(6)]),
            lambda x: b + k[0]*x + sum([k[i+1]*(x - self.xp[i+1]) for i in range(7)]),
            lambda x: b + k[0]*x + sum([k[i+1]*(x - self.xp[i+1]) for i in range(8)]),
            lambda x: b + k[0]*x + sum([k[i+1]*(x - self.xp[i+1]) for i in range(9)]),
        ]
        return np.piecewise(x, condlist, funclist)

    def piecewise_linear_n(self, ndata):
        numseg = len(self.xp) - 1
        if ndata < (numseg+1):
            raise ValueError('Het aantal waterstandsniveaus is te groot voor het aantal datapunten. Kies minder niveaus of gebruik te methode "interpoleren".')
        if numseg > 10:
            raise NotImplementedError('Segmented linear regression kan gebruikt worden voor maximaal 10 segementen.')
        return getattr(self, f'_piecewise_linear_{numseg:d}')

    def _get_relevant_levels(self, waterlevels):
        """Method to get relevant breaks based on waterlevels
        and total breaks.
        
        Parameters
        ----------
        waterlevels : numpy.ndarray
            Water levels
        """
        breaks = np.array(self.settings['hydraulic_loads']['waterlevels'])
        wlevbins = np.maximum(1, np.minimum(np.digitize(waterlevels, breaks), len(breaks)-1))
        relevant = np.unique(np.r_[breaks[wlevbins-1], breaks[wlevbins]])
        return relevant

    def calculate_waveconditions(self):
        """
        Use segmented regression to calculate wave conditions.
        """

        newloads = {}
        wlevbreaks = self.settings['hydraulic_loads']['waterlevels']
        groupbycols = ['Wind direction', 'Wind speed']
        load_cols = self.result_columns[:]

        logger.info('Recalculating wave conditions at: {}'.format(', '.join(map(str, wlevbreaks))))
        
        # Create datastructure to save loads
        newloads = {col: [] for col in load_cols + groupbycols}

        for comb, group in self.reindex(columns=groupbycols + load_cols).groupby(groupbycols):
            grouploads = group.reindex(columns=load_cols).sort_values(by=self.wlevcol).drop_duplicates()
            waterlevels = grouploads[self.wlevcol].values
            waveheights = grouploads['Hs'].values
            
            # Set the break points
            self.xp = self._get_relevant_levels(waterlevels)
            if all(waveheights == 0.0):
                hs_regression = np.zeros_like(self.xp)
            elif len(np.unique(waveheights)) == 1:
                hs_regression = np.ones_like(self.xp) * waveheights[0]
            else:
                if self.settings['hydraulic_loads']['recalculate_method'] == 'regression':
                    # Curve fit a segmented regression model
                    p, _ = optimize.curve_fit(self.piecewise_linear_n(len(waterlevels)), waterlevels, waveheights)
                    hs_regression = self.piecewise_linear_n(len(waterlevels))(self.xp, *p)
                elif self.settings['hydraulic_loads']['recalculate_method'] == 'interpolation': 
                    # Interpolate
                    hs_regression = interp1d(waterlevels, waveheights, fill_value='extrapolate')(self.xp)
                else:
                    raise KeyError('Unknown recalculation method: {}'.format(self.settings['hydraulic_loads']['recalculate_method']))
            
            # Add data to output struct
            newloads[self.wlevcol].extend(self.xp)
            newloads['Hs'].extend(np.maximum(0, hs_regression).tolist())

            # Interpolate other columns (Tp, Tm-1,0, wave direction)
            for col in load_cols:
                if col in [self.wlevcol, 'Hs']:
                    continue
                if col == 'Wave direction':
                    interp_loads = geometry.interp_angles(hs_regression, waveheights, grouploads[col].values)
                    newloads[col].extend(interp_loads.tolist())
                else:
                    interp_loads = np.maximum(0, np.interp(hs_regression, waveheights, grouploads[col].values))
                    newloads[col].extend(interp_loads.tolist())

            # Add load columns
            for col, val in zip(groupbycols, comb):
                newloads[col].extend([val] * len(self.xp))

        # To dataframe and add index
        newloads = pd.DataFrame.from_dict(newloads)
        newloads.index = list(range(1, len(newloads) + 1))
        newloads.index.name = 'HydraulicLoadId'

        # Add to recalculated_loads
        self.recalculated_loads.delete_all()
        self.recalculated_loads.reindex_inplace(columns=newloads.columns)
        self.recalculated_loads.set_data(newloads.round(3))
        # Convert water level 'h' to input variable 'Water level'
        self.recalculated_loads.rename(columns={'h': 'Water level'}, inplace=True)
        
    def adapt_interpolated(self):
        """Method to adapt interpolated loads as hydrauli loads
        """
        # Add interpolated loads
        self.settings['hydraulic_loads']['recalculate_waterlevels'] = True
        self.schematisation.set_selected_support_location(name=self.settings['schematisation']['support_location_name'])

    def restore_non_interpolated(self):
        """Method to restore original (non interpolated) loads to hydraulic loads class
        """
        # Restore non interpolated loads
        self.settings['hydraulic_loads']['recalculate_waterlevels'] = False
        self.schematisation.set_selected_support_location(name=self.settings['schematisation']['support_location_name'])

    def interpolate_wave_conditions(self, results, column_mapping):
        """Method to interpolate the database waterlevels to the recalculated
        water levels, before exporting.

        Parameters
        ----------
        results : pandas.DataFrame
            Dataframe with results that are interpolated on original water levels
            from hydraulic loads.
        
        Returns
        -------
        pandas.DataFrame
            DataFrame with interpolated loads
        """
        # Get loads from database again
        hrdio = io.database.HRDio(self.settings['hydraulic_loads']['HRD'])
        locationid = self.schematisation.support_location['HRDLocationId']
        loads = hrdio.read_HydroDynamicData(locationid)

        # Get the water levels from the loads, per wind direction and wind speed
        waterlevels = {}
        hydraulicloadids = {}
        for comb, group in loads.sort_values(by='h').groupby(['Wind direction', 'Wind speed']):
            waterlevels[comb] = group['h'].array
            hydraulicloadids[comb] = group.index.tolist()

        # Create datastructure for output
        output = {col: [] for col in ['Location', 'HydraulicLoadId', 'h'] + self.result_columns}
        
        # For each combination of wind direction and closing situation in loads, get the hydraulicloadids
        for comb, group in self.groupby(['Wind direction', 'Wind speed']):
            
            # Select the result data for the wind direction and wind speed
            recalculated = results.loc[results['HydraulicLoadId'].isin(group.index.array)].sort_values(by='Water level')

            # Interpolate the waterlevels for the recalculated loads, per location
            for location, recalclocdata in recalculated.groupby('Location'):
                
                # Get waterlevels from recalculated loads
                waterlevelp = recalclocdata['Water level'].array
                # Interpolate the results for the recalculated water levels on the original water levels
                for resultvar in self.result_columns:
                    # Interpolate angles
                    if resultvar == 'Wave direction':
                        output[resultvar].extend(geometry.interp_angles(
                            waterlevels[comb], waterlevelp, recalclocdata[column_mapping[resultvar]].array, extrapolate=True).tolist())
                    # Interpolate other values
                    else:
                        interpvalues = interp1d(waterlevelp, recalclocdata[column_mapping[resultvar]].array, fill_value='extrapolate')(waterlevels[comb])
                        output[resultvar].extend(np.maximum(interpvalues, 0).tolist())
                    # Add waterlevels itself
                output['h'].extend(waterlevels[comb])

                # Add location and HydraulicLoadId
                output['Location'].extend([location] * len(waterlevels[comb]))
                output['HydraulicLoadId'].extend(hydraulicloadids[comb])

        # Convert to pandas DataFrame
        output = pd.DataFrame.from_dict(output).sort_values(by=['Location', 'HydraulicLoadId'])
        # Join original hydraulic loads
        cols = loads.columns.difference(list(io.database.resultvariableids.values())).tolist()
        output = output.reindex(columns=output.columns.tolist() + cols)
        output[cols] = np.tile(loads[cols].sort_index().values, (len(output['Location'].unique()), 1))

        return output

class ModelUncertainties:
    """
    Class for modeluncertainty data. Contains methods to load and combine
    modeluncertainties for result locations.

    Parameters
    ----------
    parent : MainModel class
        Parent class
    """

    def __init__(self, mainmodel):
        """
        Constructor
        """
        # Link mainmodel
        self.mainmodel = mainmodel
        # Add dataframe for keeping model uncertainty per location
        self.metacolumns = ['Naam', 'Optie']
        self.table = datamodels.ExtendedDataFrame(columns=self.metacolumns)
        # Create empty dataframe for harbor uncertainty and combined uncertainty
        self.harbor_unc = datamodels.ExtendedDataFrame()
        self.combined_unc = datamodels.ExtendedDataFrame()

        self.variable_columns = []

    def load_modeluncertainties(self):
        """
        Method to load modeluncertainties from HRD. The method is called after the
        support location is selected. At this point the modeluncertainties, as well
        as the database format, are known.
        """

        # Add columns to table based on database format (included or excluded water level h)
        unccols = self.mainmodel.hydraulic_loads.result_columns[:]
        # Remove wave direction, has no uncertainty
        if 'Wave direction' in unccols:
            unccols.remove('Wave direction')
        # Add water level in case recalculated
        if self.mainmodel.project.settings['hydraulic_loads']['recalculate_waterlevels']:
            unccols.insert(0, 'h')

        del self.variable_columns[:]
        self.variable_columns.extend([' '.join(i[::-1]) for i in product(unccols, ['mu', 'sigma'])])
        columns = self.metacolumns + self.variable_columns
        self.table.reindex_inplace(columns=columns, overwrite_existing=False)

        # Retrieve location id from schematisation
        locationid = self.mainmodel.schematisation.support_location['HRDLocationId']

        # Load model uncertainties
        hrd = io.database.HRDio(self.mainmodel.project.settings['hydraulic_loads']['HRD'])
        uncertainties = hrd.read_UncertaintyModelFactor(locationid)
        self.supportloc_unc = uncertainties.reindex(index=unccols).dropna(how='all')

        # Create empty dataframe for harbor uncertainty and combined uncertainty
        self.harbor_unc.reindex_inplace(columns=self.supportloc_unc.columns, index=self.supportloc_unc.index, overwrite_existing=False)
        self.combined_unc.reindex_inplace(columns=self.supportloc_unc.columns, index=self.supportloc_unc.index, overwrite_existing=False)
        
    def add_result_locations(self):
        """
        Add result locations to modeluncertainty dataframe.
        It can occur that there already is a dataframe with result locations
        present. In that case, delete those result locations and overwrite
        with the new one.
        """
        # Drop all content
        if not self.table.empty:
            self.table.iloc[:, 0] = np.nan
            self.table.dropna(inplace=True)

        # Retrieve locationnames
        self.table['Naam'] = self.mainmodel.schematisation.result_locations['Naam'].values
        self.table['Optie'] = ''

    def calculate_combined_uncertainty(self):
        """
        Calculate combined uncertainty for multiplicative or additive model.
        The following formulas are implemented:

        Multiplicative (wave parameters):
        ---------------
        mu_V = mu_X * mu_Y
        Var(V) = Var(X)*Var(Y) + Var(X)(mu_Y)^2 + Var(Y)(mu_X)**2
        sigma_V = (Var(V))**0.5

        Additive (water level):
        ---------------
        mu_V = mu_X + mu_Y
        Var(V) = Var(X) + Var(Y)
        sigma_V = (Var(V))**0.5
        """

        # Convert to float
        try:
            for col in self.harbor_unc.columns:
                self.harbor_unc[col] = pd.to_numeric(self.harbor_unc[col])
        except:
            raise TypeError('Niet alle waarden in kolom {} zijn numeriek. Corrigeer de waarden.'.format(col))

        # For each parameter
        for param, row in self.combined_unc.iterrows():
            Xmu = self.supportloc_unc.at[param, 'mu']
            Xvar = self.supportloc_unc.at[param, 'sigma'] ** 2
            Ymu = self.harbor_unc.at[param, 'mu']
            Yvar = self.harbor_unc.at[param, 'sigma'] ** 2

            # Additive model
            if param == 'h':
                self.combined_unc.at[param, 'mu'] = round(Xmu + Ymu, 3)
                self.combined_unc.at[param, 'sigma'] = round((Xvar + Yvar) ** 0.5, 3)

            # Multiplicative model
            elif param in ['Hs', 'Tm-1,0', 'Tp', 'Tpm', 'Ts']:
                self.combined_unc.at[param, 'mu'] = round(Xmu * Ymu, 3)
                self.combined_unc.at[param, 'sigma'] = round((Xvar * Yvar + Xvar * Ymu ** 2 + Yvar * Xmu ** 2) ** 0.5, 3)

            else:
                raise KeyError(f'Parameter "{param}" not recognized. Cannot calculate combined uncertainty since it is not clear to me which model (additive of multiplicative) to use.')

    def save_tables(self):
        """
        Save tables to file
        """
        self.table.to_pickle(os.path.join(self.mainmodel.project.filedir, 'modeluncertainties_table.pkl'))
        if hasattr(self, 'harbor_unc'):
            self.harbor_unc.to_pickle(os.path.join(self.mainmodel.project.filedir, 'modeluncertainties_harborunc.pkl'))
            self.combined_unc.to_pickle(os.path.join(self.mainmodel.project.filedir, 'modeluncertainties_combinedunc.pkl'))

    def load_tables(self):
        """
        Save tables to file
        """
        path = os.path.join(self.mainmodel.project.filedir, 'modeluncertainties_table.pkl')
        if os.path.exists(path):
            self.table.load_pickle(path, intersection=True)

        path = os.path.join(self.mainmodel.project.filedir, 'modeluncertainties_harborunc.pkl')
        if os.path.exists(path):
            self.harbor_unc.load_pickle(path, intersection=True)
            
        path = os.path.join(self.mainmodel.project.filedir, 'modeluncertainties_combinedunc.pkl')
        if os.path.exists(path):
            self.combined_unc.load_pickle(path, intersection=True)
            
class Export:

    def __init__(self, mainmodel):
        """
        Constructor
        """
        self.mainmodel = mainmodel
        self.hydraulic_loads = mainmodel.hydraulic_loads
        self.settings = mainmodel.project.settings

        self.dfcolumns = ['Naam', 'Exportnaam', 'SQLite-database', '...']
        self.export_dataframe = pd.DataFrame(columns=self.dfcolumns)

        self.hrd_connections = []
        self.hlcd_connection = None
        self.config_connections = []

        self.resulttable = io.database.ExportResultTable(self.mainmodel)


    def add_result_locations(self):
        # Drop all content
        if not self.export_dataframe.empty:
            self.export_dataframe.iloc[:, 0] = np.nan
            self.export_dataframe.dropna(inplace=True)

        # Retrieve locationnames
        self.location_names = self.mainmodel.schematisation.result_locations['Naam'].values.tolist()
        self.export_dataframe['Naam'] = self.location_names

        # Generate export names
        # for row in self.export_dataframe.itertuples():
            # dijkring, traject = self.settings


    def add_HLCD(self, path):
        """
        Add HLCD-database for modification
        """

        # Read database
        self.HLCDpath = path
        self.settings['export']['HLCD'] = path

    def remove_locations_from_dbs(self):
        """
        Method to remove locations from hrd and hlcd.
        """
        # Remove locations from HRDLocations, with connected data
        #-------------------------------------------------------------
        hrd_remove_ids = []
        for hrd in self.hrd_connections:
            hrd_remove_ids.extend(hrd.remove_locations(self.mainmodel.schematisation.inner, exemption=[self.mainmodel.schematisation.support_location['Name']]))

        if not self.settings['export']['export_HLCD_and_config'] or not any(hrd_remove_ids):
            return None

        # Remove locations from HLCD
        #-------------------------------------------------------------
        hlcd_remove_ids = self.hlcd_connection.remove_locations(hrd_remove_ids)
        if hlcd_remove_ids is None:
            logger.warning('No locations removed from HLCD, while locations from HRD are removed.')

        # Remove locations from config
        #-------------------------------------------------------------
        for config in self.config_connections:
            config.remove_locations(hlcd_remove_ids)


    def _check_paths(self):
        """
        Check if all paths exist
        """
        export_settings = self.settings['export']

        if export_settings['export_HLCD_and_config']:
            if export_settings['HLCD'] == '':
                raise OSError('Geen HLCD opgegeven.')
            if not os.path.exists(export_settings['HLCD']):
                raise OSError('Path not found: "{}"'.format(export_settings['HLCD']))

        # Collect all paths
        paths = [export_settings['HLCD']] if export_settings['export_HLCD_and_config'] else []
        for path in self.export_dataframe['SQLite-database'].dropna(how='any').unique():
            paths.append(path)
            if export_settings['export_HLCD_and_config']:
                paths.append(path.replace('.sqlite', '.config.sqlite'))

        # Check if all paths exist:
        for path in paths:
            if not os.path.exists(path):
                raise OSError('Path not found: "{}"'.format(path))

    def create_results_locations(self):
        """
        Create table with result locations
        """
        # Select result locations
        result_locations = self.mainmodel.schematisation.result_locations.copy()

        # Add export name (Exportnaam)
        result_locations = result_locations.merge(
            self.export_dataframe.dropna(subset=['SQLite-database'])[['Naam', 'Exportnaam']], on='Naam', how='inner')

        # Get system id
        systemid = self.hrd_connections[0].get_system_id()

        # Get track id
        for hrd in self.hrd_connections:
            # Get names in database
            names = self.export_dataframe.loc[self.export_dataframe['SQLite-database'].eq(hrd.path), 'Naam']
            hrdidx = result_locations['Naam'].isin(names)
            # Assign trackid
            trackid = hrd.get_track_id()
            result_locations.loc[hrdidx, 'TrackId'] = trackid


        # Get maximum location id (in watersystem).
        for hrd in self.hrd_connections:
            # Get names in database
            names = self.export_dataframe.loc[self.export_dataframe['SQLite-database'].eq(hrd.path), 'Naam']
            hrdidx = result_locations['Naam'].isin(names)

            # If no HLCD is known, get the max location id from the database and append the new locations
            if not self.settings['export']['export_HLCD_and_config']:
                maxlocid = max(hrd.get_max_hrdlocation_id() for hrd in self.hrd_connections)
                # Set HRDLocationId, LocationId is not needed
                result_locations.loc[hrdidx, 'HRDLocationId'] = np.arange(hrdidx.sum()) + 1 + maxlocid

            # If HLCD is known, the HRDLocationId's still need to be defined. Find the max per database, and append.
            else:
                maxlocid, descriptive_id = self.hlcd_connection.get_max_hrdlocation_id(systemid, trackid=hrd.get_track_id())
                # Set HRDLocationId and add LocationId
                result_locations.loc[hrdidx, 'HRDLocationId'] = np.arange(hrdidx.sum()) + 1 + maxlocid
                if descriptive_id:
                    result_locations.loc[hrdidx, 'LocationId'] = result_locations.loc[hrdidx, 'HRDLocationId']
                else:
                    result_locations.loc[hrdidx, 'LocationId'] = result_locations.loc[hrdidx, 'HRDLocationId'] + systemid * 100000

        # Do a last check for all LocationId's and HRDLocationId's, in case databases are not set up as expected
        if self.settings['export']['export_HLCD_and_config']:
            locationid_count = self.hlcd_connection.check_element_presence(table='Locations', column='LocationId', elements=result_locations['LocationId'].tolist())
            # If LocationIds not unique
            if locationid_count:
                logger.warning('Non unique LocationIds found. Generating new Ids from absolute maximum.')
                results_locations['LocationId'] = np.arange(len(result_locations)) + 1 + self.hlcd_connection.get_max_location_id()

        # Count the present HRDLocationId's
        hrdlocationid_count = 0
        for hrd in self.hrd_connections:
            hrdlocationid_count += hrd.check_element_presence(table='HRDLocations', column='HRDLocationId', elements=result_locations['HRDLocationId'].tolist())

        # If HRDLocationIds not unique
        if hrdlocationid_count:
            logger.warning('Non unique HRDLocationIds found. Generating new Ids from absolute maximum.')
            maxhrdlocationid = max(hrd.get_max_hrdlocation_id() for hrd in self.hrd_connections)
            maxhrdlocationid = maxhrdlocationid if self.hlcd_connection is None else max(self.hlcd_connection.get_max_hrdlocation_id()[0], maxhrdlocationid)
            results_locations = np.arange(len(result_locations)) + 1 + maxhrdlocationid

        # Add InterpolationSupportId
        if self.settings['export']['export_HLCD_and_config']:
            supportlocid = self.mainmodel.schematisation.support_location['HRDLocationId']
            int_support_id = self.hlcd_connection.get_interpolation_support_id(systemid, supportlocid=supportlocid)
        else:
            if (self.hrd_connections[0].get_type_of_hydraulic_load_id() == 3) and (systemid != 14):
                logger.warning('Geen InterpolationSupportId gevonden. Deze wordt wel verwacht voor een kustsysteem.')
            int_support_id = np.nan
        result_locations['InterpolationSupportId'] = int_support_id

        # Add TypeOfHydraulicDataId
        result_locations['TypeOfHydraulicDataId'] = self.hrd_connections[0].get_type_of_hydraulic_load_id()

        return result_locations

    def get_results(self, result_locations):

        # Get flags from model settings
        simple = self.settings['calculation_method']['method'] == 'simple'
        advanced = self.settings['calculation_method']['method'] == 'advanced'
        pharos = self.settings['calculation_method']['include_pharos']
        hares = self.settings['calculation_method']['include_hares']

        # Get the right function to add results
        if self.settings['hydraulic_loads']['recalculate_waterlevels']:
            add_function = self.resulttable.add_interpolated_results
        else:
            add_function = self.resulttable.add_results

        # Simple
        if simple:
            column_mapping = self.mainmodel.simple_calculation.result_parameters
            add_function(self.mainmodel.simple_calculation.combinedresults.output.reset_index(), column_mapping)

        # Advanced, swan and pharos
        elif advanced and pharos:
            column_mapping = self.mainmodel.pharos.result_parameters
            add_function(self.mainmodel.pharos.calculation_results, column_mapping)

        # Advanced, swan and hares
        elif advanced and hares:
            column_mapping = self.mainmodel.hares.result_parameters
            add_function(self.mainmodel.hares.calculation_results, column_mapping)

        # Advanced, only swan
        elif advanced and not (hares or pharos):
            column_mapping = self.mainmodel.swan.result_parameters
            add_function(self.mainmodel.swan.calculation_results, column_mapping)

        else:
            raise ValueError(f'No known combination of models could be derived from settings: simple={simple} advanced={advanced} pharos={pharos} hares={hares}')

        # Add HRDLocationId from result_locations
        self.resulttable.set_hrdlocationid(result_locations[['Naam', 'HRDLocationId']])

    def export_output_to_database(self, progress_function=None):
        """
        Export output to databasepath

        Steps:
            1. Check if all paths exist, if not: error
            2. Get system ids and check if all databases are from the same system
            3. Determine which locations should be removed from the databases
            4. Remove and add locations to the HLCD, if present
            5. For each HRD to export to:
                - Remove locations with corresponding data
                - Add new locations
                - Add new result data
                - Add new uncertainty data
                - Remove data from config database, if present
                - Add data to config database, if present


        Parameters
        ----------
        databasepath : str
            path to database which is modified
        group : geopandas.GeoDataFrame
            GeoDataFrame with at least the columns Naam and Exportnaam
        """

        if progress_function is None:
            progress_function = lambda x: None

        # First check the paths
        self._check_paths()
        progress_function(5)

        # Create connections
        for database in self.export_dataframe['SQLite-database'].unique():
            self.hrd_connections.append(io.database.HRDio(database))

        if self.settings['export']['export_HLCD_and_config']:
            for database in self.export_dataframe['SQLite-database'].tolist():
                self.config_connections.append(io.database.Configio(database.replace('.sqlite' , '.config.sqlite')))
            self.hlcd_connection = io.database.HLCDio(self.settings['export']['HLCD'])

        # Determine systemid's. If multiple, raise an error
        if len(set([hrd.get_system_id() for hrd in self.hrd_connections])) > 1:
            # Raise error for multiple systems
            raise NotImplementedError((
                'In de lijst van exportdatabases komen verschillende watersystemen '
                f'(id\'s: "{", ".join(map(str, systemids))}" voor. Dit is niet toegestaan omdat niet alle '
                'watersystemen eenzelfde opbouw van de database hebben. Voer de export apart uit voor elk watersysteem.'
            ))

        # Create result location overview
        # Note that the location id's are determined before the locations are removed from hrd (and hlcd),
        # to prevent reusing id's
        result_locations = self.create_results_locations()
        progress_function(5)

        # Remove present locations
        self.remove_locations_from_dbs()
        progress_function(5)

        # Add locations to HLCD
        if self.settings['export']['export_HLCD_and_config']:
            self.hlcd_connection.add_locations(result_locations)

        progress_function(5)

        # Add results to resulttable
        self.get_results(result_locations)
        progress_function(15)

        # Export per HRD
        #=======================================================================
        nhrds = len(self.hrd_connections)
        for i, hrd in enumerate(self.hrd_connections):
            # Get names of locations that need to be in database
            names = self.export_dataframe.loc[self.export_dataframe['SQLite-database'].eq(hrd.path), 'Naam']

            if len(set(names)) != len(names):
                raise ValueError('Niet alle locatienamen binnen de database zijn uniek.')

            # Select the part of the result locations in the group
            group_location = result_locations.loc[result_locations['Naam'].isin(names), :]

            # Export HRDLocations
            #-------------------------------------------------------------
            hrd.add_hrd_locations(group_location)
            progress_function(10/nhrds)

            # Export HydroDynamicData
            #-------------------------------------------------------------
            hrd.add_hydro_dynamic_data(
                resultdata=self.resulttable.loc[self.resulttable['HRDLocationId'].isin(group_location['HRDLocationId']), :],
                supportlocid=self.mainmodel.schematisation.support_location['HRDLocationId']
            )
            progress_function(30/nhrds)

            # # Check lengths
            # lenloc = len(resultdata.loc[idx])
            # if lenloc != len(loads):
            #     raise ValueError(f'Size of the loads do not match size of the result variables ({lenloc} and {len(loads)})')

            # Export UncertaintyModelFactor
            #-------------------------------------------------------------
            uncertainties = self.mainmodel.modeluncertainties.table.drop('Optie', axis=1).set_index('Naam')
            if hasattr(self.mainmodel.modeluncertainties, 'waterlevel_unc'):
                uncertainties['mu ZWL'] = self.mainmodel.modeluncertainties.waterlevel_unc.loc['ZWL', 'mu']
                uncertainties['sigma ZWL'] = self.mainmodel.modeluncertainties.waterlevel_unc.loc['ZWL', 'sigma']

            hrd.add_uncertainty_model_factor(group_location, uncertainties)
            progress_function(10/nhrds)

            if self.settings['export']['export_HLCD_and_config']:
                # Add calculation settings
                #-------------------------------------------------------------
                self.config_connections[i].add_numerical_settings(group_location)
            progress_function(10/nhrds)

        del self.hrd_connections[:]
        self.hlcd_connection = None
        del self.config_connections[:]
        
    def save_table(self):
        """
        Save table to file
        """
        self.export_dataframe.to_pickle(os.path.join(self.mainmodel.project.filedir, 'export_table.pkl'))

    def load_table(self):
        """
        Save table to file
        """
        path = os.path.join(self.mainmodel.project.filedir, 'export_table.pkl')
        if os.path.exists(path):
            imported_df = pd.read_pickle(path)
            # Drop content
            if not self.export_dataframe.empty:
                self.export_dataframe.drop(self.export_dataframe.index, inplace=True)
            # Add new content
            for col in imported_df.columns:
                self.export_dataframe[col] = imported_df[col]

class HarborSchematisation:

    def __init__(self, parent):
        self.mainmodel = parent

        self.breakwaters = datamodels.ExtendedGeoDataFrame(
            geotype=LineString, required_columns=['hoogte', 'alpha', 'beta', 'geometry'], required_types=[np.floating, np.floating, np.floating])

        self.harborarea = datamodels.ExtendedGeoDataFrame(
            geotype=Polygon, required_columns=['hoogte', 'geometry'], required_types=[np.floating])

        # Initialize geometries
        self.flooddefence = datamodels.ExtendedGeoDataFrame(geotype=LineString, required_columns=['traject-id', 'geometry'])
        self.inner = None
        self.buffered_inner = None
        self.entrance = None
        self.area_union = None

        self.support_locations = datamodels.ExtendedGeoDataFrame(geotype=Point, required_columns=['HRDLocationId', 'Name', 'XCoordinate', 'YCoordinate', 'geometry'])
        self.support_location = gpd.GeoSeries(index=['HRDLocationId', 'Name', 'XCoordinate', 'YCoordinate', 'geometry'])

        self.result_locations = datamodels.ExtendedGeoDataFrame(
            geotype=Point, columns=['Naam'], required_columns=['Normaal', 'geometry'], required_types=[(np.floating, np.integer)])


        self.settings = parent.project.settings

    def add_breakwater(self, path):
        """
        Add breakwater geometry to schemetisation

        Prerequisite is that the harbor area is loaded, since this is used to
        determine which end of the breakwater is the head.
        """
        if not np.size(self.harborarea):
            raise AttributeError('Harbor area is not loaded. Add this first, since it is used to determine the location of the head(s) of the breakwater(s).')
        if self.flooddefence.empty:
            raise AttributeError('Flood defence is not loaded. Add this first.')
        # Read file
        self.breakwaters.read_file(path)

        # Check if one breakwater end crosses the harbor area
        for idx, row in self.breakwaters.iterrows():
            ncrosses = [self.harborarea.contains(Point(row.geometry.coords[i])).any() for i in [0, -1]]
            if sum(ncrosses) == 0:
                raise ValueError('Een of meer van de havendammen kruist het haventerrein niet; zorg dat één uiteinde het haventerrein kruist.')
            elif sum(ncrosses) == 2:
                raise ValueError('Voor één of meer van de havendammen kruisen beide uiteinden het haventerrein; zorg dat één uiteinde het haventerrein kruist.')

        # Determine for all breakwaters the head location
        # these are determined as the point futhest away from the harbor area
        self.breakwaters['breakwaterhead'] = [Point(0, 0)] * len(self.breakwaters)
        for idx, row in self.breakwaters.iterrows():
            # Determine distance from both ends of breakwater to harbor area
            dists = {i: self.harborarea.distance(Point(row.geometry.coords[i])).min() for i in [0, -1]}
            # Add the furhtest away end as breakwaterhead
            self.breakwaters.at[idx, 'breakwaterhead'] = Point(row.geometry.coords[max(dists, key=dists.get)])
            self.breakwaters.at[idx, 'headindex'] = max(dists, key=dists.get)

        # We now have enough geometries to generate the harbor bounds, if there
        # are two breakwaters
        if len(self.breakwaters) == 2:
            self.generate_harbor_bound()
        # Set simple method on 'not finished'
        self.settings['simple']['finished'] = False
        # Save path to project structure
        self.settings['schematisation']['breakwater_shape'] = path


    def add_harborarea(self, path):
        """
        Add harbor area geometry to schematisation
        """
        # Check if flooddefence is chosen
        if self.flooddefence.empty:
            raise AttributeError('Er is nog geen normtraject toegevoegd.')

        # Read file
        self.harborarea.read_file(path)

        # # Check if the parts do not overlap
        # if len(self.harborarea) > 1:
        #     if not isinstance(_unary_union(self.harborarea['geometry'].values.tolist()), Polygon):
        #         raise ValueError("""Verschillende onderdelen van het haventerrein
        #                          vormen geen overlappend geheel. Zorg dat dit
        #                          wel het geval is""")

        # Check if all parts cross the flooddefence
        #TODO: ZOEK EEN STABIELE OPLOSSING VOOR DE LOSSE HAVENELEMENTEN DIE WEL LOS MOGEN LIGGEN, MAAR
        # DE GEOMETRIE NIET VERSTOREN
        # for area in self.harborarea.itertuples():
        #     if not self.flooddefence.intersects(area.geometry).any():
        #         raise ValueError('Het haventerrein overlapt niet met de waterkering. Dit is een vereiste.')

        # Add prepared geometries
        self.harborarea['prepgeo'] = [_prep(area['geometry']) for _, area in self.harborarea.iterrows()]

        # Calculate union of harbor area
        self.area_union = unary_union(self.harborarea['geometry'].values.tolist())

        # Set simple method on 'not finished'
        self.settings['simple']['finished'] = False

        # Save path to project structure
        self.settings['schematisation']['harbor_area_shape'] = path


    def add_flooddefence(self, section_id, datadir=None):
        """
        Add flood defence geometry to schematisation
        """
        # Use default datadir if not given
        if datadir is None:
            datadir = self.mainmodel.datadir

        # Load geometry
        geometry = io.geometry.import_section_geometry(section_id, datadir)
        if not geometry:
            return None

        # Add to settings
        if section_id not in self.settings['schematisation']['flooddefence_ids']:
            self.settings['schematisation']['flooddefence_ids'].append(section_id)

        # Add to geodataframe (first)
        logger.info(f'Adding flood defence section {section_id}')
        self.flooddefence.at[section_id, 'traject-id'] = section_id
        self.flooddefence.at[section_id, 'geometry'] = geometry

    def check_entrance_coordinate(self, crd):
        """
        Set the entrance coordinate, in case of one breakwater

        Parameters
        ----------
        crd : tuple
            Entrance coordinate which together with the breakwater head
            spans the entrance
        """
        if not np.size(self.breakwaters):
            raise AttributeError('Havendammen zijn nog niet ingeladen. Laad deze eerst in.')

        elif np.size(self.breakwaters) == 2:
            raise AttributeError('Er zijn twee havendammen aanwezig. Het is niet mogelijk een extra punt voor de hanveningang op te geven.')

        # Check the input types
        if not isinstance(crd, tuple):
            raise ValueError('Expected tuple.')
        for i in [0, 1]:
            if not isinstance(crd[i], (int, np.int, float, np.float)):
                raise ValueError('Expected integer of float inside the tuple: {}'.format(crd))

        # Check if the entrance coordinate is inside the harbor area
        if not self.harborarea.contains(Point(crd)).any():
            valid = False
        else:
            valid = True

        return valid

    def del_flooddefence(self, section_id):
        """
        Remove flood defence geometry from schematisation
        """
        if self.flooddefence.empty:
            return None

        # Remove from settings
        if section_id in self.settings['schematisation']['flooddefence_ids']:
            self.settings['schematisation']['flooddefence_ids'].remove(section_id)

        # Adjust dataframe
        trajectids = self.flooddefence['traject-id'].index.tolist()
        if section_id not in trajectids:
            return None

        self.flooddefence.drop(section_id, inplace=True)

    def add_support_locations(self, loc):
        """
        Add support location

        Parameters
        ----------
        loc : geopandas.GeoDataFrame
            dataframe with properties and geometry of support location
        """
        self.support_locations.delete_all()
        self.support_locations.set_data(loc)

    def add_result_locations(self, path):
        """
        Add result locations from a shapefile as geodataframe

        Parameters
        ----------
        path : str
            path to shapefile with locations. Shapefile only needs a geometry,
            a name (column "Naam") is generated if not present.
        """

        for naam, element in zip(
            ['Haventerrein', 'Havendammen', 'Waterkering'],
            [self.harborarea, self.breakwaters, self.flooddefence]
        ):
            if not np.size(element):
                raise AttributeError('{} is niet aanwezig. Deze moet aanwezig zijn voordat de uitvoerlocaties toegevoegd kunnen worden.'.format(naam))

        # Read file
        self.result_locations.read_file(path)

        # Add name if not present
        if self.result_locations['Naam'].isnull().all():
            self.result_locations['Naam'] = ['Locatie_{:04d}'.format(i) for i in range(1, len(self.result_locations) + 1)]

        # Add locations to model uncertainties
        self.mainmodel.modeluncertainties.add_result_locations()

        # Add locations to export
        self.mainmodel.export.add_result_locations()

        # Init pharos table
        logger.info('Trying to initialize pharos calculation')
        self.mainmodel.pharos.init_calculation_table()

        # Toegevoegd Svasek 04/10/18 - Initialiseer de Hares tabel
        # Init hares table
        logger.info('Trying to initialize hares calculation')
        self.mainmodel.hares.init_calculation_table()

        # Set simple method on 'not finished'
        self.settings['simple']['finished'] = False
        self.settings['schematisation']['result_locations_shape'] = path


    def set_selected_support_location(self, name):
        """
        Set selected support location by name. The location geometry is
        selected from the support locations (HRD) set.

        The fact that the support location is loaded implies that the hydraulic
        loads as well as the model uncertainties can also be loaded. This is
        thus done at this point, so a change will have a an effect on the
        steps to follow.

        Parameters
        ----------
        name : str
            name of the support location
        """

        if self.support_locations.empty:
            raise AttributeError('HRD-locaties moeten ingeladen zijn voordat er een streunpuntlocatie gekozen kan worden.')

        if name not in self.support_locations['Name'].values.tolist():
            raise ValueError('Naam van de steunpuntlocatie ("{}") komt niet voor in de HRD ({})'.format(name, self.mainmodel.project.settings['hydraulic_loads']['HRD']))

        self.support_location.loc[:] = self.support_locations.loc[self.support_locations['Name'].eq(name)].iloc[0]

        # Import hydraulic loads
        self.mainmodel.hydraulic_loads.load_hydraulic_loads()
        # On loading, the recalculated loads can be empty, recalculate here
        if self.settings['hydraulic_loads']['recalculate_waterlevels']:
            self.mainmodel.hydraulic_loads.calculate_waveconditions()
            self.mainmodel.hydraulic_loads.load_hydraulic_loads(interpolated=True)
        
        # At this point the iteration table for the advanced method is also loaded
        logger.info('Trying to initialize swan calculation')
        self.mainmodel.swan.iteration_results.initialize()

        # Init pharos table
        logger.info('Trying to initialize pharos calculation')
        self.mainmodel.pharos.init_calculation_table()

        # Toegevoegd Svasek 04/10/18 - Initialiseer de Hares tabel
        logger.info('Trying to initialize hares calculation')
        self.mainmodel.hares.init_calculation_table()

        # Import model uncertainties
        logger.info('Loading model uncertainties from support location.')
        self.mainmodel.modeluncertainties.load_modeluncertainties()

        # Set simple method on 'not finished'
        self.settings['simple']['finished'] = False

        # Change support location in settings
        self.settings['schematisation']['support_location_name'] = name


    def set_bedlevel(self, bedlevel):
        """
        Set the bed level and check if valid

        Parameters
        ----------
        bedlevel : float
            Representative bed level of the harbor basin

        """

        valid = self.check_bedlevel(bedlevel)

        if valid:
            self.settings['schematisation']['representative_bedlevel'] = bedlevel

        # Set simple method on 'not finished'
        self.settings['simple']['finished'] = False

    def check_bedlevel(self, bedlevel):
        """
        Check the consistency with the harbor area

        Parameters
        ----------
        bedlevel : float
            Representative bed level of the harbor basin

        """
        if not np.size(self.harborarea):
            raise ValueError('Haventerrein is nog niet ingeladen. Doe dit voor het opgeven van het bodemniveau.')

        if bedlevel > self.harborarea['hoogte'].min():
            return False
        else:
            return True

    def _check_locations(self, locations):
        """
        Function that removes locations which are not inside the basin

        Parameters
        ----------
        locations : geopandas.GeoDataFrame
            GeoDataFrame with locations
        """

        if not self.inner:
            self.generate_harbor_bound()

        dropids = [idx for idx, loc in locations.iterrows() if not self.innerpoly.contains(loc['geometry'])]

        if dropids:
            locations.drop(dropids, axis=0, inplace=True)

        return locations

    def _check_types(self, geodataframe, geometrytype):
        """
        Check for type of geometries
        """
        # Check for type of geometries
        for idx, row in geodataframe.iterrows():
            if not isinstance(row.geometry, geometrytype):
                raise TypeError('Geometrietype "{}" vereist. De ingevoerde shapefile heeft geometrietype "{}".'.format(
                    re.findall('([A-Z].*)\'', repr(geometrytype))[0],
                    re.findall('([A-Z].*)\'', repr(type(row.geometry)))[0],
                ))

    def generate_result_locations(self, distance, interval, interp_length):
        """
        Generate result locations on a certain offset.

        This method also determines on which side of the flooddefence the
        locations are generated. This is done by checking on which side of the
        flood defence line the support location is located. The side is checked
        on the nearest point.

        Parameters
        ----------
        distance : float or LineString
            offset of the locations from the flood defence line. Alternatively
            a LineString can be given on which the result locations are generated
        interval : float
            distance between the locations along the flood defence
        interp_length : float
            length of the levee used for determining the angle
        """

        # generate harbor entrance if it does not exist yet
        if not np.size(self.entrance):
            self.generate_harbor_bound()

        # Generate a respresentative point within the harbor bound
        pt = self.inner.representative_point()

        # Keep track of the location count per section
        location_count = {}
        sections = []
        section_ids = []

        # For each section the part along the harbor inner area needs to be found. For
        # this the inner bound is slightly buffered, after which the intersection with
        # the flood defence is calculated. This is shortened by the buffer, to get the
        # original length again.
        for _, section in self.flooddefence.iterrows():

            # Calculate the part of the section that intersects the harbor inner bound
            geo = section['geometry'].intersection(self.inner.buffer(0.1))

            # Potential empty geometries should be skipped
            if geo.is_empty:
                continue

            # Add each linestring in case of multilinestring
            elif isinstance(geo, MultiLineString):
                sections += [geometry.split_line_by_distance(linestring, [0.1, linestring.length-0.2])[1]
                             for linestring in geo]
                section_ids += [section['traject-id'] for linestring in geo]

            # Add single linestring in case of single geometry
            elif isinstance(geo, LineString):
                sections.append(geometry.split_line_by_distance(geo, [0.1, geo.length-0.2])[1])
                section_ids.append(section['traject-id'])

            # Raise error if geometry type does not match the expected types
            else:
                raise NotImplementedError(('Geometry type "{}" not understood. Will not generate result'
                                           'locations for this part of the flood defence'.format(type(geo))))

            # Set location count to zero for section
            location_count[section['traject-id']] = 0

        # For each flood defence
        for sectiongeo, sectionid in zip(sections, section_ids):
            # First determine on which side of the section the points should be
            # generated
            #-------------------------------------------------------------
            # Select nearest point of support point on flood defence lin
            nearest = sectiongeo.project(pt)
            # Select a tiny linesegment
            linesegment = [sectiongeo.interpolate(nearest - 0.01).coords[0],
                           sectiongeo.interpolate(nearest + 0.01).coords[0]]
            # Check side
            side = 'left' if geometry.is_left(pt.coords[0], linesegment) else 'right'

            # Now generate point along the flood defence
            #-------------------------------------------------------------

            if isinstance(distance, (float, int, np.int, np.float)):
                iterations = [side, 'left' if side == 'right' else 'right']
            else:
                # If a line is given, only check the line
                iterations = [side]

            # For both iterations
            for side in iterations:
                if isinstance(distance, (float, int, np.int, np.float)):
                    # if scalar, generate line
                    parallel = sectiongeo.parallel_offset(distance, side=side)
                    maxdist = distance * 1.5
                    # delete the parts the interfere with other sections
                    for section in sections:
                        if section == sectiongeo:
                            continue
                        parallel = parallel.difference(section.buffer(distance))
                else:
                    # if LineString, copy
                    parallel = distance
                    # Determine max distance
                    maxdist = gpd.GeoDataFrame(geometry=[Point(*crd) for crd in parallel.coords[:]]).distance(sectiongeo).max() * 1.5

                # Generate points
                points = self._generate_points(
                    sectiongeo,
                    parallel,
                    interval,
                    interp_length,
                    max_dist=maxdist
                )

                # Check if the points are inside the harbor.
                points = self._check_locations(points)
                # If any remain, dont try the other side
                if not points.empty:
                    break
            if points.empty:
                continue

            # Generate names
            points['Naam'] = ['{}_{:05d}'.format(sectionid, i) for i in np.arange(len(points)) + location_count[sectionid] + 1]

            # Add to existing dataframe
            self.result_locations.set_data(points)
            
            # Add location count for generating names when sections have multiple line segments
            location_count[sectionid] += len(points)

        if self.result_locations.empty:
            raise ValueError('No result locations where generated. Check your harbor schematisation')

    def _generate_points(self, sectiongeo, parallel, interval, interp_length, max_dist=1000000):
        """
        Method to generate the points from a line based on an angle

        sectiongeo : shapely.geometry.LineString
            line of the flood defence
        parallel : shapely.geometry.LineString
            line along the flood defence on which the locations are generated
        interval : float
            distance between the locations along the flood defence
        interp_length : float
            length of the levee used for determining the angle
        """
        pts = {}
        n = 0

        # Generate direction of line
        line = np.vstack(sectiongeo.coords)
        # cumulative length of the line segments
        dists = np.r_[0, np.cumsum(np.hypot(*np.diff(line, axis=0).T))]
        # slope (complex)
        dxdy = [complex(*segment) for segment in np.diff(line, axis=0)]
        # angle
        angles = np.angle(dxdy)

        rem = sectiongeo.length % interval
        for ptdist in np.arange(0.5 * rem, sectiongeo.length, interval):
            # Find out the weight factor of each segment
            factors = np.diff(np.interp(dists, [ptdist - 0.5 * interp_length, ptdist + 0.5 * interp_length], [0, 1]))
            # Calculate average angle
            angle = geometry.average_angle(angles[(factors > 0.0)], factors[(factors > 0.0)]) + 0.5 * np.pi
            # Create line from midpoint
            origin = sectiongeo.interpolate(ptdist)

            line = geometry.extend_point_to_linestring(
                origin,
                geometry.car2nau(np.degrees(angle)),
                (-max_dist, max_dist),
                as_LineString=True
            )
            # Find intersection of line with parallel
            isect = geometry.find_nearest_intersection(parallel, line, origin)

            # If intersection, add to dict
            if isect:
                # Caculate angle with north
                origin = origin.coords[0]
                angle = np.degrees(geometry.calculate_angle(
                    a=(origin[0], origin[1] + 1),
                    b=origin,
                    c=isect.coords[0])
                )
                if isect.coords[0][0] < origin[0]:
                    angle = (-1 * angle) % 360
                # Add to dictionairy
                n += 1
                pts[n] = {
                    'geometry': isect,
                    'Naam': '',
                    'Normaal': angle
                }

        return gpd.GeoDataFrame.from_dict(pts, orient='index')


    def generate_harbor_bound(self):
        """
        Generates the interior line of the harbor.

        For multiple purposes HB Havens needs to know what is inside and what
        is outside the harbor.

        Steps to get the harbor inner area:
            1.  Determine the harbor entrance. For a harbor with two break-
                waters this is the line between the two breakwater heads. For a
                harbor with one breakwater it is the line from the breakwater
                head to the given entrance coordinate

            2.  Create a line collection of the entrance, breakwater(s) and
                the harbor area union exteriors.

            3.  Polygonize the line collection.

            4.  Clip by the flood defence line

        """

        logger.info('Generating harbor bound')
        # 1.  Determine harbor entrance
        # Generate line trough harbor entrance
        if len(self.breakwaters) == 2:
            pt1, pt2 = self.breakwaters['breakwaterhead'].values.tolist()

        elif len(self.breakwaters) == 1:
            breakwater = self.breakwaters.iloc[0]
            # The first point is the breakwaterhead
            pt1 = breakwater['breakwaterhead']
            if not hasattr(self, 'entrance_coordinate'):
                raise ValueError('Extra entrance coordinate is not known.')
            pt2 = self.entrance_coordinate

        # Create entrance as linestring
        self.entrance = LineString([pt1, pt2])

        # Delete overlapping part with harborarea in case of 1 breakwater
        if len(self.breakwaters) == 1:
            self.entrance = self.entrance.difference(self.area_union)

        # 2.  Merge lines.  The entrance is slightly buffered to force an
        # intersection with the harbor area in case of one breakwater

        # get harbor area lines
        if isinstance(self.area_union, Polygon):
            area_exteriors = [self.area_union.exterior]
        elif isinstance(self.area_union, MultiPolygon):
            area_exteriors = [area.exterior for area in self.area_union]

        # Get the snapped flood defence line
        # Convert to lists of linestrings
        flooddefencegeos = list(chain(*map(geometry.as_linestring_list, self.flooddefence['geometry'])))
        for i, geo in enumerate(flooddefencegeos):
            # Break rings if present
            if geo.is_ring:
                crds = np.vstack(geo.coords[:])
                dists = np.hypot(*(crds - np.array(self.entrance.centroid)[None, :]).T)
                flooddefencegeos[i] = geometry.as_linestring_list(geo.difference(Point(*crds[np.argmax(dists)]).buffer(1.0)))            

        # Get coordinates
        flooddefencegeos = [np.vstack(line.coords[:]) for line in geometry.as_linestring_list(flooddefencegeos)]
            
        # Snap all lines
        snapped_fd_lines = [LineString(line) for line in geometry.snap_flooddefence_lines(
            lines=flooddefencegeos, max_snap_dist=self.settings['schematisation']['fd_snap_distance'])]

        lines = [self.entrance.buffer(0.01).exterior] + area_exteriors + self.breakwaters['geometry'].values.tolist() + snapped_fd_lines

        # 3.  Get inner polygon from lines
        polygons = [p for p in polygonize(unary_union(linemerge(lines)))]

        if len(polygons) == 0:
            raise ValueError('Polygonizing lines did not succeed, check your geometries.')
        polygon = unary_union(polygons)

        # 4.  Clip the polygon with the flood defence line.
        correct_side_pt = polygon.difference(self.area_union.buffer(0.1)).representative_point()

        lines = [poly.exterior for poly in geometry.multiple_polygons(polygon)] + snapped_fd_lines
        polygons = [p for p in polygonize(unary_union(linemerge(lines)))]

        self.inner = [polygon for polygon in polygons if polygon.intersects(correct_side_pt)][0]

        # Convert to other geometry types
        self.innerbound = LineString(self.inner.exterior.coords[:])
        self.innerpoly = _prep(Polygon(self.inner))
        self.buffered_inner = _prep(self.inner.buffer(-1))

    def is_reachable(self, point, wavedir):
        """
        Method to check if a point can be reached from a (nautical) direction, given the
        harbor geometry. A point is extended in the direction the wave
        is coming from. Since the wavedirection if accepted in
        nautical convention, this should be inverted.
        If this line intersects the inner part of the harbor, the
        point cannot be reached.

        Parameters
        ----------
        point : tuple
            Coordinate for which is checked if it can be reached
        wavedir : float
            Wave direction (nautical)
        """
        line = geometry.extend_point_to_linestring(
            pt=point,
            direction=(wavedir - 180) % 360,
            extend=(100000., 0.),
            as_LineString=True
        )

        return not self.buffered_inner.intersects(line)

    def plot_schematisation(self, buffer=200, locations=True, figsize=(15,15)):
        """
        Method to plot the schematisation. Primarily used for testing
        and debugging purposes.
        """

        # Create figure and set aspect
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_aspect(1.0)

        # plot harbor area
        self.harborarea.plot(ax=ax, zorder=2, alpha=0.5)

        # Plot breakwaters. Add row id at head
        for breakwater in self.breakwaters.itertuples():
            ax.plot(*breakwater.geometry.xy, color='C1')
            ax.text(*breakwater.breakwaterhead.coords[0], breakwater.Index)

        if locations:
            # Plot result locations with names
            self.result_locations.plot(ax=ax, color='k', marker='.', zorder=10)
            for _, loc in self.result_locations.iterrows():
                ax.text(loc['geometry'].coords[0][0], loc['geometry'].coords[0][1], loc['Naam'])

        # Set limits
        xlim = ax.get_xlim()
        ax.set_xlim(xmin=xlim[0] - buffer, xmax=xlim[1] + buffer)
        ylim = ax.get_ylim()
        ax.set_ylim(ymin=ylim[0] - buffer, ymax=ylim[1] + buffer)

        # Plot flood defence
        self.flooddefence.plot(ax=ax, color='C3')

        # Mark harbor inner area
        ax.add_patch(PolygonPatch(self.inner, color='0.90'))

        return fig, ax

    def get_harbor_bounds(self):
        """
        Method to get the harbor bounds based on loaded geometries.
        This function can be used to determine a plotting extent
        """
        # if geometries are already loaded, zoom to harbor extent
        if self.inner is not None:
            bounds = self.inner.bounds
            bounds = self.inner.buffer((bounds[2] - bounds[0]) * 0.3).bounds
        elif self.area_union is not None:
            bounds = self.area_union.bounds
            bounds = self.area_union.buffer((bounds[2] - bounds[0]) * 0.3).bounds
        else:
            bounds = None

        return bounds

class InterpolationTable:

    def __init__(self, path):
        """
        Diffraction Table class. Facilitates methods to check boundaries
        and carry out interpolation.

        Parameters
        ----------
        path : str
            Path of the csv-file
        """
        # Read table from csv
        data = pd.read_csv(path, sep=',', index_col=[0])

        # Add values
        self.values = data.values[:]
        # Add x-indices, and invert if necessary
        self.XL = data.columns.values.astype(float)
        if (self.XL[-1] < self.XL[0]):
            self.XL = self.XL[::-1]
            self.values = self.values[:, ::-1]
        # Add y-indices, and invert if necessary
        self.YL = data.index.values.astype(float)
        if (self.YL[-1] < self.YL[0]):
            self.YL = self.YL[::-1]
            self.values = self.values[::-1, :]

        # Create interpolation_function
        self._init_interp2d()

    def _init_interp2d(self):
        """
        Initialize 2D-interpolation by creating a scipy.interpolate function
        """
        # Extend om extrapolatie tegen te gaan
        x = np.r_[self.XL.min() - 1, self.XL, self.XL.max() + 1]
        y = np.r_[self.YL.min() - 1, self.YL, self.YL.max() + 1]
        vals = np.zeros((len(y), len(x)))
        vals[1:-1, 1:-1] = self.values
        vals[1:-1, 0] = self.values[:, 0]
        vals[1:-1, -1] = self.values[:, -1]
        vals[0, :] = vals[1, :]
        vals[-1, :] = vals[-2, :]
        # Add to interpolation function
        self.f = interp2d(x, y, vals)


    def check_range(self, X, Y, Lr):
        """
        Class method to check if a value is within the table range

        Parameters
        ----------
        X : float
            Distance from wave direction to point

        Y : float
            Distance from X-axis to origin

        Lr : float
            Represenative wave length

        Returns
        -------
        inside : boolean
            Whether the given (X/L,Y/L) is within the table range
        """

        inside = True
        # Check x-domain
        if (self.XL[0] > X / Lr) or (self.XL[-1] < X / Lr):
            inside = False
        # Check y-domain
        if (self.YL[0] > Y / Lr) or (self.YL[-1] < Y / Lr):
            inside = False

        return inside

    def interpolate(self, X, Y, Lr):
        """
        Class method to interpolate values in table

        Parameters
        ----------
        X : float
            Distance from wave direction to point

        Y : float
            Distance from X-axis to origin

        Lr : float
            Represenative wave length

        Returns
        -------
        Kd : float
            Diffraction coefficient
        """
        #TODO: Treat boundaries

        return self.f(X / Lr, Y / Lr)[0]
        # Interpolate over first axis
#        Yvals = [np.interp(X/Lr, self.XL, row) for row in self.values]
        # Interpolate over second axis
#        Kd = np.interp(Y/Lr, self.YL, Yvals)

class Project():
    """
    description of class
    """

    def __init__(self, parent):

        self.mainmodel = parent

        self.name = None
        self.created = None
        self.lastModified = None
        self.dirty = False

        self.default_settings = {
            'project': {
                'software': {
                    'name': self.mainmodel.appName,
                    'version': self.mainmodel.appVersion,
                    'date': self.mainmodel.appDate,
                },
                'name': '',
                'user': {
                    'name': '',
                    'email': '',
                },
                'created': {
                    'date': '',
                    'time': '',
                },
                'last_modified': {
                    'date': '',
                    'time': '',
                },
                'progress': 0,
            },
            'schematisation': {
                'harbor_area_shape': '',
                'breakwater_shape': '',
                'entrance_coordinate': '',
                'flooddefence_ids': [],
                'fd_snap_distance': 10,
                'representative_bedlevel' : '',
                'support_location_name': '',
                'result_locations_shape': '',
            },

            'hydraulic_loads': {
                'HRD' : '',
                'recalculate_waterlevels': False,
                'recalculate_method': 'regression',
                'waterlevels': [],
            },

            'calculation_method': {
                'method': 'general',
                'condition_geometry': True,
                'condition_reflection': False,
                'condition_flow': False,
                'motivation': '',
                'include_pharos': True,
                'include_hares': False
            },

            'simple': {
                'finished': False,
                'processes': [
                    'Diffractie',
                    'Transmissie',
                    'Lokale golfgroei'
                ]
            },
            'swan': {
                'mastertemplate': '',
                'depthfile': '',
                'swanfolder': '',
                'factor Tm Tp': 1.1,
                'use_incoming_wave_factors': True,
                'calculations': {
                    'I1': {
                        'folder' : '',
                        'input_generated': False
                    },
                    'I2': {
                        'folder' : '',
                        'input_generated': False
                    },
                    'I3': {
                        'folder' : '',
                        'input_generated': False
                    },
                    'D': {
                        'folder' : '',
                        'input_generated': False
                    },
                    'TR': {
                        'folder' : '',
                        'input_generated': False
                    },
                    'W': {
                        'folder' : '',
                        'input_generated': False
                    }
                }
            },
            'pharos': {
                'initialized': False,
                'input_generated': False,
                'use_incoming_wave_factors': True,
                'hydraulic loads': {
                    'factor Tm Tp': 1.1,
                    'water depth for wave length': np.nan
                },
                'wave directions': {
                    'lowest': 0,
                    'highest': 360,
                    'bin size': 5
                },
                'frequencies': {
                    'lowest': np.nan,
                    'highest': np.nan,
                    'number of bins': 20,
                    'scale': 'lineair',
                    'checked': []
                },
                '2d wave spectrum': {
                    'spread': 10.0,
                    'gamma': 3.3,
                    'min energy': 0.01
                },
                'paths': {
                    'pharos folder': '',
                    'schematisation folder': ''
                },
                'water levels': {
                    'checked': []
                },
                'transformation': {
                    'dx': np.nan,
                    'dy': np.nan
                },
                'schematisations': {}
            },
            # Toegevoegd Svasek 04/10/18 - Initialiseer de Hares settings
            'hares': {
                'initialized': True,
                'input_generated': True,
                'use_incoming_wave_factors': True,
                'hydraulic loads': {
                    'factor Tm Tp': 1.1,
                    'water depth for wave length': np.nan
                },
                'wave directions': {
                    'lowest': 0,
                    'highest': 360,
                    'bin size': 5
                },
                'frequencies': {
                    'lowest': np.nan,
                    'highest': np.nan,
                    'number of bins': 20,
                    'scale': 'lineair',
                    'checked': []
                },
                '2d wave spectrum': {
                    'spread': 10.0,
                    'gamma': 3.3,
                    'min energy': 0.01
                },
                'paths': {
                    'hares folder': '',
                    'schematisation folder': ''
                },
                'water levels': {
                    'checked': []
                },
                'transformation': {
                    'dx': np.nan,
                    'dy': np.nan
                },
                'schematisations': {}
            },
            'export' : {
                'export_HLCD_and_config': True,
                'HLCD': '',
                'export_succeeded': False,
            }
        }

        self.initSettings()

    def initSettings(self):
        """
        Initialize project settings

        Parameters
        ----------
        appName : string
            Name of the application
        appVersion : string
            Version of the application
        appDate : string
            Date of the application
        """
        self.settings = self.default_settings.copy()

    def getGroupSettings(self, group):
        """
        Get group from application settings
        """
        if group in self.settings:
            return self.settings[group]
        else:
            raise KeyError('Key "{}" not present in settings'.format(group))

    def open_from_file(self, fname):
        """
        Open project file
        """
        # Load settings from json
        with open(fname, 'r') as data_file:
            loaded_settings = json.load(data_file)

        self.check_settings(loaded_settings, self.default_settings)

        # Replace settings with loaded settings.
        # Do not overwrite, since we want the identifier to remain equal!
        for key, val in loaded_settings.items():
            self.settings[key] = val

        self.name = fname
        self.filedir = os.path.splitext(fname)[0]

        self.mainmodel.load_from_project()


    def check_settings(self, to_check, template):
        """
        Method to check if all settings are present. If not, extend the settings
        with the default parameters
        """

        # For each dictionary item
        for key, val in template.items():
            # If the value is not a dictionary, check with template
            if not isinstance(val, dict):
                if key not in to_check.keys():
                    to_check[key] = val
            # If the value is a dictionary, go a level deeper
            else:
                if key not in to_check.keys():
                    to_check[key] = {}
                self.check_settings(to_check[key], template[key])

    def save(self):
        """
        Save project
        """
        self.updateLastModified()
        with open(self.name, 'w') as outfile:
            json.dump(self.settings, outfile, indent=2)

        # Create a directory to save the file
        if not os.path.exists(self.filedir):
            os.mkdir(self.filedir)
        elif not os.path.isdir(self.filedir):
            os.mkdir(self.filedir)

    def save_as(self, fname, overwrite_all=False):
        """
        Save project as
        """
        # Set project name
        self.name = fname
        self.filedir = os.path.splitext(fname)[0]

        # Set project dirty
        self.dirty = True

        # Delete the directory if it already exists
        if os.path.exists(self.filedir):

            if os.path.isdir(self.filedir) and overwrite_all:
                shutil.rmtree(self.filedir, ignore_errors=True)

            elif os.path.isdir(self.filedir) and any(os.listdir(self.filedir)):
                raise OSError((
                    f'De map "{self.filedir}" is nodig voor het opslaan van bestanden,'
                    'maar deze bestaat al. Leeg of verwijder de map, of kies een andere naam om onder'
                    'op te slaan.'))

        self.setCreated()
        self.save()


    def updateLastModified(self):
        now = time
        lastModified = {}
        lastModified['date'] = now.strftime("%d-%b-%Y")
        lastModified['time'] = now.strftime("%H:%M:%S")

        self.settings['project']['last_modified'] = lastModified

    def setCreated(self):
        now = time
        created = {}
        created['date'] = now.strftime("%d-%b-%Y")
        created['time'] = now.strftime("%H:%M:%S")

        self.settings['project']['created'] = created
