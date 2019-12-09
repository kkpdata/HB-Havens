# Toegevoegd Svasek 03/10/18 - Hele nieuwe Class toegevoegd
import itertools
import json
import logging
import os
import re
from math import gamma, isclose

import numpy as np
import pandas as pd
from scipy.interpolate import interpolate
from scipy.optimize import bisect, newton

from hbhavens import io
from hbhavens.core.geometry import average_angle
from hbhavens.core.spectrum import (JONSWAPSpectrum, Spectrum2D,
                                    incoming_wave_factors, jonswap, jonswap_2d)

logger = logging.getLogger(__name__)


g = 9.81

class Hares:

    def __init__(self, parent):
        """
        Class Hares

        Parameters
        ----------
        parent : MainModel
            Pointer to the mainmodel object
        """
        self.mainmodel = parent
        self.supportlocation = self.mainmodel.schematisation.support_locations
        self.hydraulic_loads = self.mainmodel.hydraulic_loads

        # Add HaresIO class
        self.haresio = io.hares.HaresIO(self)

        # Give the name of the result parameters for export
        self.result_parameters = {
                        'h' : 'h',
                        'Hs' : 'Hs totaal',
                        'Tp' : 'Tp totaal',
                        'Tm-1,0' : 'Tm-1,0 totaal',
                        'Wave direction': 'Wave direction totaal'
                    }	
		
        # Get all settings
        self.settings = parent.project.settings
        
        # Initialize tables
        self.calculation_results = pd.DataFrame()

        # Empty objects
        self.grid_names = {}
        self.project_names = {}
        self.output_location_indices = {}

    def init_calculation_table(self):
        """
        Initialize table with calculation results for each combination of
        hydraulic loads and result locations.
        """
        # Determine individual and multiindex
        if not np.size(self.mainmodel.schematisation.result_locations):
            return None

        if self.hydraulic_loads.empty:
            return None

        # Get location names and id's
        self.result_locations = self.mainmodel.schematisation.result_locations
        self.nlocations = len(self.result_locations)
        
        # # Get load ids
        self.hydraulicloadids = self.hydraulic_loads.index.astype(int)
        self.nloads = len(self.hydraulicloadids)

        # Empty calculation results, in case it needs to be refilled
        if not self.calculation_results.empty:
            self.calculation_results.iloc[:, 0] = np.nan
            self.calculation_results.dropna(inplace=True)
            self.calculation_results.drop(self.calculation_results.columns, axis=1, inplace=True)
        
        # Add index and columns to table
        self.calculation_results['Location'] = np.repeat(self.result_locations['Naam'], self.nloads)
        self.calculation_results['HydraulicLoadId'] = np.tile(self.hydraulicloadids, self.nlocations)

        # Get load parameters
        load_parameters = self.hydraulic_loads.columns.intersection(['h', 'Hs', 'Tp', 'Tm-1,0', 'Wave direction', 'Wind direction', 'Wind speed', 'Water level'])
        for col in load_parameters:
            self.calculation_results[col] = np.nan
        
        # Add values from iterations to calculation results
        tiled = np.tile(self.hydraulic_loads[load_parameters].values, (self.nlocations , 1))
        self.calculation_results[load_parameters] = tiled
        self.calculation_results.reset_index(inplace=True, drop=True)

        # Add geometry
        result_locations = self.mainmodel.schematisation.result_locations
        result_locations = result_locations.reindex(columns=result_locations.columns.tolist()+list('XY'))
        result_locations[['X',  'Y']] = np.vstack([pt.coords[0] for pt in result_locations['geometry']])

        for col in ['X',  'Y', 'Normaal']:
            self.calculation_results[col] = np.nan

        self.calculation_results[['X',  'Y', 'Normaal']] = self.calculation_results[['Location']].merge(
            result_locations[['X',  'Y', 'Normaal', 'Naam']].round(3),
            left_on='Location',
            right_on='Naam'
        ).drop(['Naam', 'Location'], axis=1).values
        
        # # Add columns for result
        self.swan_columns = ['Hm0 swan', 'Tm-1,0 swan', 'Tp swan', 'Wave direction swan']
        self.hares_columns = ['Hs hares', 'Tp hares', 'Tm-1,0 hares', 'Wave direction hares']

        for col in self.hares_columns + self.swan_columns:
            self.calculation_results[col] = np.nan
        
        # Add load combination description
        self.calculation_results['Load combination'] = list(map(
            self.mainmodel.hydraulic_loads.description_dict.get,
            self.calculation_results['HydraulicLoadId']
        ))
        

    def load_from_project(self):
        """
        Load Hares class (advanced calculation) from project
        First some often 
        """
        # Load calculation table if it exists
        calculation_results_path = os.path.join(self.mainmodel.project.filedir, 'hares_calculation_results.pkl')
        if os.path.exists(calculation_results_path):
            logger.info("..Loaded HARES calculation table")
            self.calculation_results = pd.read_pickle(calculation_results_path)

    def save_tables(self):
        """
        Method to save spectrum and calculation table to pickle
        """
        if hasattr(self, 'calculation_results'):
            self.calculation_results.to_pickle(os.path.join(self.mainmodel.project.filedir, 'hares_calculation_results.pkl'))


            
    def read_calculation_results(self, progress_function=None):
        """
        Method to read HARES csv postprocessed output
        """
        files = os.listdir(self.settings['hares']['hares folder'])
        
        loads = self.calculation_results['Load combination'].unique()
        for file in files:
            if file[:-4] not in loads:
                continue
            
            df = pd.read_csv(os.path.join(self.settings['hares']['hares folder'], file), na_values='NaN')

            # Find location of load combination in calculation results
            indices = self.calculation_results['Load combination'].eq(file[:-4])
            
            # For all locations with the load combination
            for row in self.calculation_results.loc[indices].itertuples():
                
                # Match the location with the HARES result location
                index = df.index[df['Location'].eq(row.Location).array]
                if len(index) > 1:
                    raise ValueError('Locations appears multiple times in HARES results.')
                if len(index) == 0:
                    continue
                index = index[0]

                self.calculation_results.at[row.Index, 'Hs hares'] = df.at[index, 'Hs'] 
                self.calculation_results.at[row.Index, 'Tp hares'] = df.at[index, 'Tp']
                self.calculation_results.at[row.Index, 'Tm-1,0 hares'] = df.at[index, 'Tm-10']
                self.calculation_results.at[row.Index, 'Wave direction hares'] = df.at[index, 'Wave direction']
                    
                if progress_function is not None:
                    progress_function(1)

        self.calculation_results[self.hares_columns] = self.calculation_results[self.hares_columns].round(3)

        self.combine_with_swan()

    def combine_with_swan(self):
        """
        Class to combine calculation results with swan output.
        Prerequisite is that the swan results are filled.
        """

        # Copy results from calculation results
        merge_columns = ['Location', 'HydraulicLoadId']
        
        # Merge with swan final results
        swan_results = self.mainmodel.swan.calculation_results
        if np.isnan(swan_results[self.swan_columns].values).any():
            raise ValueError('NaN values in SWAN-data.')
        
        # Merge (use values because of index in merging)
        self.calculation_results[self.swan_columns] = self.calculation_results[merge_columns].merge(
            swan_results[self.swan_columns + merge_columns],
            on=merge_columns
        ).drop(merge_columns, axis=1).values
        
        if np.isnan(self.calculation_results[self.swan_columns + self.hares_columns].values).any():
            raise ValueError('NaN values in combined SWAN and HARES data. This indicates an error in merging the results from both models.')

        # Calculate combined energies
        f_swan, f_hares = (self.calculation_results[['Hm0 swan', 'Hs hares']].fillna(0.0) ** 2).values.T
        
        # Calculate combined significant wave height
        self.calculation_results['Hs totaal'] = np.hypot(*self.calculation_results[['Hm0 swan', 'Hs hares']].values.T)

        # Calculate combined Tm -1,0
        f_total = np.sum([f_swan, f_hares], axis=0)
        idx = f_total > 0
        self.calculation_results.loc[~idx, 'Tm-1,0 totaal'] = 0.0
        self.calculation_results.loc[idx, 'Tm-1,0 totaal'] = (self.calculation_results.loc[idx, 'Tm-1,0 swan'].fillna(0.0) * f_swan[idx] + 
                                                              self.calculation_results.loc[idx, 'Tm-1,0 hares'].fillna(0.0) * f_hares[idx]) / f_total[idx]

        # Calculate combined Tp
        self.calculation_results['Tp totaal'] = np.round(np.max(self.calculation_results[['Tp swan', 'Tp hares']].fillna(0.0), axis=1), 3)
        
        # Calculate combined wave direction
        self.calculation_results['Wave direction totaal'] = average_angle(
            angles=self.calculation_results[['Wave direction swan', 'Wave direction hares']].values,
            factors=np.vstack([f_swan, f_hares]).T,
            degrees=True
        )

        # Round to three decimals
        combined_columns = ['Hs totaal', 'Tm-1,0 totaal', 'Tp totaal', 'Wave direction totaal']
        self.calculation_results[combined_columns] = self.calculation_results[combined_columns].round(3)

        if self.calculation_results.isnull().any().any():
            nancols = ', '.join(list(self.calculation_results.columns.array[self.calculation_results.isnull().any(axis=0).values]))
            raise ValueError(f'NaN values in HARES-SWAN combined results. In columns: {nancols}')
        
