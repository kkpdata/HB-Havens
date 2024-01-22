# -*- coding: utf-8 -*-
"""
Created on  : Fri Aug 11 11:33:42 2017
Author      : Guus Rongen, HKV Lijn in Water
Project     : HB Havens
Description : General classes for HB Havens
    
"""

import numpy as np
import pandas as pd
from scipy.interpolate import interp2d as _interp2d
    
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
        x = np.r_[self.XL.min()-1, self.XL, self.XL.max()+1]
        y = np.r_[self.YL.min()-1, self.YL, self.YL.max()+1]
        vals = np.zeros((len(y), len(x)))
        vals[1:-1, 1:-1] = self.values
        vals[1:-1, 0] = self.values[:, 0]
        vals[1:-1,-1] = self.values[:,-1]
        vals[0,  :] = vals[1, :]
        vals[-1, :] = vals[-2, :]
        # Add to interpolation function
        self.f = _interp2d(x, y, vals)
        
        
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
        if (self.XL[0] > X/Lr) or (self.XL[-1] < X/Lr):
            inside = False
        # Check y-domain
        if (self.YL[0] > Y/Lr) or (self.YL[-1] < Y/Lr):
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
        
        return self.f(X/Lr, Y/Lr)
        
def replace_column(dataframe, in_tag, out_tag, dictionary):
    """
    Function to replace hydraulic load id for description in dataframe
    """

    # Replace when in columns
    if in_tag in dataframe.columns:
        dataframe[in_tag] = list(map(dictionary.get, dataframe[in_tag]))
        dataframe.columns = [out_tag if col == in_tag else col for col in dataframe.columns]

    # Replace in index
    elif in_tag in dataframe.index.names:
        # Get new values
        descriptions = list(map(dictionary.get, dataframe.index.get_level_values(in_tag).tolist()))
        # Create new set of arrays for index
        index_arrays = [descriptions if name == in_tag else dataframe.index.get_level_values(name) for name in dataframe.index.names]
        # Create new index
        dataframe.index = pd.MultiIndex.from_arrays(
            index_arrays,
            names=[out_tag if name == in_tag else name for name in dataframe.index.names]
        )
        
    else:
        raise KeyError('Did not found {} in columns ({}) or index names ({}).'.format(
            in_tag, ', '.join(dataframe.columns.tolist()), ', '.join(dataframe.index.names)))
