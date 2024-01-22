# -*- coding: utf-8 -*-
"""
Created on  : Tue Jul 11 11:35:49 2017
Author      : Guus Rongen
Project     : PR3594.10.00
Description :

"""

import os

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString, MultiLineString

from hbhavens import core
from hbhavens.core.geometry import is_left, as_linestring_list
from hbhavens.core.datamodels import ExtendedDataFrame

# Global constants:
global g
g = 9.81


class SimpleCalculation:

    def __init__(self, parent):
        self.mainmodel = parent
        self.results_changed = True

        self.result_parameters = {
            'h' : 'h',
            'Hs' : 'Hs,out',
            'Tp' : 'Tp',
            'Tm-1,0' : 'Tm-1,0',
            'Wave direction': 'Combined wave direction',
            'Storage situation VZM': 'Storage situation VZM'
        }

        # Initialize simple calculation classes
        self.diffraction = Diffraction(self)
        self.transmission = Transmission(self)
        self.wavegrowth = LocalWaveGrowth(self)
        self.wavebreaking = WaveBreaking(self)
        self.combinedresults = CombineResults(self)


    def initialize(self):
        """
        Initialize the simple calculation. Can be done only after
        the schematisation is complete and the choice for the simple
        method has been made
        """
        self.diffraction.initialize()
        self.transmission.initialize()
        self.wavegrowth.initialize()
        self.wavebreaking.initialize()


    def run_all(self, processes=None, progress_function=None):
        """
        Run all classes

        Parameters
        ----------
        progressclass : class
            Thread that can handle the progress (not implemented)
        """
        if processes is None:
            processes = ['Diffractie', 'Transmissie', 'Lokale golfgroei', 'Golfbreking']

        if progress_function:
            progress_function(1, 'Initialiseren...')
        self.initialize()


        if 'Diffractie' in processes:
            if progress_function:
                progress_function(1, 'Berekenen van diffractie...')
            self.diffraction.run()

        if 'Transmissie' in processes:
            if progress_function:
                progress_function(1, 'Berekenen van transmissie...')
            self.transmission.run()

        if 'Lokale golfgroei' in processes:
            if progress_function:
                progress_function(1, 'Berekenen van lokale golfgroei...')
            self.wavegrowth.run(progress_function=progress_function)

        if 'Golfbreking' in processes:
            if progress_function:
                progress_function(1, 'Berekenen van golfbreking...')
            self.wavebreaking.run()

        # Combine
        if progress_function:
            progress_function(1, 'Combineren van resultaten...')
        self.combinedresults.run(processes)

        self.mainmodel.project.settings['simple']['finished'] = True

        if progress_function:
            progress_function(1, 'Berekening voltooid.')


    def save_results(self):
        """
        Save results to pickle
        """
        directory = self.mainmodel.project.filedir
        processes = self.mainmodel.project.settings['simple']['processes']

        if 'Diffractie' in processes:
            self.diffraction.output.to_pickle(os.path.join(directory, 'simple_diffraction.pkl'))
        if 'Transmissie' in processes:
            self.transmission.output.to_pickle(os.path.join(directory, 'simple_transmission.pkl'))
        if 'Lokale golfgroei' in processes:
            self.wavegrowth.output.to_pickle(os.path.join(directory, 'simple_wavegrowth.pkl'))
        if 'Golfbreking' in processes:
            self.wavebreaking.output.to_pickle(os.path.join(directory, 'simple_wavebreaking.pkl'))
        self.combinedresults.output.to_pickle(os.path.join(directory, 'simple_combinedresults.pkl'))

    def load_results(self):
        """
        Load results from pickle
        """

        directory = self.mainmodel.project.filedir

        if not os.path.exists(os.path.join(directory, 'simple_combinedresults.pkl')):
            raise OSError('Simple method results not found in "{}"'.format(directory))

        self.initialize()

        path = os.path.join(directory, 'simple_diffraction.pkl')
        if os.path.exists(path):
            self.diffraction.output.load_pickle(path)

        path = os.path.join(directory, 'simple_transmission.pkl')
        if os.path.exists(path):
            self.transmission.output.load_pickle(path)

        path = os.path.join(directory, 'simple_wavegrowth.pkl')
        if os.path.exists(path):
            self.wavegrowth.output.load_pickle(path)

        path = os.path.join(directory, 'simple_wavebreaking.pkl')
        if os.path.exists(path):
            self.wavebreaking.output.load_pickle(path)

        self.combinedresults.output.load_pickle(os.path.join(directory, 'simple_combinedresults.pkl'))



class Diffraction:

    def __init__(self, simple_calculation):
        """
        Function to calculate wave reduction due to diffraction. The wave
        reduction is calculated based on the diffraction tables given by
        Goda (1978, 2000), which can be used for one or two breakwaters.

        Parameters
        ----------
        simple_calculation : hbhavens.core.models.SimpleCalculation
            calculation class which connects to all other models via
            MainModel

        Returns
        -------
        Kd : float
            reduction factor for diffraction
        """

        # Create always existing links
        self.mainmodel = simple_calculation.mainmodel
        self.schematisation = self.mainmodel.schematisation
        self.hydraulic_loads = self.mainmodel.hydraulic_loads
        self.result_locations = self.mainmodel.schematisation.result_locations
        self.datadir = self.mainmodel.datadir

        # Create output dataframe
        self.output = ExtendedDataFrame(
            columns=['Wave direction', 'Lr', 'Beq', 'Smax', 'table_type', 'breakwater', 'shading', 'X', 'Y', 'Kd', 'Diffraction direction'],
            dtype=float)


    def initialize(self):
        """
        Initialize calculation. Geometries are retrieved from schematization, and the output
        table if initialized
        """
        # Geometries
        self.wlevcol = self.mainmodel.hydraulic_loads.wlevcol
        self.breakwaters = self.schematisation.breakwaters
        self.bedlevel = self.schematisation.bedlevel
        self.inner = self.schematisation.inner
        self.buffered_inner = self.schematisation.buffered_inner

        # Check content of breakwater geodataframe
        required_columns = ['geometry', 'breakwaterhead']
        for col in required_columns:
            if col not in self.breakwaters.columns:
                raise AttributeError('Column "{}" is not present. Expected at least "{}"'.format(col, '", "'.join(required_columns)))

        # Raise error if number of breakwaters is larger than 2
        self.nbreakwaters = len(self.breakwaters)
        if self.nbreakwaters > 2:
            raise ValueError('More than two breakwaters: simple method is not applicable!')

        # No breakwater
        if not np.size(self.breakwaters):
            Kd = 1.0
            return Kd

        # Empty if not already empty
        self.output.delete_all()
        
        # Index
        locations = pd.Series(self.result_locations['Naam'].tolist(), dtype='category')
        index = pd.MultiIndex.from_product((locations, self.hydraulic_loads.index), names=['Location', 'HydraulicLoadId'])
        
        # Set wave directions with index
        self.output['Wave direction'] = pd.Series(
            data=np.tile(self.hydraulic_loads['Wave direction'].values, len(self.result_locations)),
            index=index
        )
        # self.output.
        self.output.sort_index(inplace=True)
    

    def diffraction_angle(self, bwi, location):
        """
        Calculate diffraction angle
        
        Parameters
        ----------
        bwi : break water index
            List or array of indeces of the breakwater to use in the calculation
        locationcrd : tuple
            Coordinate of the location to use for calculating the diffraction direction.
        """

        breakwaterindex = np.atleast_1d(bwi)

        angles = np.zeros(len(breakwaterindex))

        for idx in np.unique(breakwaterindex):
            
            # Determine the coordinate from which the wave comes
            if idx < 2:
                bwcrd = self.breakwaters.at[idx, 'breakwaterhead'].coords[0]
            else:
                heads = [p.coords[0] for p in self.breakwaters['breakwaterhead']]
                bwcrd = ((heads[0][0] + heads[1][0]) / 2, (heads[0][1] + heads[1][1]) / 2)

            crds = np.vstack([bwcrd, location.geometry.coords[0]])
            angle = core.geometry.car2nau(np.angle(complex(*np.diff(crds, axis=0).squeeze()), deg=True) % 360)
            
            # Assign to array
            angles[breakwaterindex == idx] = angle
        
        if isinstance(bwi, (np.floating, np.integer, int, float)):
            angles = angles.squeeze()

        return angles


    def _determine_shading_breakwater(self, wavedirections, locationcrd):
        """
        Determine which of the breakwaters is shading a location, given a wavedirection
        core.geometry.

        Parameters
        ----------
        wavedirection : float
            wave direction
        locationcrd : tuple
            coordinate of the location

        Returns
        -------
        breakwaterindex : int
            Index of the shading breakwater
        """

        heads = [row.breakwaterhead.coords[0] for row in self.breakwaters.itertuples()]
        index = []

        for wavedir in wavedirections:
            # Draw a line from the first breakwater in the wave direction
            firstline = core.geometry.extend_point_to_linestring(heads[0], (wavedir + 180) % 360, (100000., -100000.))
            # If the head of the second breakwater is on the left, the first breakwater is the right one
            rightindex = 0 if is_left(heads[1], firstline) else 1
            # Assign opposite index to rightindex
            leftindex = 1 if rightindex == 0 else 0

            # If the location is on the left of left breakwater
            leftline = core.geometry.extend_point_to_linestring(heads[leftindex], (wavedir + 180) % 360, (100000., -100000.))
            rightline = core.geometry.extend_point_to_linestring(heads[rightindex], (wavedir + 180) % 360, (100000., -100000.))
            if is_left(locationcrd, leftline):
                index.append(leftindex)

            # If the location is on the right of the right breakwater
            elif ~is_left(locationcrd, rightline):
                index.append(rightindex)

            else:
                index.append(2)

        return np.array(index)
            

    def run(self):
        """
        Run with the following steps:
        1. Calculate representative wave length for each combination
        2. Calculate equivalent opening width per situation, and check if
           the harbor entrance is reachable by the wave (the wave should not cross
           the inner area of the harbor)
        3. Calculate the diffraction coefficients
        """

        # Change dataframe index
        self.output.reset_index(inplace=True)
        self.output.sort_values(by=['Location', 'Wave direction', 'HydraulicLoadId'], inplace=True)
        self.output.set_index(['Location', 'Wave direction'], inplace=True)

        # Calculate represenatative wave length
        #--------------------------------------------------------------------
        # Wave parameters are function input, represenative wave length is calculated:
        # print(self.output.columns.tolist())
        # Get water levels and Tm-1,0 from hydraulic loads
        wlev, tmm10 = self.hydraulic_loads.reset_index().sort_values(by=['Wave direction', 'HydraulicLoadId'])[[self.wlevcol, 'Tm-1,0']].values.T
        
        Lr = calc_repr_wave_length(d=wlev-self.bedlevel, T=tmm10)

        self.output['Lr'] = np.tile(Lr, len(self.result_locations))

        # Calculate equivalent opening width harbor entrance
        #--------------------------------------------------------------------
        # If only one breakwater, no equivalent opening width
        
        # If two breakwaters, calculate width
        if self.nbreakwaters == 2:
            # Create a array to assign values
            Beq = np.ones(len(self.output)) * np.nan
            # Calculate equivalent width per wave direction
            wavedirections = self.output.index.get_level_values(1).array
            # wavedirections = self.hydraulic_loads['Wave direction'].values
            for wavedirection in np.unique(wavedirections):
                # Calculate equivalent width for each direction
                Beq[wavedirections == wavedirection] = self.calc_eq_opening_width(wavedirection)

            # Add to dataframe
            self.output['Beq'] = Beq
            isnone = np.isnan(Beq)
            self.output.loc[isnone, 'Kd'] = 0.0
            self.output.loc[isnone, 'table_type'] = 0

            # In case of two breakwaters, in can occur that the inner breakwater
            # crosses the line between the location and the outer breakwater head
            # in this case, return Kd = 0:
            #-----
            for location in self.result_locations.itertuples():
                # determine outer head
                dists = self.breakwaters.distance(location.geometry)
                outerindex = dists.idxmax()
                innerindex = dists.idxmin()

                # generate line between outer head and location
                line = LineString([self.breakwaters.loc[outerindex, 'breakwaterhead'].coords[0], location.geometry.coords[0]])

                # check if line crosses other breakwater
                if line.crosses(self.breakwaters.loc[innerindex, 'geometry']):
                    self.output.loc[location.Naam, 'Kd'] = 0.0
        
        else:
            # In case of one breakwater, also check if the entrance can be reached;
            wavedirections = self.output.index.get_level_values(1)
            index = np.ones(len(wavedirections), dtype=bool)
            for wavedirection in np.unique(wavedirections):
                # check both ends of the harbor entrance
                if any(self.mainmodel.schematisation.is_reachable(crd, wavedirection) for crd in self.mainmodel.schematisation.entrance.coords):
                    index[wavedirections == wavedirection] = False
            
            # For the directions that are not reachable, set Kd to zero.
            self.output.loc[index, 'Kd'] = 0.0
            self.output.loc[index, 'table_type'] = 0

        # Step 5C: Choice of diffraction table
        #--------------------------------------------------------------------
        # For Dutch breakwater, set Smax is 10, or a large directional spread
        self.output['Smax'] = 10

        # First determine whether to use tabel for single or double breakwater
        self.output.loc[self.output['table_type'] != 0, 'table_type'] = 1
        
        if self.nbreakwaters == 2:
            # Equivalent width should be smaller than 5 times represntative wave length, and equivalent width should nonzero
            condition = ((self.output['Beq'] < 5 * self.output['Lr']) & (self.output['Beq'] > 0.001))
            self.output.loc[condition, 'table_type'] = 2
       
        # Step 5D: Location where diffractiecoefficient is determined
        #--------------------------------------------------------------------
        # Load diffraction tables
        self._load_diffraction_tables()
        
        # Three categories: table_type 2, origin in middle, table type 1, origin
        # on breakwater 1, table_type 1, origin on second breakwater (if 2 bws)
        # for locnaam, group in self.output.groupby(level=0, observed=False):
        for locnaam, group in self.output.groupby(level=0, observed=False):
            location = self.result_locations.set_index('Naam').loc[locnaam, 'geometry'].coords[0]
            wavedirections = group.index.get_level_values(1).values

            breakwater_index = np.ones_like(wavedirections) * np.nan
            shading_index = breakwater_index.copy()
            x = breakwater_index.copy()
            y = breakwater_index.copy()
            
            # Single breakwater (must by table type 1)
            if self.nbreakwaters == 1:
                origin = self.breakwaters.at[0, 'breakwaterhead'].coords[0]
                x, y = core.geometry.calculate_XY(
                    origin=origin, location=location, wavedirection=wavedirections, breakwater=self.breakwaters.iloc[0])
                breakwater_index = 0
                
            # Table type 2: affected by two breakwaters
            if any(group['table_type'] == 2):
                condition = (group['table_type'] == 2).values
                # Calculate the location of the origin, between the two breakwaters
                origin = tuple(np.vstack([p.coords[0] for p in self.breakwaters['breakwaterhead']]).mean(axis=0))
                x[condition], y[condition] = core.geometry.calculate_XY(origin, location, wavedirections[condition])
                
                # Use the center of both breakwaters
                breakwater_index[condition] = 2
                
            # Two breakwaters, but Beq is such that table type is 1
            if self.nbreakwaters == 2 and any(group['table_type'] == 1):
                # Check where the opening is wide, the other cases are Beq = 0.0
                wide_opening = (group['Beq'].gt(5 * group['Lr']) & group['table_type'].eq(1)).array
                
                if any(wide_opening):
                    """
                    Three cases for the wide opening:
                    1. The location is not shadowed by the breakwaters. Use the largest Kd of both.
                    2. The location is shadowed by breakwater 1, use 1
                    3. The location is shadowed by breakwater 2, use 2
                    """

                    # First determine the shading breakwaters for the location and the directions
                    shading_index[wide_opening] = self._determine_shading_breakwater(wavedirections[wide_opening], location)
                    breakwater_index[wide_opening] = shading_index[wide_opening]

                    # For the wide opening, calculate both diffraction coefficients, and pick the largest
                    if any(shading_index == 2):
                        Kd = []
                        for breakwater in self.breakwaters.itertuples():
                            x_tmp, y_tmp,= core.geometry.calculate_XY(
                                origin=breakwater.breakwaterhead.coords[0],
                                location=location,
                                wavedirection=wavedirections[shading_index == 2],
                                breakwater=breakwater
                            )
                            Kd.append([self.interp_diffraction(X, Y, row.Lr, row.Beq, row.table_type, row.Smax)
                                    for X, Y, row in zip(x_tmp, y_tmp, group.iloc[np.where(wide_opening==True)].itertuples())])
                        Kd = np.c_[Kd]
                        # Add diffraction directions for the used breakwater (max Kd)
                        breakwater_index[shading_index == 2] = np.argmax(Kd, axis=0)
                        
                # For the other cases (Beq = 0.0)
                no_opening = ((group['Beq'] < 0.001) & (group['table_type'] == 1)).values

                if any(no_opening):

                    # Determine which breakwater is reached first
                    # 1. create lines (points) from center towards wave directions
                    middle = tuple(np.vstack([p.coords[0] for p in self.breakwaters['breakwaterhead']]).mean(axis=0))
                    lines = core.geometry.extend_point_to_linestring(middle, (wavedirections[no_opening] - 180) % 360, 10000)
                    extended_point = lines[:, -1, :]
                    
                    # 2. Determine distances to breakwaterheads, and determine nearest
                    dists = np.c_[
                        [np.hypot(extended_point[:, 0] - bw.breakwaterhead.x, extended_point[:, 1] - bw.breakwaterhead.y)
                         for bw in self.breakwaters.itertuples()]]
                    breakwater_index[no_opening] = np.argmin(dists, axis=0)
                    shading_index[no_opening] = breakwater_index[no_opening]

                # Calculate for each breakwater the X and Y
                # The checks if the breakwater and location are reachable are already carried out
                for bwi in np.unique(breakwater_index[no_opening | wide_opening]):
                    # Check for which of the conditions to use this breakwater
                    index = (breakwater_index == bwi) & (no_opening | wide_opening)
                    # Calculate x and y, and add to the dataframe
                    x[index], y[index] = core.geometry.calculate_XY(
                        origin=self.breakwaters.at[int(bwi), 'breakwaterhead'].coords[0],
                        location=location,
                        wavedirection=wavedirections[index],
                        shading=shading_index[index] == bwi
                    )
            
            # Add group values to dataframe
            shading = shading_index == breakwater_index
            self.output.loc[locnaam, ['breakwater', 'X', 'Y', 'shading']] = np.c_[[np.ones(len(x)) * breakwater_index, x, y, shading]].T

        # Calculate diffraction direction
        # Determine which breakwater is the nearest (for calculating the diffraction angle)
        names = self.output.index.get_level_values(0).values
        breakwater_index = self.output['breakwater'].values
        
        for location in self.result_locations.itertuples():
            index = (names == location.Naam) & ~pd.isnull(breakwater_index)
            self.output.loc[index, 'Diffraction direction'] = self.diffraction_angle(breakwater_index[index].astype(int), location)            
        
        # Step 5E: Calculation of diffractioncoëfficient
        #--------------------------------------------------------------------
        # First determine where Lr == nan, no wave period, no waves
        # Also the combinations that already are assigned 0 should be skipped
        indexes = pd.isnull(self.output['Lr']) | (self.output['Kd'] == 0.0)
        self.output.loc[indexes, 'Kd'] = 0.0
        self.output.loc[~indexes, 'Kd'] = [self.interp_diffraction_row(row) for row in self.output.loc[~indexes].itertuples()]

        # Check valid range
        if any(self.output['Kd'] < 0.0) or any(self.output['Kd'] > 1.1):
            raise ValueError('Some values for Kd are outside the valid range (0 - 1.1)')

        # Change dataframe index
        self.output.reset_index(inplace=True)
        self.output.set_index(['Location', 'HydraulicLoadId'], inplace=True)
        
        
    def _load_diffraction_tables(self):
        """
        Load diffraction tables
        """

        # Import backgroundinfo
        tabledir = os.path.join(self.datadir, 'diffraction_tables')
        # make empty dictionary
        self.diffraction_tables = {}
        # Get all tables from dir
        tablefiles = [file for file in os.listdir(tabledir) if file.endswith('.csv')]
        # Loop through tables
        for table in tablefiles:
            # Get tablename from filepath
            tablename = os.path.splitext(table)[0]
            # Add to output dictionary
            self.diffraction_tables[tablename] = core.models.InterpolationTable(os.path.join(tabledir, table))            


    def calc_eq_opening_width(self, wavedir):
        """
        Calculates equivalent opening width of the harbor entrance.

        There are a number of special cases that should be considered:

        1. When the center of the breakwater heads (diffraction points)
        cannot be reached by the wave without crossing the harbor,
        None is returned. The entrance is shaded, so the wave cannot
        enter the harbor. Diffraction is not relevant in that case (Kd = 0)

        2. The second case is an equivalent width of 0.0. This can be the
        case when the entrance is shaded, but the wave can reach the breakwaters
        (condition 1 is not the case). In this case the equivalent width is 0
        and only one breakwater will give diffraction.

        To check condition one, draw a line from both breakwater heads, and see
        if one of them does not intersect the harbor. To check the second condition,
        draw a line from the breakwater origin, and check if it crosses the harbor.
        """

        # Check if at least one or breakwater heads are reachable;
        heads = [p.coords[0] for p in self.breakwaters['breakwaterhead']]
        if not any(self.mainmodel.schematisation.is_reachable(head, wavedir) for head in heads):
            return np.nan

        # Check if center of entrance is reachable
        origin = ((heads[0][0] + heads[1][0]) / 2, (heads[0][1] + heads[1][1]) / 2)
        # Unreachable, continue
        if not self.mainmodel.schematisation.is_reachable(origin, wavedir):
            return 0.0

        # Else we calculate the equivalent distance between the heads,
        # by drawing a line from one of the heads, and calculate the distance
        # to the other breakwaterhead
        # Generate line
        line = core.geometry.extend_point_to_linestring(
            pt=heads[0],
            direction=wavedir,
            extend=(100000., -100000.)
        )
        distance = core.geometry.perp_dist_to_line(pt=heads[1], line=line)
        
        return distance


    def interp_diffraction_row(self, row):
        return self.interp_diffraction(row.X, row.Y, row.Lr, row.Beq, int(row.table_type), int(row.Smax))


    def interp_diffraction(self, X, Y, Lr, Beq, table_type, Smax):
        """
        Function to interpolate values in diffraction table

        Parameters
        ----------
        X : float
                Distance from wave direction to point
        Y : float
            Distance from X-axis to origin
        Lr : float
            Represenative wave length
        Beq : float
            Equivalent entrance width
        table_type : int
            enumerator of the table_type
        Smax : int
            Coefficient for directional spread

        Returns
        -------
        Kd : float
            Diffraction coefficient
        """
        
        # catch Beq = nan --> conditions with periods = 0 lead to Lr = nan and Beq = nan

        if table_type == 1:
            # Select correct table
            tablename = 'type_1_smax{}'.format(Smax)
            Kd = self.diffraction_tables[tablename].interpolate(X, Y, Lr)
        else: # table_type == 2
            if np.isnan(Beq):
                Kd = 0
            else:
                # Calculate Beq / Lr
                BL = Beq / Lr

                # For Beq/Lr < 1.0, use the B/L=1 tables
                if BL <= 1.0:
                    # Check if X / Lr and Y / Lr are in the domain, if so, use the detailed table
                    size = 'small' if self.diffraction_tables['type_2_BL1_small_smax{}'.format(Smax)].check_range(X, Y, Lr) else 'large'
                    tablename = 'type_2_BL1_{}_smax{}'.format(size, Smax)
                    Kd = self.diffraction_tables[tablename].interpolate(X, Y, Lr)

                # For Beq/Lr >= 8.0, use the B/L=8 tables
                elif BL >= 8.0:
                    # Check if X / Lr and Y / Lr are in the domain, if ss, use the detailed table
                    size = 'small' if self.diffraction_tables['type_2_BL8_small_smax{}'.format(Smax)].check_range(X, Y, Lr) else 'large'
                    tablename = 'type_2_BL8_{}_smax{}'.format(size, Smax)
                    Kd = self.diffraction_tables[tablename].interpolate(X, Y, Lr)

                # For values of Beq/Lr in between 1.0 and 8.0: interpolate
                else:
                    # Check what the lower and upper table for the value of Beq/Lr are
                    BLgrid = np.array([1, 2, 4, 8])
                    BL_small = BLgrid[BLgrid < BL].max()
                    BL_large = BLgrid[BLgrid > BL].min()

                    # Check if X / Lr and Y / Lr are in the domain, if so, use the detailed table
                    size = 'small' if self.diffraction_tables['type_2_BL{}_small_smax{}'.format(BL_small, Smax)].check_range(X, Y, Lr) else 'large'
                    tablename = 'type_2_BL{}_{}_smax{}'.format(BL_small, size, Smax)
                    Kd_small = self.diffraction_tables[tablename].interpolate(X, Y, Lr)

                    # Check if X / Lr and Y / Lr are in the domain, if so, use the detailed table
                    size = 'small' if self.diffraction_tables['type_2_BL{}_small_smax{}'.format(BL_large, Smax)].check_range(X, Y, Lr) else 'large'
                    tablename = 'type_2_BL{}_{}_smax{}'.format(BL_large, size, Smax)
                    Kd_large = self.diffraction_tables[tablename].interpolate(X, Y, Lr)

                    # Interpolate between de small and large values
                    Kd = np.interp(BL, [BL_small, BL_large], [Kd_small, Kd_large])

        return Kd


def calc_repr_wave_length(d, T):
    """
    Function to calculate the representative wave length Lp.

    Method based on 4th order Padé approximation:
    y = (4*pi^2)/g * d/T^2
    x = y(y+ 1./(1 + 0.667y + 0.445y^2 - 0.105y^3 + 0.272y^4))
    L = 2*pi*d/sqrt(x)

    Parameters
    ----------
    d : float
        depth
    T : float
        wave period

    Returns
    -------
    Lr : float
        representative wave length

    References
    ----------
    RWS, 2014. Golfbelastingen in havens en afgeschermde gebieden - een methode voor het
        bepalen van golfbelastingen voor het toetsen van waterkeringen. Rapport versie 3.
        RWS.2014.001. Rijkswaterstaat Water, Verkeer en Leefomgeving. 31 augustus 2014.
    """

    # Convert datatype if necessary
    if isinstance(T, (int, np.integer, float, np.floating)):
        T = np.array([T])

    # Create empty array for Lr
    Lr = np.zeros_like(T)
    Lr[np.where(T == 0)] = np.nan

    # Remove zero values of T (so no division by zero)
    d = d[np.where(T != 0)]
    T = T[np.where(T != 0)]

    # If there are values left, calculate Lr
    if np.size(T):
        y = (4 * np.pi**2) / g * d / T**2
        x = y * (y + 1. / (1 + 0.667 * y + 0.445 * y**2 - 0.105 * y**3 + 0.272 * y**4))
        Lr[np.where(Lr == 0)] = 2 * np.pi * d / (x)**0.5

    return Lr.squeeze()


class Transmission:

    def __init__(self, simple_calculation):

        """
        Class to calculate wave transmission trough one or two breakwaters.

        Parameters
        ----------
        breakwater : geopandas.GeoDataFrame
            geodataframe with breakwaters. Should contain at least the columns
            "geometry" (LineString) and "breakwaterhead" (Point).

        hrdlocations : geopandas.GeoDataFrame
            Locations at which the reduction is calculated

        hydraulic_loads : pandas.DataFrame
            dataframe with hydraulic loads

        """

        self.mainmodel = simple_calculation.mainmodel
        self.result_locations = self.mainmodel.schematisation.result_locations
        self.hydraulic_loads = self.mainmodel.hydraulic_loads

        hl_cols = ['Water level', 'Hs', 'Wave direction']
        self.output = ExtendedDataFrame(columns=hl_cols + ['Breakwater', 'Alpha', 'Beta', 'Vrijboord', 'Kt'], dtype=float)

    def initialize(self):
        """
        Initialize calculation. Geometries are retrieved from schematization and the output
        table is initialized.
        """
        self.wlevcol = self.mainmodel.hydraulic_loads.wlevcol
        self.breakwaters = self.mainmodel.schematisation.breakwaters
        self.breakwater_properties = self.mainmodel.project.settings['schematisation']['breakwater_properties']
        self.inner = self.mainmodel.schematisation.inner
        self.area_union = self.mainmodel.schematisation.area_union

        # Checks on function input
        #--------------------------------------------------------------------
        # Check content of breakwater geodataframe
        required_columns = ['geometry', 'breakwaterhead', 'hoogte', 'alpha', 'beta']
        for col in required_columns:
            if col not in self.breakwaters.columns:
                raise AttributeError('Column "{}" is not present. Expected at least "{}"'.format(col, '", "'.join(required_columns)))

        # Raise error if number of breakwaters is larger than 2
        nbreakwaters = len(self.breakwaters)
        if nbreakwaters > 2:
            raise ValueError('More than two breakwaters: simple method is not applicable!')

        # Generate output dataframe
        #--------------------------------------------------------------------
        self.output.delete_all()
        # Create index
        locations = pd.Series(self.result_locations['Naam'].tolist(), dtype='category')
        index = pd.MultiIndex.from_product((locations, self.hydraulic_loads.index), names=['Location', 'HydraulicLoadId'])
        # Add relevant columns from hydarulic loads (sorted!)
        for col in ['Water level', 'Hs', 'Wave direction']:
            self.output[col] = pd.Series(
                data=np.tile(self.hydraulic_loads[col.replace('Water level', self.wlevcol)].array, len(locations)),
                index=index
            )
        self.output.sort_index(inplace=True)
        
        self.output['Kt'] = 0.0

        if not np.size(self.breakwaters):
            return None

    def run(self):
        """
        Run calculation of wave transmission.
        """
        # Step 6A: Calculate transmission zone for each wave direction where Hs is nonzero        

        loccrds = self.result_locations.sort_values(by='Naam').get_coordinates().to_numpy()
        locationnames = self.result_locations.sort_values(by='Naam')['Naam'].values
        hl_cols = [self.wlevcol, 'Hs', 'Wave direction']

        waterlevels, Hs, wavedirections = self.hydraulic_loads.sort_index()[hl_cols].values.T
        unique_directions = np.unique(wavedirections)

        # Collect breakwater properties. Ignore properties that don't match number of breakwaters
        heights = [self.breakwater_properties['height'][i] for i,_ in self.breakwaters.iterrows()] 
        alphas = [self.breakwater_properties['alpha'][i] for i,_ in self.breakwaters.iterrows()] 
        betas = [self.breakwater_properties['beta'][i] for i,_ in self.breakwaters.iterrows()]
        order = sorted(range(len(heights)), key=lambda k: heights[k])

        # Determine locations in transmission zone for each of the breakwaters
        transmission_zones = {}
        for idx in order:
            breakwater = self.breakwaters.iloc[idx]
            # Create an empty dictionary for the breakwater, to add wave directions per location
            transmission_zones[idx] = {name: [] for name in locationnames}
            geo = as_linestring_list(breakwater.geometry.difference(self.area_union))
            bwcrds = [geo[0].coords[0], geo[-1].coords[-1]]
            for wdir in unique_directions:
                # Determine if the breakwater can be reached
                if not any(self.mainmodel.schematisation.is_reachable(crd, wdir) for crd in bwcrds):
                    continue
                # Determine transmission zone
                left, right = self.determine_transmission_zone(idx, wdir)
                # Find locations in the zone
                ind = is_left(loccrds, right) & ~is_left(loccrds, left) & is_left(loccrds, [left[0], right[0]])
                # Add to dictionary
                for locname in locationnames[ind]:
                    transmission_zones[idx][locname].append(wdir)

        # Determine from which direction the entrance can be reached
        entrance_free = []
        for wavedirection in unique_directions:
            # check both ends of the harbor entrance
            if any(self.mainmodel.schematisation.is_reachable(crd, wavedirection) for crd in self.mainmodel.schematisation.entrance.coords):
                entrance_free.append(wavedirection)
        
        # Loop trough locations
        params = ['Kt', 'Breakwater', 'Vrijboord', 'Alpha', 'Beta']
        outdata = []
        for name, crd in zip(locationnames, loccrds):

            # Create dictionary with output arrays
            outarr = {key: np.zeros(len(self.hydraulic_loads)) * np.nan for key in params}
            outarr['Kt'][:] = 0.0
            
            # Locations that can be reached uninterrupted by the waves should have Kt = 1
            # Determine these locations by drawing a line in the direction the wave comes from, and checking if it
            # intersects the entrance. Also the entrance should be reachable from this direction
            lines = core.geometry.extend_point_to_linestring(crd, (unique_directions - 180) % 360, (0, 100000), as_LineString=True)
            location_free = [wdir for wdir, line in zip(unique_directions, lines) if self.mainmodel.schematisation.entrance.intersects(line)]
            outarr['Kt'][np.isin(wavedirections, list(set(entrance_free) & set(location_free)))] = 1.0

            # For each breakwater, from low (much transmission) to high (small transmission)
            # First the lowest breakwater, since that gives the most transmission, after this
            # the higher, since this can extra shade the location 
            for idx in reversed(order):
                height = heights[idx]
                alpha = alphas[idx]
                beta = betas[idx]

                # Determine which of the wave directions is in the transmission zone for the breakwater
                inzone = np.isin(wavedirections, transmission_zones[idx][name])
                                        
                # Step 6B: Calculating freeboard (height of breakwater above water, positive means low transmission)
                freeboard = -waterlevels[inzone] + height
                # Step 6C: Calculating transmission coefficient
                outarr['Kt'][inzone] = self._calc_transmission_coefficient(freeboard, Hs[inzone], alpha, beta)
                outarr['Alpha'][inzone] = alpha
                outarr['Beta'][inzone] = beta
                outarr['Vrijboord'][inzone] = freeboard
                outarr['Breakwater'][inzone] = idx
            
            # Merge output before adding to dataframe
            outdata.append(np.c_[[outarr[p] for p in params]].T)
            
        self.output[params] = np.vstack(outdata)
                
    def determine_transmission_zone(self, bwidx, wavedirection):
        """
        Calculate transmission zone based on wave direction and breakwater
        core.geometry.

        Parameters
        ----------
        breakwater : LineString
            GeoSeries with breakwater properties

        wavedirection : float
            wave direction

        Returns
        -------
        leftline : tuple
            line limiting the transmission zone on the left

        rightline : tuple
            line limiting the transmission zone on the right
        """

        breakwater = self.breakwaters.loc[bwidx, 'geometry']

        # Draw a line from the first point of the breakwater in wave direction
        # determine what end is on what side
        firstline = core.geometry.extend_point_to_linestring(
            breakwater.coords[0],
            wavedirection,
            (-100000., 100000.)
        )

        # Check if the first coordinate is left or right, and assign on index to leftindex accordingly
        leftindex = -1 if is_left(breakwater.coords[-1], firstline) else 0
        # Assign opposite index to rightindex
        rightindex = 0 if leftindex == -1 else -1

        # Create sufficiently long lines to contain all points
        leftline = core.geometry.extend_point_to_linestring(
            breakwater.coords[leftindex],
            (wavedirection - 15) % 360,
            (0, 100000)
        )
        rightline = core.geometry.extend_point_to_linestring(
            breakwater.coords[rightindex],
            (wavedirection + 15) % 360,
            (0, 100000)
        )

        return leftline, rightline



    def _calc_transmission_coefficient(self, freeboard, Hs, alpha, beta):
        """
        Calculate transmission coefficient

        ----------
        Parameters
        freeboard : float
            waterlevel above dam
        Hs : float
            significant wave height
        alpha : float
            dam coefficient
        beta : float
            dam coefficient

        Returns
        -------
        Kt : float
            transmission coefficient

        """

        # Create output array with all zeros
        Kt = np.zeros_like(Hs)

        frac = freeboard / np.maximum(Hs, 0.0001)
        
        # Set other values satisfying other condition to a calculated Kd
        indices = (frac <= (alpha - beta))
        Kt[indices] = 0.5 * (1 - np.sin(0.5*np.pi * (frac[indices] + beta) / alpha))
        # Set values satisfying condition to 1.0
        Kt[frac <= (-alpha - beta)] = 1.0
        
        # Other values to 0
        Kt[(Hs <= 0.0)] = 0.0

        return Kt


class LocalWaveGrowth:

    def __init__(self, simple_calculation, step=1.0, spread_angle=60.0):
        """
        Class for calculating local wave growth.

        Parameters
        ----------
        shematisation : hbhavens.core.models.Schematisation class
            schematisation class with geometries and loads

        step : optional float or int
            Step size between the directions, default = 1.0.
        spread_angle : optional float or int
            Angle (spread) in which the maximum fetch is taken, default is 60

        TODO: Use energy already present in wave to make a more precise Bretschneider calculation

        """
        # Links
        self.mainmodel = simple_calculation.mainmodel
        self.result_locations = self.mainmodel.schematisation.result_locations
        self.hydraulic_loads = self.mainmodel.hydraulic_loads
        self.datadir = self.mainmodel.datadir

        # Output dataframe
        self.output = ExtendedDataFrame(columns=['Wind direction', 'Wind speed', 'Water level', 'U10', 'Feq', 'maxdir', 'Terrain level', 'Water depth', 'Hs,lg', 'Elg'], dtype=float)
        
        # Dictionaries for fetch
        self.max_fetch_per_direction = {}
        self.average_depth_per_direction = {}

        # Load wind speed conversion for Bretschneider
        self._load_Upot_to_U10()

        # Round spread angle to step
        self.step = step
        self.spread_angle = np.round(spread_angle/self.step) * self.step

    def initialize(self):
        """
        Initialize calculation. Geometries are retrieved from schematization and the output
        table is initialized.
        """
        self.wlevcol = self.mainmodel.hydraulic_loads.wlevcol
        self.breakwaters = self.mainmodel.schematisation.breakwaters
        self.flooddefence = self.mainmodel.schematisation.flooddefence
        self.inner = self.mainmodel.schematisation.inner
        self.harborarea = self.mainmodel.schematisation.harborarea
        self.bedlevel = self.mainmodel.schematisation.bedlevel

        # Prepare output structure
        self.output.delete_all()
        
        # Create index
        locations = pd.Series(self.result_locations['Naam'].tolist(), dtype='category')
        index = pd.MultiIndex.from_product((locations, self.hydraulic_loads.index), names=['Location', 'HydraulicLoadId'])
        
        # add loads
        self.output['Wind speed'] = pd.Series(data=np.tile(self.hydraulic_loads['Wind speed'], len(self.result_locations)), index=index)
        self.output['Wind direction'] = pd.Series(data=np.tile(self.hydraulic_loads['Wind direction'], len(self.result_locations)), index=index)
        self.output['Water level'] = pd.Series(data=np.tile(self.hydraulic_loads[self.wlevcol], len(self.result_locations)), index=index)

        self.output.sort_index(inplace=True)

    def run(self, progress_function=None):
        """
        Run the calculations for local wave growth

        Steps:
        1.  Get the wind directions from the hydraulic loads.
            Note that the direction in the hydraulic loads indicates where the wind comes from.
        2.  Determine fetch lines for each degree (default step size = 1) and for each location.
            The lines are drawn as the wind blows to the location; so the line for 45 degrees
            is going from the north-east harbor bound to the location in south-west direction.
        3.  Find fetch per direction.
            Per fetch line the length until crossing the harbor bound is calculated.
        4.  Find the average bed level per direction.
            Per fetch line, the average bed level is calculated. For this the height of
            the harbor terrain and the representative bed level is calculated.
        5.  Look up the fetch length and terrain level for each hydraulic load.
            The wind-directions (inverted, step 1) are used in this step.
            Example for 45 degrees wind direction:
            -   Wind comes from North east (45 degrees)
            -   Look up the fetch line for this direction, this is the line
                from the harbor bound to the location (pointing south-west),
                also assigned to direction 45, even though the direction of the line
                is 225 degrees.
        6.  Determine significant wave height with Bretschneider. Note that already present
            wave energy is not taken into account, which leads to an overestimation of the
            wave energy in these cases.

        """

        # Add 180 because wind direction is the direction the wind comes from, not where it goes to
        winddirections = (((self.hydraulic_loads.sort_index()['Wind direction']/self.step).round() * self.step) % 360).squeeze().values

        if progress_function:
            progress_function(1, 'Berekenen van lokale golfgroei - Bepalen strijklengte')

        self._determine_fetch_lines()

        if progress_function:
            progress_function(1, 'Berekenen van lokale golfgroei - Bepalen maximale strijklengte')

        # Calculate maximum fetch per direction
        self._find_fetch_per_direction()

        if progress_function:
            progress_function(1, 'Berekenen van lokale golfgroei - Gemiddelde bodemhoogte')

        # Calculate average bottom level per direction
        self._find_average_bed_level_per_direction()

        for locname in self.result_locations['Naam']:
            
            # Calculate fetch for location by looking up each winddirection in the dataframe
            for var1, var2 in zip(['max_fetch', 'max_direction', 'average_depth'], ['Feq', 'maxdir', 'Terrain level']):
                dct = self.fetchlines.loc[locname, var1].to_dict()
                self.output.loc[locname, var2] = [dct[wdir] for wdir in winddirections]

        self.output['Feq'] = self.output['Feq'].astype(float)
        self.output['Terrain level'] = self.output['Terrain level'].astype(float)

        self.output['Water depth'] = self.output['Water level'] - self.output['Terrain level']

        if progress_function:
            progress_function(1, 'Berekenen van lokale golfgroei - Bretschneider')

        # Calculate U10
        windspeeds = list(map(self._calc_U10, self.hydraulic_loads.sort_index()['Wind speed']))
        self.output.loc[:, 'U10'] = np.tile(windspeeds, len(self.result_locations))

        # Calculate local wave growth
        self.output.loc[:, 'Hs,lg'], _ = self.bretschneider(
            self.output.loc[:, 'Water depth'].values,
            self.output.loc[:, 'Feq'].values,
            self.output.loc[:, 'U10'].values
        )

        # Calculate wave energy
        self.output.loc[:, 'Elg'] = 0.0625 * self.output.loc[:, 'Hs,lg'] ** 2

    def _determine_fetch_lines(self):
        """
        Determine the fetch line per direction

        Parameters
        ----------
        step : optional float or int
            Step size between the directions, default = 1.0.

        """
        # Create empty GeoDataFrame for fetchlines
        self.directions = np.arange(0., 360., self.step)
        locations = sorted(self.result_locations['Naam'].tolist())
        index = pd.MultiIndex.from_product([locations, self.directions], names=['Location', 'Direction'])
        self.fetchlines = gpd.GeoDataFrame(index=index, columns=['fetch', 'max_fetch', 'average_depth', 'geometry'], crs='epsg:28992')

        # Loop trough locations
        lines = []
        for locname in locations:
            location = self.result_locations.set_index('Naam').loc[locname, 'geometry']
            # Determine fetch lines in each direction
            lines += core.geometry.extend_point_to_linestring(
                location,
                (self.directions - 180) % 360,
                (100000, 0),
                as_LineString=True
            )

        self.fetchlines['geometry'] = lines

        # Determine part of fetchlines that overlap with harbor inner area
        lines = self.fetchlines.intersection(self.inner)
        # In case of a MultiLineString, use only the part connected to the origin
        lines = [line.geoms[-1] if isinstance(line, MultiLineString) else line for line in lines]
        # Add new parts to geometry
        self.fetchlines['geometry'] = lines
        self.fetchlines['fetch'] = self.fetchlines.length


    def _find_fetch_per_direction(self):
        """
        Find the maximum fetch length in a direction with a spread.

        First the fetch lengths are calculated with a certain step (default 1
        degree), than the maximum within a certain spread are gathered.

        Parameters
        ----------
        locidx : pandas.Index
            Index of location for which the fetch is calculated
        """

        # Create an empty dataframe to save the distances
        minangle, maxangle = -1*self.spread_angle/2., 359+self.spread_angle/2.
        distances = pd.DataFrame(index=np.arange(minangle, maxangle+0.1*self.step, self.step), columns=['fetch'])

        for locname, linedata in self.fetchlines.groupby(level=0):

            distances.loc[self.directions, 'fetch'] = linedata.length.values

            # Fill the < 0  and > 360 angles in the distances dataframe
            distances.loc[360:maxangle+0.1*self.step, 'fetch'] = distances.loc[0:(maxangle+0.1*self.step-360), 'fetch'].values
            distances.loc[minangle:0, 'fetch'] = distances.loc[(360+minangle):360, 'fetch'].values

            window = int(round(self.spread_angle // self.step)) + 1
            # Determine maximum fetch by rolling maximum
            maxfetch = distances['fetch'].rolling(window=window, center=True).max().dropna()
            # Get max direction
            maxdir = distances['fetch'].rolling(window=window, center=True).apply(pd.Series.idxmax, raw=False).dropna()
            
            self.fetchlines.loc[locname, 'max_fetch'] = maxfetch.values
            self.fetchlines.loc[locname, 'max_direction'] = maxdir.values  % 360

    def _find_average_bed_level_per_direction(self):
        """
        Find the average terrain level per direction with the following steps:
        Create a

        Parameters
        ----------
        locidx : pandas.Index
            Index of location for which the fetch is calculated

        """
        self.fetchlines['average_depth'] = 0.0

        remaining_length = self.fetchlines.length

        # For each harbor area
        for idx, area in self.harborarea.iterrows():
            # Calculate the intersection distance
            isect_length = self.fetchlines.intersection(area['geometry']).length
            # Add the fraction of the total length multiplicated with the terrain level tot the average
            self.fetchlines['average_depth'] += isect_length / self.fetchlines.length * area['hoogte']

            # Subtract from the remaining length, so the intersection length with the bed level is known
            remaining_length -= isect_length

        # Add the bed level for the remaining length
        self.fetchlines['average_depth'] += remaining_length / self.fetchlines.length * self.bedlevel

    def _load_Upot_to_U10(self):
        """
        Function to load conversion table from potential wind speed (present
        in HRD's) to U10, which is needed for the fetch calculation.
        """

        filepath = os.path.join(self.datadir, 'Up2U10', 'Up2U10.dat')
        self.windtrans = pd.read_csv(filepath, comment='%', header=None, delim_whitespace=True, names=['Upot', 'U10'], encoding='latin-1', usecols=[0,3])


    def _calc_U10(self, Upot):
        """
        Function to calculate the U10 from the potential wind speed by
        interpolation

        Parameters
        ----------
        Upot : float or numpy.array
            potential wind speed

        Returns
        -------
        U10 : float of numpy.array
            U10
        """

        return np.interp(Upot, self.windtrans['Upot'].values, self.windtrans['U10'].values, left=np.nan, right=np.nan)

    def _calc_local_wave_growth(self, Feq, U10):
        """
        Calculate local wave growth from fetch and U10.

        Parameters
        ----------
        Feq : float or numpy.array
            Maximum fetch length per wind direction (in formula, equivalent fetch length)

        U10 : float or numpy.array
            U10

        Returns
        -------
        Hs,lg : float of numpy.array
            Local wave growth

        """

        # Convert input to array format
        if isinstance(Feq, (float, np.floating, int, np.integer)):
            Feq = np.array([Feq])
        if isinstance(U10, (float, np.floating, int, np.integer)):
            U10 = np.array([U10])

        Hs_lg = np.zeros_like(Feq)

        cif = g * Feq / U10**2

        mask = cif > 1e-2
        Hs_lg[mask] = (0.3 * (1-(1+0.004*cif[mask]**0.5)**-2)) * U10[mask]**2 / g
        Hs_lg[~mask] = (2.4e-3 * cif[~mask]**0.5) * U10[~mask]**2 / g

        return Hs_lg.squeeze()

    def bretschneider (self, d, fe, u):
        """
        Calculate wave conditions with Bretschneider.
        Based on "subroutine Bretschneider" in Hydra-NL, programmed by
        Matthijs Duits

        Parameters
        ----------
        d : float
            Water depth
        fe : float or numpy.array
            Maximum fetch length per wind direction (in formula, equivalent fetch length)
        u : float or numpy.array
            U10

        Returns
        -------
        hs : float of numpy.array
            Local wave growth
        tp : float
            Peak period
        """

        # Converteer scalar input to arrays
        if isinstance(d, (float, np.floating, int, np.integer)):
            d = np.array([d])
        if isinstance(fe, (float, np.floating, int, np.integer)):
            fe = np.array([fe])
        if isinstance(u, (float, np.floating, int, np.integer)):
            u = np.array([u])

        # Create output arrays to fill
        hs = np.empty_like(d)
        tp = np.empty_like(d)

        # Check for zeros
        isnull = ((d <= 0.0) | (fe <= 0.0) | (u <= 0.0))
        hs[isnull] = 0.0
        tp[isnull] = 0.0

        # Initialiseer constanten
        g = 9.81

        # Bereken Hs en Tp
        dtilde = (d[~isnull] * g) / (u[~isnull]**2)
        v1 = np.tanh(0.53 * (dtilde ** 0.75))
        v2 = np.tanh(0.833 * (dtilde ** 0.375))

        ftilde = (fe[~isnull] * g) / (u[~isnull]**2)

        hhulp = (0.0125 / v1) * (ftilde ** 0.42)
        htilde = 0.283 * v1 * np.tanh(hhulp)
        hs[~isnull] = (u[~isnull]**2 * htilde) / g

        thulp = (0.077 / v2) * (ftilde ** 0.25)
        ttilde = 2.4 * np.pi * v2 * np.tanh(thulp)
        tp[~isnull] = (1.08 * u[~isnull] * ttilde) / g

        return hs, tp


class WaveBreaking:

    def __init__(self, simple_calculation):
        """
        Class to calculate the maximum wave-height that can occur due to a
        undeep foreshore (if present). This maximum wave height is 0.7 times
        the water depth.

        Parameters
        ----------
        shematisation : hbhavens.core.models.Schematisation class
            schematisation class with geometries and loads

        TODO: Use the direction the wave will actually come from to determine the wave breaking

        """
        # Links
        self.mainmodel = simple_calculation.mainmodel
        self.schematisation = self.mainmodel.schematisation
        self.hydraulic_loads = self.mainmodel.hydraulic_loads
        self.output = ExtendedDataFrame(columns=['Water level', 'Tm-1,0', 'Wave direction', 'Terrain level', 'L0,p', 'Breaking length', 'Hs,max'], dtype=float)
        
        # Dataframe for wave lines
        self.wavelines = gpd.GeoDataFrame()

        # Parameters
        self.step = 1.0
        self.breakfraction = 0.7

    def initialize(self):
        """
        Initialize calculation. Geometries are retrieved from schematization and the output
        table is initialized.
        """

        # Create index
        locations = pd.Series(self.schematisation.result_locations['Naam'].tolist(), dtype='category')
        index = pd.MultiIndex.from_product((locations, self.hydraulic_loads.index), names=['Location', 'HydraulicLoadId'])
        
        # Prepare output structure, add loads
        self.wlevcol = self.mainmodel.hydraulic_loads.wlevcol
        self.output.delete_all()
        for col in ['Water level', 'Tm-1,0', 'Wave direction']:
            self.output[col] = pd.Series(
                data=np.tile(self.hydraulic_loads[col.replace('Water level', self.wlevcol)].array, len(self.schematisation.result_locations)),
                index=index
            )
        self.output.sort_index(inplace=True)

    def run(self):
        """
        Run calculation for all locations

        1. Determine lines over which waves travel
        2. Determine minimum bed level over lines. If no harbor elements
           are crossed, use the terrain level.
        3. Determine where (wave length is larger than L0,p, AND terrain elements present),
           For these locations: calculate max wave height with fraction * water depth
           For the other locations: calculate max wave height with fraction * (water level - rep. bottom level)
        """

        rep_bedlevel = self.schematisation.bedlevel

        # Determine wave breaking lines. This is per location, the fetch crossing the harbor terrain
        self.determine_wave_breaking_lines()
        # Determine l0,p (diep water golflengte)
        self.output.loc[:, 'L0,p'] = g / (2*np.pi) * self.output.loc[:, 'Tm-1,0'] ** 2

        # Determine bottom level for all locations
        for location in self.schematisation.result_locations.itertuples():
            # Determine terrain height
            heights = np.asarray(
                [area.hoogte for area in self.schematisation.harborarea.itertuples() if area.geometry.contains(location.geometry)],
                dtype=float
            )
            if len(heights) == 0:
                bedlevel = rep_bedlevel
            elif len(heights) == 1:
                bedlevel = heights[0]
            else:
                raise ValueError('More heights than expected')

            self.output.loc[location.Naam, 'Terrain level'] = bedlevel

            # Determine breaking length for all wave directions
            breaking_length = self.wavelines.loc[location.Naam, 'max_br_dst'].to_dict()
            wave_directions = np.round(self.output.loc[location.Naam, 'Wave direction'].array / self.step) * self.step
            
            self.output.loc[location.Naam, 'Breaking length'] = [breaking_length[wdir] for wdir in wave_directions]

        # Breaking length are the lengths in the direction of the wave until the harbor is reached
        # If the wave length is larger than this, no breaking, use representative harbor depth
        # If the wave length is shorter than this, breaking, use terrain level            

        # Calculate maximum wave height
        will_break = self.output['Breaking length'] > self.output['L0,p']
        wont_break = ~will_break
        # At location where breaking does not play a role, the bottom level is used
        self.output.loc[wont_break, 'Hs,max'] = np.maximum(
            (self.output.loc[wont_break, 'Water level'].values - rep_bedlevel) * self.breakfraction,
            0.0
        )
        # At the other locations, determine the maximum wave heigth based on the terrain level
        self.output.loc[will_break, 'Hs,max'] = np.maximum(
            (self.output.loc[will_break, 'Water level'].values - self.output.loc[will_break, 'Terrain level']) * self.breakfraction,
            0.0
        )

    def determine_wave_breaking_lines(self):
        """
        Determine the wave line per direction.
        
        1. Draw lines in all directions
        3. Pick part of lines that is connected to origin
        2. Determine intersections with harborterrain
        4. Determine length of lines
        """
        # Create empty GeoDataFrame for fetchlines
        self.directions = np.arange(0., 360. + self.step*0.1, self.step)
        locations = sorted(self.schematisation.result_locations['Naam'].tolist())
        index = pd.MultiIndex.from_product([locations, self.directions], names=['Location', 'Direction'])
        self.wavelines = gpd.GeoDataFrame(index=index, columns=['max_br_dst', 'geometry'], crs='epsg:28992')

        # Loop trough locations
        lines = []
        for locname in locations:
            location = self.schematisation.result_locations.set_index('Naam').loc[locname, 'geometry']
            # Determine fetch lines in each direction
            lines += core.geometry.extend_point_to_linestring(
                location,
                (self.directions - 180) % 360,
                (0, 100000),
                as_LineString=True
            )

        self.wavelines['geometry'] = lines

        # Determine part of wave lines that overlap with the harbor terrain
        lines = self.wavelines.intersection(self.schematisation.area_union)
        # In case of a MultiLineString, use only the part connected to the origin
        lines = [line.geoms[0] if isinstance(line, MultiLineString) else line for line in lines]
        # Add new parts to geometry
        self.wavelines['geometry'] = lines
        self.wavelines['max_br_dst'] = self.wavelines.length


class CombineResults:

    def __init__(self, simple_calculation):

        # Links
        self.mainmodel = simple_calculation.mainmodel
        self.hydraulic_loads = self.mainmodel.hydraulic_loads
        self.result_locations = self.mainmodel.schematisation.result_locations

        self.result_columns = [
            'Location', 'Kd', 'Kt', 'Kd,t', 'Ed', 'Et', 'Ed,t', 'Diffraction direction',
            'Hs,lg', 'Elg', 'Etotaal', 'Hs,totaal', 'Hs,max', 'Hs,out', 'Tm-1,0,out', 'Combined wave direction'
        ]

        # Output dataframe
        self.output = ExtendedDataFrame(dtype=float)

        # Links to other mechanisms
        self.diffraction = simple_calculation.diffraction.output
        self.transmission = simple_calculation.transmission.output
        self.wavegrowth = simple_calculation.wavegrowth.output
        self.wavebreaking = simple_calculation.wavebreaking.output

    def reset_columns(self):
        """Change the table columns"""
        columns = self.hydraulic_loads.columns.tolist() + self.hydraulic_loads.index.names + self.result_columns
        for col in columns:
            self.output[col] = np.nan
        for col in self.output.columns:
            if col not in columns:
                self.output.drop(col, axis=1, inplace=True)

    def delete_all(self):
        """
        """
        # Empty results
        if not self.output.empty:
            self.output.iloc[:, 0] = np.nan
            self.output.dropna(inplace=True)        
        self.reset_columns()

    def run(self, processes):
        """
        Combine the different processes to one
        """
        self.delete_all()

        if processes is not None:
            self.calc_diffraction = True if 'Diffractie' in processes else False
            self.calc_transmission = True if 'Transmissie' in processes else False

        # Add locations
        self.output['Location'] = np.repeat(self.result_locations['Naam'], len(self.hydraulic_loads))

        # Add hydraulic loads
        cols = self.hydraulic_loads.reset_index().columns.intersection(self.output.columns)
        loads = self.hydraulic_loads.reset_index()[cols]
        tiled = np.tile(loads, (len(self.result_locations), 1))
        self.output.loc[:, cols] = tiled
           
        # Set index
        self.output.set_index(['Location', 'HydraulicLoadId'], inplace=True)
        self.output.sort_index(inplace=True)
        
        # Join diffraction and transmission
        self.output[['Kd', 'Diffraction direction']] = self.diffraction[['Kd', 'Diffraction direction']]
        self.output['Kt'] = self.transmission['Kt']
        self.output[['Hs,lg', 'Elg']] = self.wavegrowth[['Hs,lg', 'Elg']]
        self.output['Hs,max'] = self.wavebreaking['Hs,max']
        
        # Calculate combined factors
        if self.calc_diffraction and self.calc_transmission:
            self.output['Kd,t'] = ((1-self.output['Kt']**2)*self.output['Kd']**2+self.output['Kt']**2)**0.5
        elif self.calc_diffraction:
            self.output['Kd,t'] = self.output['Kd']
        elif self.calc_transmission:
            self.output['Kd,t'] = self.output['Kt']

        # Calculate combined energy
        if not self.calc_diffraction and not self.calc_transmission:
            self.output['Ed,t'] = 0.0
            self.output['Et'] = 0.0
            self.output['Ed'] = 0.0
        else:
            self.output['Ed,t'] = self.output['Kd,t']**2 * (0.25 * self.output['Hs'])**2
            self.output['Et'] = self.output['Kt'].fillna(0.0)**2 * (0.25 * self.output['Hs'])**2
            self.output['Ed'] = self.output['Ed,t'] - self.output['Et']

        # Join local wave growth and calculate total energy
        self.output['Etotaal'] = self.output['Ed,t'].fillna(0.0) + self.output['Elg'].fillna(0.0)
        self.output['Hs,totaal'] = 4 * self.output['Etotaal']**0.5

        # Calculate waveheight
        self.output['Hs,out'] = np.nanmin(self.output[['Hs,max', 'Hs,totaal']], axis=1)
        self.output['Tm-1,0,out'] = self.output['Tm-1,0']

        # Combine wave directions
        self._combine_wave_directions()

        # Reset index
        self.output.reset_index(['Location', 'HydraulicLoadId'], inplace=True)

    def _combine_wave_directions(self):
        """
        Private method to combine the wave directions from the different
        calculation steps.

        The following directions are combined:
        - diffraction: from diffraction point to output point
        - transmission: wave direction in support location (NOT IMPLEMENTED)
        - local wave growth: wind direction for hydraulic load combination
        - wave breaking: ? (NOT IMPLEMENTED)

        Combining is done with the wave energies.
        """

        # Calculate average angle
        self.output['Combined wave direction'] = core.geometry.average_angle(
            angles=self.output[['Diffraction direction', 'Wave direction', 'Wind direction']].fillna(0.0).values,
            factors=self.output[['Ed', 'Et', 'Elg']].fillna(0.0).values,
            degrees=True
        )
        