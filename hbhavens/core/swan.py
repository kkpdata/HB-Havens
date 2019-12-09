import logging
import os
import time
import warnings

import numpy as np
import pandas as pd
from scipy.spatial import KDTree

from hbhavens import core
from hbhavens.core.datamodels import ExtendedDataFrame
from hbhavens.core.geometry import average_angle
from hbhavens.core.spectrum import incoming_wave_factors
from hbhavens.io.swan import SwanIO, read_swan_table

logger = logging.getLogger(__name__)


class IterationResults(ExtendedDataFrame):
    _columns = [
        'Load combination', 'ClosingSituationId', 'Wind direction', 'Wind speed', 'Water level',
        'h', 'Hs', 'Tp', 'Tm-1,0', 'Wave direction',
        'depth', 'Hs steunpunt 1', 'Tp,s steunpunt 1', 'alpha 1', 'beta 1', 'Hs rand 2', 'Tp,s rand 2',
        'Hs steunpunt 2', 'Tp,s steunpunt 2', 'alpha 2', 'beta 2', 'Hs rand 3', 'Tp,s rand 3',
        'Hs steunpunt 3', 'Tp,s steunpunt 3',
    ]
     
    # normal properties
    _metadata = [
        'settings', 'description', 'initialized', 'hydraulic_loads',
        'swanio', 'iter_columns'
    ]

    def __init__(self, parent):
        super(IterationResults, self).__init__(columns=self._columns)

        self.settings = parent.settings
        
        self.initialized = False
        self.hydraulic_loads = parent.mainmodel.hydraulic_loads
        self.description = {}
        self.swanio = parent.swanio

        self.iteration_finished = parent.iteration_finished

        self.iter_columns = parent.iter_columns

    
    def initialize(self):

        # Empty calculation results, in case it needs to be refilled
        self.delete_all()
        
        # Add index and columns to table
        for col, vals in self.hydraulic_loads.iteritems():
            if col == 'Description':
                col = 'Load combination'
            if col not in self._columns:
                continue
            self[col] = vals.values
        self.index = self.hydraulic_loads.index.values
        self.index.name = 'HydraulicLoadId'

        # Add columns for iteration results
        for col in self.iter_columns['I1'] + self.iter_columns['I2'] + self.iter_columns['I3']:
            self[col] = 0.0
        
        for k, v in self['Load combination'].str.replace(' ', '_').str.replace('=', '_').to_dict().items():
            self.description[k] = v

        self.initialized = True

    def load_from_pickle(self, path):
        # Empty calculation results if it is not already empty
        self.delete_all()
        self.load_pickle(path)
        self.initialized = True

    def read_results(self, step, progress_function=None):
        """
        Read results from SWAN iteration at support location

        Parameters
        ----------
        step : string
            Current step
        progress_bar : QProgressBar
            Progress bar updated on import
        """

        # Get calculation settings
        if step not in self.settings['swan']['calculations']:
            raise KeyError('Step {} not present in calculation settings'.format(step))

        # Get calculation and results folder
        folder = self.settings['swan']['calculations'][step]['folder']

        # Check where non-zero
        nonzero = ((self['Hs'].values != 0) & (self['Tm-1,0'].values != 0))
        if progress_function is not None:
            progress_function((sum(~nonzero)))
        
        # Create result columns
        arrs = {col: vals.loc[nonzero].values for col, vals in self.iteritems()}
        hydrodynamicdataids = self.index.values[nonzero]
        
        Hs_swan = np.zeros(sum(nonzero))
        Tps_swan = np.zeros(sum(nonzero))
        depth = np.zeros(sum(nonzero))

        # Loop trough all hydraulic load combinations
        for i, hydrodynamicdataid in enumerate(hydrodynamicdataids):
        
            # Update progress bar if present
            if progress_function is not None:
                progress_function(1)

            # Derive case id
            case_id = self.description[hydrodynamicdataid]
            
            # read swan table results
            fpath = os.path.join(folder, 'table', case_id + '.tab')
            if not os.path.exists(fpath):
                raise OSError('Resultaten voor case "{}" niet gevonden op de volgende locatie:\n{}'.format(case_id, fpath.replace('/', '\\')))
            
            # Calculated hydraulic loads (get the first row from the table, this contains the support location)
            iterresult = read_swan_table(fpath)
            Hs_swan[i] = iterresult['Hsig']
            Tps_swan[i] = iterresult['TPsmoo']
            depth[i] = iterresult['Depth']

        # Determine alfa and beta
        gamma = arrs['Hs'] / Hs_swan
        delta = arrs['Tp'] / Tps_swan
        beta = delta
        alpha = delta * gamma ** 2

        if step == 'I1':
            # Assign calculated values to arrays
            arrs['depth'] = depth
            arrs['Hs steunpunt 1'] = Hs_swan
            arrs['Tp,s steunpunt 1'] = Tps_swan
            arrs['alpha 1'] = alpha
            arrs['beta 1'] = beta

            # New hydraulic load second iteration
            arrs['Hs rand 2'] = arrs['Hs'] * np.sqrt(abs(alpha))
            arrs['Tp,s rand 2'] = arrs['Tp'] * delta
            
        elif step == 'I2':
            eps = 0.01
            idx = (np.absolute(alpha - arrs['alpha 1']) > eps) | (np.absolute(beta - arrs['beta 1']) > eps)

            arrs['Hs steunpunt 2'] = Hs_swan
            arrs['Tp,s steunpunt 2'] = Tps_swan
            
            arrs['alpha 2'] = alpha
            arrs['beta 2'] = beta

            arrs['Hs rand 3'][idx] = ((arrs['Hs'] - arrs['Hs steunpunt 1']) * ((arrs['Hs rand 2'] - arrs['Hs']) / (arrs['Hs steunpunt 2'] - arrs['Hs steunpunt 1'])) + arrs['Hs'])[idx]
            arrs['Tp,s rand 3'][idx] = ((arrs['Tp'] - arrs['Tp,s steunpunt 1']) * ((arrs['Tp,s rand 2'] - arrs['Tp']) / (arrs['Tp,s steunpunt 2'] - arrs['Tp,s steunpunt 1'])) + arrs['Tp'])[idx]
            
            arrs['Hs rand 3'][~idx] = arrs['Hs rand 2'][~idx]
            arrs['Tp,s rand 3'][~idx] = arrs['Tp,s rand 2'][~idx]

        elif step == 'I3':
            arrs['Hs steunpunt 3'] = Hs_swan
            arrs['Tp,s steunpunt 3'] = Tps_swan
            
        else:
            raise ValueError('Step "{}" not recognized'.format(step))

        # At to dataframe and round
        for col in self.iter_columns[step]:
            self.loc[nonzero, col] = arrs[col].round(3)
            
        # On the last iteration step, the calculation table can be initialized
        if step == 'I3':
            self.iteration_finished()

class CalculationResults(ExtendedDataFrame):
    
    _columns = [
        'Location', 'LocationId', 'Load combination', 'HydraulicLoadId', 'Water level',
        'h', 'Hs', 'Tp', 'Tm-1,0', 'Wave direction',
        'Hs rand 3', 'Tp,s rand 3',
        'X', 'Y', 'Normaal',
        'Hm0_D', 'Tmm10_D', 'Tp_D', 'Theta0_D',
        'Hm0_TR', 'Tmm10_TR', 'Tp_TR', 'Theta0_TR',
        'Hm0_W', 'Tmm10_W', 'Tp_W', 'Theta0_W',
        'Hm0 swan', 'Tm-1,0 swan', 'Tp swan', 'Wave direction swan'
    ]
    # normal properties
    _metadata = [
        'result_locations', 'iteration_results', 'result_parameters',
        'settings', 'description', 'initialized', 'hydraulic_loads',
        'swanio',
    ]

    def __init__(self, parent):
        super(CalculationResults, self).__init__(columns=self._columns)
        
        self.result_locations = parent.mainmodel.schematisation.result_locations
        self.iteration_results = parent.iteration_results
        self.result_parameters = parent.result_parameters
        self.settings = parent.settings
        self.description = parent.iteration_results.description
        self.initialized = False
        self.hydraulic_loads = parent.mainmodel.hydraulic_loads
        self.swanio = parent.swanio

    def initialize(self):
        
        # Empty calculation results, in case it needs to be refilled
        self.delete_all()

        if len(self.result_locations) == 0:
            raise ValueError('Result locations have not been filled')
        
        # Count locations and loads
        nlocations = len(self.result_locations)
        nloads = len(self.iteration_results)

        # Add index and columns to table
        self['Location'] = np.repeat(self.result_locations['Naam'], nloads)
        self['LocationId'] = np.repeat(self.result_locations.index.values, nloads)
        self['HydraulicLoadId'] = np.tile(self.iteration_results.index.values, nlocations)
        self['Load combination'] = np.tile(self.iteration_results['Load combination'].values, nlocations)
        self['Water level'] = np.tile(self.iteration_results['Water level'].values, nlocations)

        # Create initial table
        present_parameters = [param for param in self.result_parameters if param in self.iteration_results]
        first_columns = present_parameters + ['Hs rand 3', 'Tp,s rand 3']
                
        # Add values from iterations to calculation results
        tiled = np.tile(self.iteration_results[first_columns].values, (nlocations , 1))
        self[first_columns] = tiled

        # Add geometry
        # Get x, y and normaal per location, and repeat this for all lead combinations
        x_y_norm = np.round([[loc.geometry.x, loc.geometry.y, loc.Normaal] for loc in self.result_locations.itertuples()], 3)
        self[['X',  'Y', 'Normaal']] = np.repeat(x_y_norm, nloads, axis=0)

        self.initialized = True


    def load_from_pickle(self, path):
        # Empty calculation results if it is not already empty
        self.delete_all()
        self.load_pickle(path)
        
        # # Read data from pickle
        # df = pd.read_pickle(path)

        # # Copy all columns and index
        # for name, col in df[self.columns].iteritems():
        #     self[name] = col
        # self.index = df.index
        # self.index.name = df.index.name

        if not self.empty:
            self.initialized = True

    def calculate_total_energy(self):
        """
        Calculate the total energy SWAN
        """
        wd_dict = self.hydraulic_loads['Wind direction'].to_dict()
        self['Wind direction'] = [wd_dict[loadid] for loadid in self['HydraulicLoadId']]

        factor_E_W = np.maximum(self['Hm0_W'].fillna(0.0).values, 0.0) ** 2
        factor_E_D = np.maximum(self['Hm0_D'].fillna(0.0).values, 0.0) ** 2
        factor_E_TR = np.maximum(self['Hm0_TR'].fillna(0.0).values, 0.0) ** 2

        # Energy difference between wind growth and diffraction
        c = np.maximum(factor_E_W - factor_E_D, 0)
        # Energy difference between transmission and wind growth
        d = np.maximum(factor_E_TR - factor_E_W, 0)

        # Aangepast Svasek 04/10/18 - Toevoegen van de include_hares optie 
        # When PHAROS or HARES is used, subtract the influence of wave intrusion
        if self.settings['calculation_method']['include_pharos'] or self.settings['calculation_method']['include_hares']:

            # Total wave height is sum of: (W - D) and (TR - W)
            self['Hm0 swan'] = np.sqrt(c + d).round(3)

            # Same as c and d, but multiple with wave periods to calculate the weight averaged period
            a = factor_E_W * self['Tmm10_W'] - factor_E_D * self['Tmm10_D']
            b = factor_E_TR * self['Tmm10_TR'] - factor_E_W * self['Tmm10_W']
            Tm_10_weighted = np.maximum(((a + b) / (c + d)).round(3).fillna(0.0), 0.0)
            Tm_10_weighted[np.isinf(Tm_10_weighted)] = 0.0
            self['Tm-1,0 swan'] = Tm_10_weighted
            # It is possible to get infinity wave periode, if the Diffraction component is larger or equal than Wind and Diffraction together
            # c + d will be zero, resulting in an infinite 

            # The peak period is the maximum of the found periods
            self['Tp swan'] = np.round(np.nanmax(self[['Tp_W', 'Tp_TR']], axis=1), 3)
            # calculate the angle
            self['Wave direction swan'] = average_angle(
                angles=self[['Theta0_W', 'Theta0_D', 'Theta0_TR', 'Theta0_W']].values,
                factors=np.vstack([factor_E_W, -factor_E_D, factor_E_TR, -factor_E_W]).T,
                degrees=True
            ).round(3)
            idx = (factor_E_W + factor_E_TR + factor_E_D) == 0
            self.loc[idx, 'Wave direction swan'] = self.loc[idx, 'Wind direction']

        # Without Pharos or Hares
        else:
            # Total wave height is sum of: (W - D) and (TR - W)
            self['Hm0 swan'] = np.sqrt(factor_E_D + c + d).round(3)

            # Same as c and d, but multiple with wave periods to calculate the weight averaged period
            a = factor_E_W * self['Tmm10_W'] - factor_E_D * self['Tmm10_D']
            b = factor_E_TR * self['Tmm10_TR'] - factor_E_W * self['Tmm10_W']
            Tm_10_weighted = np.maximum(((factor_E_D * self['Tmm10_D'] + a + b) / (factor_E_D + c + d)).round(3).fillna(0.0), 0.0)
            Tm_10_weighted[np.isinf(Tm_10_weighted)] = 0.0
            self['Tm-1,0 swan'] = Tm_10_weighted

            # The peak period is the maximum of the found periods
            self['Tp swan'] = np.round(np.nanmax(self[['Tp_W', 'Tp_TR', 'Tp_D']], axis=1), 3)

            # calculate the angle
            self['Wave direction swan'] = average_angle(
                angles=self[['Theta0_D', 'Theta0_W', 'Theta0_D', 'Theta0_TR', 'Theta0_W']].values,
                factors=np.vstack([factor_E_D, factor_E_W, -factor_E_D, factor_E_TR, -factor_E_W]).T,
                degrees=True
            )
            self['Wave direction swan'] = self['Wave direction swan'].round(3)
            
            # When there is no wave energy from any of the components, use the wind direction
            idx = (factor_E_W + factor_E_TR + factor_E_D) == 0
            self.loc[idx, 'Wave direction swan'] = self.loc[idx, 'Wind direction']
            
    def read_results(self, step, progress_function=None):
        """
        Read results from SWAN calculation at result locations

        Parameters
        ----------
        step : string
            Current step
        """

        if not self.initialized:
            raise RuntimeError('Calculation results have not been initialized.')

        # Get calculation settings
        if step not in self.settings['swan']['calculations']:
            raise KeyError('Step {} not present in calculation settings'.format(step))

        # Get calculation and results folder
        folder = self.settings['swan']['calculations'][step]['folder']

        # Collect the results
        results = []
        nlocations = len(self.result_locations)
        zeros = np.zeros((nlocations, 4))
        dijknormalen = self.result_locations.sort_values(by='Naam')['Normaal'].array
        windrichting = self.iteration_results['Wind direction'].to_dict()

        # Create a list of coordinates to check
        check_crds = np.vstack([(row.geometry.x, row.geometry.y) for row in self.result_locations.sort_values(by='Naam').itertuples()]).round(4)

        for hydraulicloadid, row in self.iteration_results.sort_index().iterrows():

            if progress_function is not None:
                progress_function(1)

            # If no waves for the load combination
            if (row['Hs rand 3'] == 0.0) or (row['Tp,s rand 3'] == 0.0):
                results.append(np.zeros((nlocations, 4)))
                continue

            # Get case id
            case_id = self.description[hydraulicloadid]

            # Read 2D spectral series file from SWAN spectral file
            fpath = os.path.join(folder, 'spectra', case_id + '.s2d')
            if not os.path.exists(fpath):
                raise OSError('Spectrum file for case {} not found!'.format(case_id))
                # zeros[:, -1] = windrichting[hydraulicloadid]
                # results.append(zeros)
                # continue

            T, crds = self.swanio.read_output_spectrum(fpath)
            if len(crds) != len(check_crds):
                raise ImportError('The number of locations in the SWAN Spectrum differ from the number of HB Havens locations.')

            # Compare coordinates from SWAN file with result locations
            diff = np.absolute(crds - check_crds)
            if (diff > 0.1).any():
                # Try to connect HBH locs ans SWAN locs with KDTree
                distance, order = KDTree(crds).query(check_crds)
                # Count matches
                u, c = np.unique(order, return_counts=True)
                if any(u[c > 1]):
                    raise IndexError('The SWAN output locations do not match the HB Havens result locations. Cannot find unique pairs based on nearest neighbours.')
                if any(distance > 1):
                    logger.warning('Some location pairs are more than 1 meter apart.')
            else:
                order = np.arange(len(crds))
                
            # Correct incoming waves for levee normals
            if self.settings['swan']['use_incoming_wave_factors']:
                fac = incoming_wave_factors(T.direction, dijknormalen[np.argsort(order)])

                # Correct energy with incoming wave factor for locations and directions
                # Evaluates the Einstein summation over the original energy table
                # The table has dimensions: Time, Location, Frequentie, Direction
                # The following function multiplies the second and fourth dimension (j, k) with the factors
                a = np.einsum('ijkl,jl->ijkl', T.energy, fac)
                T.energy = np.ma.masked_array(a, T.energy.mask)
                
            # Get the wave parameters for all locations
            results.append(np.r_[T.Hm0(), T.Tp_smooth(), T.Tmm10(), T.Theta0()].T[order])

        # Before appending the results, make sure the locations are sorted such that the
        # locations are in the order that the results are read.
        # This is the location id, not the nae, since the order does not have to be alphabetic!
        self['before_order'] = list(range(len(self)))
        self.sort_values(by=['HydraulicLoadId', 'Location'], inplace=True)
        self[['Hm0_' + step, 'Tp_' + step, 'Tmm10_' + step, 'Theta0_' + step]] = np.vstack(results).round(3)
        self.sort_values(by='before_order', inplace=True)
        self.drop('before_order', axis=1, inplace=True)

        # if last step, calculate total energy
        if step == 'W':
            self.calculate_total_energy()

class Swan:
    """
    Swan calculation class

    TODO: Suggesties Matthijs Benit voor verbetering methode:
    - Na iteratiestap 1 checken welke windrichtingen relevant zijn om verder te itererern
    - Kollomnamen in iteratietabellen verduidelijken
    - Voorkomen weggooien bestanden bij genereren invoer?
    - Windsnelheid 0.1 bij D (diffractie). Dit convergeert slecht, zou ook opgelost kunnen worden met wind = 0.0 en quad OFF
    """

    def __init__(self, parent):
        """
        Class Swan

        Parameters
        ----------
        mainmodel : MainModel
            Pointer to the mainmodel object
        """
        self.mainmodel = parent

        # Get swan settings
        self.settings = parent.project.settings

        # Add SwanIO class
        self.swanio = SwanIO()

        # Give the name of the result parameters for export
        self.result_parameters = {
            'h': 'h',
            'Hs' : 'Hm0 swan',
            'Tp' : 'Tp swan',
            'Tm-1,0' : 'Tm-1,0 swan',
            'Wave direction': 'Wave direction swan',
        }

        # Set location changed to False (default)
        self.location_changed = False

        # Give the iteration table columns
        self.iter_columns = {
            'I1': ['depth', 'Hs steunpunt 1', 'Tp,s steunpunt 1', 'alpha 1', 'beta 1', 'Hs rand 2', 'Tp,s rand 2'],
            'I2': ['Hs steunpunt 2', 'Tp,s steunpunt 2', 'alpha 2', 'beta 2', 'Hs rand 3', 'Tp,s rand 3'],
            'I3': ['Hs steunpunt 3', 'Tp,s steunpunt 3'],
        }
        self.iteration_results = IterationResults(self)

        # Give the calculation table columns
        self.calc_columns = {
            'D': ['Hm0_D', 'Tmm10_D', 'Tp_D', 'Theta0_D'],
            'TR': ['Hm0_TR', 'Tmm10_TR', 'Tp_TR', 'Theta0_TR'],
            'W': ['Hm0_W', 'Tmm10_W', 'Tp_W', 'Theta0_W'],
        }
        self.calculation_results = CalculationResults(self)
        
        # Load U10 interpolation table
        self._load_Upot_to_U10()
        
    def load_from_project(self):
        """
        Load SWAN class (advanced calculation) from project
        """

        # Load iteration results
        iteration_results_path = os.path.join(self.mainmodel.project.filedir, 'swan_iteration_results.pkl')
        if os.path.exists(iteration_results_path):
            logger.info("..Loaded SWAN iteration table")
            self.iteration_results.load_from_pickle(iteration_results_path)
        
        # Load calculation results
        calculation_results_path = os.path.join(self.mainmodel.project.filedir, 'swan_calculation_results.pkl')
        if os.path.exists(calculation_results_path):
            logger.info("..Loaded SWAN calculation table")
            self.calculation_results.load_from_pickle(calculation_results_path)

    def save_tables(self):
        """
        Method to iteration and calculation table to pickle
        """
        if hasattr(self, 'iteration_results'):
            pd.DataFrame(self.iteration_results.loc[:, :]).to_pickle(
                os.path.join(self.mainmodel.project.filedir, 'swan_iteration_results.pkl')
            )
        if hasattr(self, 'calculation_results'):
            pd.DataFrame(self.calculation_results.loc[:, :]).to_pickle(
                os.path.join(self.mainmodel.project.filedir, 'swan_calculation_results.pkl')
            )

    def generate(self, step, progress_function=None):
        """
        Create all input files for a SWAN step, in three steps:
        1. Create SWAN folder,
        2. Create location files
        3. Create input files

        Parameters
        ----------
        step : string
            Calculation step, one of: 'I1', 'I2', 'I3', 'D', 'TR', 'W'
        progress_bar : QtWidgets.QProgressBar
            Class to which the progress can be passed.
        """

        # Check if SWAN folder is defined
        if not self.settings['swan']['swanfolder']:
            raise AttributeError('Swan folder is not defined')
        else:
            swan_folder = self.settings['swan']['swanfolder']
        
        # Add Tp if it is not in the database
        if 'Tp' not in self.mainmodel.hydraulic_loads.columns:
            self.iteration_results['Tp'] = self.iteration_results['Tm-1,0'] * self.settings['swan']['factor Tm Tp']

        # Check input
        if step not in ['I1', 'I2', 'I3', 'D', 'TR', 'W']:
            raise ValueError('Step "{}" not recognized'.format(step))

        # Set folder path according to iteration or calculation
        if step.startswith('I'):
            folder = os.path.join(swan_folder, 'iterations', step)
        else: 
            folder = os.path.join(swan_folder, 'calculations', step)

        
        self.swanio.createSWANFolder(folder)
        if progress_function is not None:
            progress_function(1)

        if step.startswith('I'):
            # support location
            self.swanio.writeSupportLocation(os.path.join(folder, 'points'), self.mainmodel.schematisation.support_location)
        else:
            locations = self.mainmodel.schematisation.result_locations
            self.swanio.writeLocations(os.path.join(folder, 'points') , locations)

        # Write locations to sub directory points
        if progress_function is not None:
            progress_function(1)

        # Create all input files for this iteration SWAN
        self.create_input_files(step, folder, progress_function=progress_function)

        # Adjust settings
        self.settings['swan']['calculations'][step]['folder'] = folder
        self.settings['swan']['calculations'][step]['input_generated'] = True
                
    def iteration_finished(self):
        self.calculation_results.initialize()

    def create_input_files(self, step, folder, progress_function=None):
        """
        Create all input files for SWAN

        Parameters
        ----------
        step : string
            Calculation step {'I1', 'I2', 'I3'. 'D', 'TR', 'W'}
        folder : string
            Name of the input folder
        progress_bar : QtWidgets.QProgressBar
            Class to which the progress can be passed.

        """

        # path of swanrun.bat
        swanrun = os.path.join(self.mainmodel.datadir, 'swan', 'swanrun.bat')

        # copy master template, depth file and swanrun.bat to swanfolder
        self.swanio.copyFiles(
            swan_folder=folder,
            master=self.settings['swan']['mastertemplate'],
            bathymetry=self.settings['swan']['depthfile'],
            swanrun=swanrun
        )

        # Read master file
        self.swanio.readMaster(os.path.join(folder, 'inputs', 'masters', 'master.swn'))
        # Check if all replace strings are present
        self.swanio.check_master(['[NR]' , '[LEV]', '[U10]', '[UDIR]', '[HM0]', '[TP]', '[DIR]', '[CASE_ID]'])

        # counter
        nr = 0

        # for all hydraulic loads
        for hydrodynamicdataid, hydraulic_load in self.iteration_results.iterrows():
            if progress_function is not None:
                progress_function(1)

            if hydraulic_load['Hs'] == 0 or hydraulic_load['Tm-1,0'] == 0:
                continue

            # add to counter
            nr += 1

            # make case id
            case_id = self.iteration_results.description[hydrodynamicdataid]

            # Get master file
            self.swanio.getMaster()

            # Update master file
            self.swanio.updateMaster("[NR]", "'{}'".format(str(nr).zfill(4)))
            waterlevel = hydraulic_load['Water level'] if not np.isnan(hydraulic_load['Water level']) else hydraulic_load['h']
            self.swanio.updateMaster("[LEV]", f'{waterlevel:.3f}')
            self.swanio.updateMaster("[CONV]", 'NAUTical')

            # Wind
            u10 = self._calc_U10(hydraulic_load['Wind speed'])

            if step.startswith('I'):
                # Without reflection
                self.swanio.setNoReflection()
            elif step == 'D':
                # Without wind
                u10 = 0.1
                # Without transmission
                self.swanio.setNoTransmision()
            elif step == 'W':
                # With wind without transmissie
                self.swanio.setNoTransmision()
            
            if u10 <= 0:
                u10 = 0.1

            # Wind
            self.swanio.updateMaster("[U10]", str(u10))
            self.swanio.updateMaster("[UDIR]", str(hydraulic_load['Wind direction']))

            # Golven
            if step == 'I1':

                # Initial simulation:
                self.swanio.updateMaster("[HM0]", str(hydraulic_load['Hs']))
                self.swanio.updateMaster("[TP]", str(hydraulic_load['Tp']))

            elif step == 'I2':
                # Correctiefactoren afleiden op basis van de eerste
                # modelberekening en definiëren van de nieuwe
                # belastingen op de modelrand

                hsig_swan = self.iteration_results.at[hydrodynamicdataid, 'Hs rand 2']
                tps_swan = self.iteration_results.at[hydrodynamicdataid, 'Tp,s rand 2']

                self.swanio.updateMaster("[HM0]", str(hsig_swan))
                self.swanio.updateMaster("[TP]", str(tps_swan))

            else:

                hsig_swan = self.iteration_results.at[hydrodynamicdataid, 'Hs rand 3']
                tps_swan = self.iteration_results.at[hydrodynamicdataid, 'Tp,s rand 3']

                self.swanio.updateMaster("[HM0]", str(hsig_swan))
                self.swanio.updateMaster("[TP]", str(tps_swan))

            self.swanio.updateMaster("[DIR]", str(hydraulic_load['Wave direction']))
            self.swanio.updateMaster("[CASE_ID]", case_id)

            self.swanio.writeMaster(os.path.join(folder ,'inputs', case_id + '.swn'))

        # Create batch file for calculation
        fpath = os.path.join(folder, 'runcases.bat')
        indices = ((self.iteration_results['Hs'] != 0.0) & (self.iteration_results['Tm-1,0'] != 0.0)).values.squeeze()
        with open(fpath, 'w') as fp:
            # Add swanrun.bat to each (modified) description, and write the lines to file
            fp.write('\n'.join(('call swanrun.bat ' + self.iteration_results.loc[indices, 'Load combination'].str.replace('=', '_').str.replace(' ', '_')).tolist()))

    def _load_Upot_to_U10(self):
        """
        Function to load conversion table from potential wind speed (present
        in HRD's) to U10, which is needed for the fetch calculation.
        """

        filepath = os.path.join(self.mainmodel.datadir, 'Up2U10', 'Up2U10.dat')
        self.windtrans = pd.read_csv(filepath, comment='%', header=None, delim_whitespace=True, names=['Upot', 'U10'])

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
