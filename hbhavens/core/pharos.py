
import itertools
import json
import logging
import os
import re
from math import gamma, isclose

import numpy as np
import pandas as pd
from scipy.optimize import newton

from hbhavens import io
from hbhavens.core.geometry import average_angle, nau2car
from hbhavens.core.spectrum import (JONSWAPSpectrum, Spectrum2D,
                                    incoming_wave_factors, jonswap, jonswap_2d)

g = 9.81

logger = logging.getLogger(__name__)

class Pharos:

    def __init__(self, parent):
        """
        Class Pharos

        Parameters
        ----------
        parent : MainModel
            Pointer to the mainmodel object
        """
        self.mainmodel = parent
        self.supportlocation = self.mainmodel.schematisation.support_locations
        self.hydraulic_loads = self.mainmodel.hydraulic_loads

        # Add PharosIO class
        self.pharosio = io.pharos.PharosIO(self)

        # Give the name of the result parameters for export
        self.result_parameters = {
            'h': 'h',
            'Hs' : 'Hs totaal',
            'Tp' : 'Tp totaal',
            'Tm-1,0' : 'Tm-1,0 totaal',
            'Wave direction': 'Wave direction totaal',
        }

        # Get all settings
        self.settings = parent.project.settings
        
        # Initialize tables
        self.spectrum_table = pd.DataFrame()
        columns = [
            'Location', 'LocationId', 'Load combination', 'HydraulicLoadId', 'h', 'Hs', 'Tp', 'Tm-1,0', 'Wave direction', 'Wind direction', 
            'Wind speed', 'Water level', 'Hs pharos', 'Tp pharos', 'Tm-1,0 pharos', 
            'Wave direction pharos', 'X', 'Y', 'Normaal', 'Hs totaal', 'Tp totaal', 'Tm-1,0 totaal', 'Wave direction totaal',
            'Hm0 swan', 'Tp swan', 'Tm-1,0 swan', 'Wave direction swan',
        ] 
        self.calculation_results = pd.DataFrame(columns=columns)

        # Empty objects
        self.grid_names = {}
        self.project_names = {}
        self.output_location_indices = {}

        self.result_locations = self.mainmodel.schematisation.result_locations
        
    def get_frequency_range(self):
        """
        Determine frequency range for based on hydraulic loads,
        the presence of Tp or the factor Tp-Tm-1,0.
        """

        if 'Tp' in self.hydraulic_loads.columns:
            # Get min and max
            Tp = self.hydraulic_loads['Tp'].values.squeeze()
            Tp_min = Tp[Tp > 0.0].min()
            Tp_max = Tp.max()
        
        elif 'Tm-1,0' in self.hydraulic_loads.columns:
            # Get factor
            factor = self.settings['pharos']['hydraulic loads']['factor Tm Tp']
            # Get min and max
            Tm_10 = self.hydraulic_loads['Tm-1,0'].values.squeeze()
            Tp_min = Tm_10[Tm_10 > 0.0].min() * factor
            Tp_max = Tm_10.max() * factor
        
        else:
            raise ValueError('Hydraulic loads are not loaded, or do not contain either Tp or Tm-1,0')

        # Calculate frequencies
        # Min frequency = 0.7 times the lowest frequency
        f_min = 0.7 * (1. / Tp_max)
        # Max frequency = 3 times the highest frequency
        f_max = 3.0 * (1. / Tp_min)

        return f_min, f_max

    def init_calculation_table(self):
        """
        Initialize table with calculation results for each combination of
        hydraulic loads and result locations.
        """
        # Determine individual and multiindex
        if self.result_locations.empty:
            return None

        if self.hydraulic_loads.empty:
            return None

        # Get location names and id's
        self.nlocations = len(self.result_locations)
        
        # # Get load ids
        self.hydraulicloadids = self.hydraulic_loads.index.astype(int)
        self.nloads = len(self.hydraulicloadids)

        # Determine load ids for which there are no input waves (and no SWAN calculations have been carried out)
        idx = (self.hydraulic_loads['Hs'] == 0.0)
        self.no_wave_ids = self.hydraulicloadids[idx]

        # Empty calculation results, in case it needs to be refilled
        if not self.calculation_results.empty:
            self.calculation_results.iloc[:, 0] = np.nan
            self.calculation_results.dropna(inplace=True)

        # Add index and columns to table
        self.calculation_results['Location'] = np.repeat(self.result_locations['Naam'], self.nloads)
        self.calculation_results['LocationId'] = np.repeat(self.result_locations.index.values, self.nloads)
        
        self.calculation_results['HydraulicLoadId'] = np.tile(self.hydraulicloadids, self.nlocations)
        # Add load combination description
        desc_dict = self.mainmodel.hydraulic_loads.description_dict
        self.calculation_results['Load combination'] = [desc_dict[hlid] for hlid in self.calculation_results['HydraulicLoadId']]

        # Get load parameters
        load_parameters = self.hydraulic_loads.columns.intersection(['h', 'Hs', 'Tp', 'Tm-1,0', 'Wave direction', 'Wind direction', 'Wind speed', 'Water level'])
            
        # Add values from iterations to calculation results
        tiled = np.tile(self.hydraulic_loads[load_parameters].values, (self.nlocations, 1))
        self.calculation_results[load_parameters] = tiled
        
        # Add geometry
        # Get x, y and normaal per location, and repeat this for all lead combinations
        x_y_norm =  np.round([[loc.geometry.x, loc.geometry.y, loc.Normaal] for loc in self.result_locations.itertuples()], 3)
        self.calculation_results[['X',  'Y', 'Normaal']] = np.repeat(x_y_norm, self.nloads, axis=0)

        # # Add columns for result
        self.calculation_results[['Hs pharos', 'Tp pharos', 'Tm-1,0 pharos', 'Wave direction pharos']] = np.nan

        
        self.calculation_results.reset_index(inplace=True, drop=True)


    def fill_spectrum_table(self):
        """
        Method to fill the spectrum table, after the input frequencies and
        directions are known
        """

        # Empty dataframe if it has any content (do not overwrite!)
        if not self.spectrum_table.empty:
            self.spectrum_table.iloc[:, 0] = np.nan
            self.spectrum_table.dropna(inplace=True)
            self.spectrum_table.drop(self.spectrum_table.columns, inplace=True, axis=1)

        # Add columns (directions)
        for col in self.theta:
            self.spectrum_table[col] = np.nan
        
        # Set index (frequencies)
        self.spectrum_table['f'] = self.f
        self.spectrum_table.set_index('f', inplace=True)
        
        # Fill in values
        self.spectrum_table[self.theta] = self.flags
        

    def load_from_project(self):
        """
        Load Pharos class (advanced calculation) from project
        First some often 
        """
        path_settings = self.settings['pharos']['paths']

        # Initialize spectrum
        if self.settings['pharos']['initialized']:
            # Initializing will overwrite the checkes frequencties,
            # which is nog necessary on start up. So load and reset
            freqs = self.settings['pharos']['frequencies']['checked'][:]
            self.initialize()
            self.settings['pharos']['frequencies']['checked'] = freqs

        # Load spectrum table is it exists
        spectrum_table_path = os.path.join(self.mainmodel.project.filedir, 'pharos_spectrum_table.pkl')
        if os.path.exists(spectrum_table_path):
            logger.info("..Loaded PHAROS spectrum table")
            self.spectrum_table = pd.read_pickle(spectrum_table_path)

        # Load calculation table if it exists
        calculation_results_path = os.path.join(self.mainmodel.project.filedir, 'pharos_calculation_results.pkl')
        if os.path.exists(calculation_results_path):
            logger.info("..Loaded PHAROS calculation table")
            self.calculation_results.iloc[:, 0] = np.nan
            self.calculation_results.dropna(inplace=True)

            df = pd.read_pickle(calculation_results_path)
            for name, col in df.reindex(columns=self.calculation_results.columns).iteritems():
                self.calculation_results[name] = col
            self.calculation_results.index = df.index

    def save_tables(self):
        """
        Method to save spectrum and calculation table to pickle
        """
        if hasattr(self, 'calculation_results'):
            self.calculation_results.to_pickle(os.path.join(self.mainmodel.project.filedir, 'pharos_calculation_results.pkl'))
        if hasattr(self, 'spectrum_table'):
            self.spectrum_table.to_pickle(os.path.join(self.mainmodel.project.filedir, 'pharos_spectrum_table.pkl'))

    def initialize(self):
        """
        Initialize Pharos 2d spectrum

        Returns
        -------
        flags : numpy.array
            Contribution flags
        """

        # Determine max Tp
        factor_tp_tm = self.settings['pharos']['hydraulic loads']['factor Tm Tp']
        
        # Frequencies
        if self.settings['pharos']['frequencies']['scale'] == 'lineair':
            self.f = np.linspace(
                float(self.settings['pharos']['frequencies']['lowest']),
                float(self.settings['pharos']['frequencies']['highest']),
                self.settings['pharos']['frequencies']['number of bins']
            )
        elif self.settings['pharos']['frequencies']['scale'] == 'logaritmisch':
            self.f = np.logspace(
                float(np.log10(self.settings['pharos']['frequencies']['lowest'])),
                float(np.log10(self.settings['pharos']['frequencies']['highest'])),
                self.settings['pharos']['frequencies']['number of bins']
            )
        else:
            raise ValueError('Scale not recognized, choose either lineair of logaritmisch.')

        # Wave directions
        step = self.settings['pharos']['wave directions']['bin size']
        self.theta = np.arange(
            float(self.settings['pharos']['wave directions']['lowest']),
            float(self.settings['pharos']['wave directions']['highest']),
            float(step)
        )

        # Initialize Pharos spectrum
        self.spectrum = JONSWAPSpectrum(self.f, self.theta)

        # Calculate jonswap 2D
        self.spread = self.settings['pharos']['2d wave spectrum']['spread']
        self.gamma = self.settings['pharos']['2d wave spectrum']['gamma']

        self.determine_relevant_bins()

        # Check all frequencies with energy > 0.0
        self.settings['pharos']['frequencies']['checked'] = list(
            itertools.compress(self.f, self.flags.sum(axis=1) > 0)
        )

        # Change settings
        self.settings['pharos']['initialized'] = True

    def determine_relevant_bins(self):
        """
        Method to determine relevant bins in hydraulic loads
        """

        # Create empty array to stack all JONSWAP spectra
        energy_per_load = np.zeros((len(self.hydraulic_loads), len(self.f), len(self.theta)), dtype=bool)

        # Get Tp
        if 'Tp' in self.hydraulic_loads.index:
            Tp_array = self.hydraulic_loads['Tp'].values
        else:
            # Determine Tp in case not present in database
            Tp_array = self.hydraulic_loads['Tm-1,0'].values * self.settings['pharos']['hydraulic loads']['factor Tm Tp']

        # Assign minimum energy to variable
        Smin = self.settings['pharos']['2d wave spectrum']['min energy']

        # Loop trough conditions
        for i, (Tp, (idx, hydraulic_load)) in enumerate(zip(Tp_array, self.hydraulic_loads.iterrows())):

            Hm0 = hydraulic_load['Hs']
            Theta = hydraulic_load['Wave direction']

            # Only if energy present
            if Hm0 != 0:

                S = jonswap_2d(
                    f=self.f,
                    theta=self.theta,
                    S_1d=jonswap(self.f, Hm0, Tp, gamma=self.gamma),
                    Hm0=Hm0,
                    Tp=Tp,
                    Theta=Theta,
                    spread=self.spread
                )

                energy_per_load[i] = S > Smin
        
        self.flags = np.sum(energy_per_load, axis=0)
        

    def generate(self, progress_function=None):
        """
        Create all input files for a Pharos calculation

        1. Create Pharos folder,
        2. Create input files

        Parameters
        ----------
        progress_function : QtWidgets.QProgressBar
            Class to which the progress can be passed.
        """

        # Check if Pharos folder is defined
        if not self.settings['pharos']['paths']['pharos folder']:
            raise AttributeError('Pharos folder is not defined')

        if not self.settings['pharos']['paths']['schematisation folder']:
            raise AttributeError('Pharos schematisationsfolder is not defined')

        pharos_settings = self.settings['pharos']
        for _, directions in pharos_settings['schematisations'].items():
            # Continue if no selected directions
            if not directions:
                continue
            
        # Create input files
        self.create_input_files(progress_function=progress_function)

        # Set finished on True
        self.settings['pharos']['input_generated'] = True

    def calc_wave_length(self, h, T):
        """
        Calculate wave length based on water depth and return period
        """
        # Calculate omega
        omega = 2 * np.pi / T
        
        # Define function to optimize
        def calc_f(k, g, h, omega):
            f = g * k * np.tanh(k * h) - omega ** 2
            return f

        # Returns wavenumber of the gravity wave dispersion relation using
        # newtons method. The initial guess is shallow water wavenumber.
        k = newton(
            calc_f,
            x0=omega / np.sqrt(g),
            args=(g, h, omega),
            maxiter=100
        )
            
        # Calculate wave length
        wave_length = 2 * np.pi / k

        return wave_length


    def create_input_files(self, progress_function=None):
        """
        Creaste Pharos project files

        Parameters
        ----------
        schematisation : string
            Current Pharos schematisation
        frequencies : np.ndarray
            Array of angular frequency
        directions : np.ndarray
            Array of directionally bins
        water_levels : np.ndarray
            Array of water levels
        """

        iterables = self.get_iterables()
        frequencies = iterables[0]
        water_levels = iterables[1]

        combinations = self.get_combinations()

        # Determine project and grid names
        self.get_proj_and_grid()

        # For each schematisation
        for schematisation, directions in self.settings['pharos']['schematisations'].items():

            # Pharos schematisation
            src_folder = os.path.join(self.settings['pharos']['paths']['schematisation folder'], schematisation)
            
            # Zoek de template
            pharcon_src = [os.path.join(src_folder, item) for item in os.listdir(src_folder) if 'pharcon' in item and item.endswith('.inp')]
            if len(pharcon_src) == 0:
                raise OSError('Geen pharcon template gevonden voor schematisatie "{}".\nZoeklocatie: {}'.format(schematisation, src_folder))
            elif len(pharcon_src) > 1:
                raise OSError('Meer dan één pharcon template gevonden voor schematisatie: {}.\nZoeklocatie: {}'.format(schematisation, src_folder))
            with open(pharcon_src[0], 'r') as f:
                pharcon_template = f.read()

            # Destination paths
            dst_folder = os.path.join(self.settings['pharos']['paths']['pharos folder'], 'calculations', schematisation)
            if not os.path.exists(dst_folder):
                os.makedirs(dst_folder)
            pharos_pha_path = os.path.join(dst_folder,  'projects.pha')

            # Copy schematisation        
            self.pharosio.create_pharos_folder(src_folder, dst_folder)

            gstat = 10
            rstat = 2

            # Write coordinate file
            self.pharosio.write_location_file(schematisation)

            # Write bat files
            self.pharosio.write_bat_files(schematisation)
            
            # Write first lines of pharos.pha file
            with open(pharos_pha_path, 'w') as pharos_pha:
                pharos_pha.write("@@PROJECT {}\n".format(self.project_names[schematisation]))
                pharos_pha.write("@@PDESC \n")
                pharos_pha.write("@@GRID {}\n".format(self.grid_names[schematisation]))
                pharos_pha.write("@@GDESC \n")
                pharos_pha.write("@@GSTAT {}\n".format(gstat))
                pharos_pha.write("@@FDIR {}\n".format(os.path.normpath(os.path.dirname(dst_folder))))


            # Create pharcon
            pharcon_file = pharcon_template[:]
            # Replace line in pharos template
            first_line_entries = pharcon_file[:pharcon_file.find('\n')].strip().split()

            # Find the line that follows the settings and the wave periods
            split_lines = re.findall('(\n\s*[0-9]+ +[0-9]+\s*\n)', pharcon_file)
            second_line, bottom_line = split_lines[0], split_lines[1]

            for frequency, water_level, sch, direction in combinations:
                if sch != schematisation:
                    continue

                # 1. Peak period
                first_line_entries[0] = '{:.4f}'.format(1./frequency)
                # 2. Number of frequencies (0, since no spectrum)
                first_line_entries[1] = '0'
                # 3. Number of wave directions
                first_line_entries[2] = '1'
                # 4. ?
                # 5. Water level
                first_line_entries[4] = '{:.4f}'.format(water_level)

                # Determine scenario name
                run = 'T{:.3f}D{:05.1f}H{:.3f}'.format(1./frequency, direction, water_level).replace('.', 'p')

                # Add to projects.pha
                with open(pharos_pha_path, 'a') as pharos_pha:
                    pharos_pha.write("@@RUN {}\n".format(run))
                    pharos_pha.write("@@RDESC Direction:{:.1f} Frequency:{:.3f} /s WaterLevel:{:.3f} m\n".format(direction, frequency, water_level))
                    pharos_pha.write("@@RSTAT {}\n".format(rstat))

                # Add first line and the rest
                pharcon_file = pharcon_template[:]
                pharcon_file = ' '.join(first_line_entries) + pharcon_file[pharcon_file.find(second_line):pharcon_file.find(bottom_line)]
                
                # Find end of object definition
                endstring = '123456789012345678901234567890'
                pharcon_file += '\n{}\n1\n{:.4f} 1.0000 0.0000\n'.format(bottom_line.strip(), nau2car(direction))+'\n'.join([endstring]*3)+'\n'

                # Write to file    
                pharcon_dst = os.path.join(dst_folder, '{}_pharcon_{}_{}.inp'.format(self.project_names[schematisation], self.grid_names[schematisation], run))
                with open(pharcon_dst, 'w') as f:
                    f.write(pharcon_file)

                # Write input file
                input_file = 'project:{project}\ngrid:{grid}\nrun:{run}\nradius:0.15 %radius\n1 %images\ndirections:0\nfrequencies:0\ncordinates:coordinates.csv'.format(
                    project=self.project_names[schematisation],
                    grid=self.grid_names[schematisation],
                    run=run
                )
                input_file_dst = os.path.join(dst_folder, 'input_{}.txt'.format(run))
                with open(input_file_dst, 'w') as f:
                    f.write(input_file)

                # Progress bar
                if progress_function is not None:
                    progress_function(1)
    
    def get_iterables(self):
        """
        Return frequencies, water levels, schematisations and directions
        to iterate over
        """
        # Get frequencies, find nearest values in self for exact match
        frequencies = []
        for frequency in self.f:
            if any((isclose(f, frequency) for f in self.settings['pharos']['frequencies']['checked'])):
                frequencies.append(frequency)

        # Get water levels
        water_levels = self.settings['pharos']['water levels']['checked']
        # Combinations of schematisations and directions
        schematisations, directions = [], []
        for sch, dirs in self.settings['pharos']['schematisations'].items():
            for d in dirs:
                schematisations.append(sch)
                directions.append(d)
        
        return frequencies, water_levels, schematisations, directions

    def get_combinations(self):
        """
        Get unique combinations. For each unique combination of the iterables,
        check if the direction should be calculated with the schematisation
        and if the bin has energy
        """
        # Get iterables
        iterables = self.get_iterables()
        # Get schematisations from settings (dictionary with directions)
        schematisations = self.settings['pharos']['schematisations']
        # Create combinations, and select unique values
        combinations = []

        # loop trough unique combinations
        for frequency, water_level, schematisation, direction in set(itertools.product(*iterables)):
            # Skip if direction should not be calculated with schematisation
            if direction not in schematisations[schematisation]:
                continue

            # TODO water level dependency

            # Skip if frequency-direction bin has no energy
            if not self.spectrum_table.at[frequency, direction]:
                continue
            
            # Add to combinations
            combinations.append([frequency, water_level, schematisation, direction])
       
        return combinations

    def get_proj_and_grid(self):
        """
        Method to determine the project name and grid name for
        the schematisations in the settings
        """
        for schematisation in self.settings['pharos']['schematisations'].keys():
            # Look for bottom file
            sch_folder = os.path.join(self.settings['pharos']['paths']['schematisation folder'], schematisation)
            for f in os.listdir(sch_folder):
                if '_editgr_' in f:
                    editgrfile = f
                    break
            else:
                raise OSError('No bottomfile found in "{}" to determine project and grid name.'.format(sch_folder))

            # Derive names from file
            project, grid = os.path.splitext(editgrfile)[0].split('_editgr_')

            # Assign to dictionaries
            self.project_names[schematisation] = project
            self.grid_names[schematisation] = grid 

    def read_output_locations(self):
        """
        Method to read the location-coordinates from a file
        
        Parameters
        ----------
        json_output : string
            Output string where the locations can be read from
        """
        # Empty dataframe
        self.output_location_indices.clear()

        # Collect all locations
        schematisations = np.unique(self.get_iterables()[2])

        # Get coordinates of result locations
        result_loc_crds = np.vstack([pt.geometry.coords[0] for pt in self.result_locations.itertuples()])

        # Correct for PHAROS model offset
        result_loc_crds[:, 0] += self.settings['pharos']['transformation']['dx']
        result_loc_crds[:, 1] += self.settings['pharos']['transformation']['dy']

        for schematisation in schematisations:

            dst_folder = os.path.join(self.settings['pharos']['paths']['pharos folder'], 'calculations', schematisation)
            
            # Read coordinate file
            with open(os.path.join(dst_folder, 'coordinates.csv'), 'r') as f:
                lines = f.readlines()
            
            # For coordinate line in file
            for line in lines:

                # Get coordinate in result locations
                crd = np.asarray(line.split(','), dtype=float)
                # Find distances between file location and result locations
                dists = np.hypot(*(result_loc_crds - crd).T)
                if dists.min() > 2.0:
                    logger.warning('The distance between a Pharos location and a HB Havens location is larger than 2 meters.')

                # Add file (line) location to dictionary, be looking for the argmin of the distance
                # TODO: Netjes oplossen
                self.output_location_indices[(int(round(crd[0])), int(round(crd[1])))] = dists.argmin()
            
    def read_calculation_results(self, progress_function=None):
        """
        Method to read PHAROS RDPA postprocessed output
        """
        # Determine project and grid names
        self.get_proj_and_grid()

        # Read output locations
        self.read_output_locations()

        # Loop trough output directories and load all output
        iterables = self.get_iterables()

        # Find direction bin for output direction
        binwidth = abs(self.theta[1] - self.theta[0])

        # Create an empty list for each water level
        waves = {w: [] for w in iterables[1]}

        for i, (frequency, water_level, schematisation, direction) in enumerate(self.get_combinations()):
            # output folder
            dst_folder = os.path.join(
                self.settings['pharos']['paths']['pharos folder'],
                'calculations',
                schematisation,
                'output'
            )
            
            # output file
            run = 'T{:.3f}D{:05.1f}H{:.3f}'.format(1./frequency, direction, water_level)
            json_file = os.path.join(dst_folder, '{}_admin_{}_{}.json'.format(
                self.project_names[schematisation],
                self.grid_names[schematisation], run)
            )

            # Read json output
            try:
                with open(json_file, 'r') as f:
                    json_output = json.load(f)
            except Exception as e:
                raise IOError(f'Failed reading Pharos output file: "{os.path.split(json_file)[-1]}".\n{e}')

            # Loop through locations
            for locidx, loc_output in enumerate(json_output):
                # Find the location id in the list
                locidx = self.output_location_indices[
                    (int(round(loc_output['input_coordinates']['x'])), int(round(loc_output['input_coordinates']['y'])))]

                # Get waves
                count = 0
                for wave in loc_output['wave_heights']:
                    # Skip if no energy in wave
                    if wave['H'] == 0:
                        continue
                    elif wave['H'] > 3.0:
                        logger.warning('H > 3.0, assuming invalid, thus skipping.')
                        continue
                    elif wave['H'] > 2.0:
                        logger.warning('H > 2.0, perhaps invalid, but not skipping.')
                    
                    # Add wave
                    waves[water_level].append([frequency, direction, locidx, count, wave['H'], wave['naut']])
                    count += 1

            if progress_function is not None:
                progress_function(1)
    
                    
        # Make dataframe from waves
        self.waves = {}
        for wlev, wavelist in waves.items():
            # Create dataframe
            self.waves[wlev] = pd.DataFrame(wavelist, columns=['frequency', 'outer_direction', 'locidx', 'wave_id', 'H', 'inner_direction'])
            # Add direction bins
            self.waves[wlev]['inner_dir_bin'] = np.digitize((self.waves[wlev]['inner_direction'] + 0.5 * binwidth) % 360, self.theta + binwidth)
            # Add frequency bins
            self.waves[wlev]['frequency_bin'] = list(map({f: i for f, i in zip(self.f, np.arange(len(self.f)))}.get, self.waves[wlev]['frequency']))
            # Set index
            self.waves[wlev].set_index(['frequency', 'outer_direction'], inplace=True)
        

    def _add_zeros(self, results, winddirection, progress_function):
        zeros = np.zeros((self.nlocations, 4))
        zeros[:, -1] = winddirection
        results.append(zeros)
        # Set progress bar
        if progress_function is not None:
            progress_function(10)
    
        
    def assign_energies(self, progress_function=None):
        """
        Method to assign energies to hydraulic loads

        1. Determine for each wave condition what the input energy in the specific bin is
        2. Calculate the wave height from the resulting energy in the spectra at the locations

        What if one location 

        """
        # import timeit
        # st = timeit.default_timer()
        
        combinations = self.get_iterables()

        dijknormalen = self.mainmodel.schematisation.result_locations['Normaal'].values
        
        frequencies = np.unique(combinations[0])
        water_levels = np.unique(combinations[1])
        directions = np.unique(combinations[3])
        
        freq_indices = np.in1d(self.f, frequencies)
        direction_indices = np.in1d(self.theta, directions)

        # Create empty energy matrix
        energy = np.zeros((len(self.output_location_indices), len(self.f), len(self.theta)))

        # Determine bin widths of frequency and direction, for determining absolute energies
        f = np.r_[self.f[0] - (self.f[1] - self.f[0]), self.f, self.f[-1] + (self.f[-1] - self.f[-2])]
        f = (f[1:] + f[:-1]) / 2
        f_bins = np.diff(f)
        dtheta = np.deg2rad(self.theta[1] - self.theta[0])
            
        # Create empty list to collect results per load combination
        results = []

        for hydraulicloadid in sorted(self.hydraulicloadids):
        
            # Get load combination from hydraulic loads
            loadcombination = self.hydraulic_loads.loc[hydraulicloadid]

            # If water level is not calculated, set zero
            if loadcombination['Water level'] not in water_levels:
                
                # Wave parameters are 0, wave direction = wind direction
                self._add_zeros(results, loadcombination['Wind direction'], progress_function)
                continue

            # Get Tp
            if 'Tp' in loadcombination:
                Tp = loadcombination['Tp']
            else:
                Tp = loadcombination['Tm-1,0'] * self.settings['pharos']['hydraulic loads']['factor Tm Tp']

            # Determine spectrum for condition
            if loadcombination['Hs'] == 0.0:
                # Wave parameters are 0, wave direction = wind direction
                self._add_zeros(results, loadcombination['Wind direction'], progress_function)
                continue

            S_1d = jonswap(self.f,
                Hm0=loadcombination['Hs'],
                Tp=Tp,
                gamma=self.gamma
            )

            # Calculate energy spectrum and convert the energy densities to energies
            energy_spectrum_outside = jonswap_2d(
                f=self.f,
                theta=self.theta,
                S_1d=S_1d,
                Hm0=loadcombination['Hs'],
                Tp=Tp,
                Theta=loadcombination['Wave direction'],
                spread=self.spread
            ) * f_bins[:, None] * dtheta

            # Set energies where frequencies and directions are not defined (since not calculated) to zero
            energy_spectrum_outside[~freq_indices, :] = 0.0
            energy_spectrum_outside[:, ~direction_indices] = 0.0

            # Make a selection of frequentie direction tuples
            indices = np.where(energy_spectrum_outside > self.settings['pharos']['2d wave spectrum']['min energy'])
            
            # If no energies in this combination of frequencies and bins
            if not np.size(indices):
                # Wave parameters are 0, wave direction = wind direction
                self._add_zeros(results, loadcombination['Wind direction'], progress_function)
                continue

            tuples = list(zip(self.f[indices[0]], self.theta[indices[1]]))
            
            # Create dataframe with energies to join to dataframe with reflection waves
            energydf = pd.DataFrame(
                data=energy_spectrum_outside[indices],
                index=pd.MultiIndex.from_tuples(
                    tuples,
                    names=['frequency', 'outer_direction']
                ),
                columns=['outer_energy']
            )
            
            # Select relevant part of model results
            wavedf = self.waves[loadcombination['Water level']].loc[
                tuples, ['locidx', 'inner_dir_bin', 'frequency_bin', 'H']].join(energydf)
            
            # Determine energy inside harbor
            wavedf['inner_energy'] = wavedf['outer_energy'] * wavedf['H'] ** 2
            
            # Reset energy to zero again
            energy[:, :, :] = 0.0
            
            # Add all energies to spectrum
            for row in wavedf.itertuples():
                energy[row.locidx, row.frequency_bin, row.inner_dir_bin] += row.inner_energy
            
            # Create spectrum
            inner_spectrum = Spectrum2D(
                frequencies=self.f,
                directions=self.theta,
                energy=energy[np.newaxis, :, :, :]
            )

            # Correct incoming waves for levee normals
            if self.settings['pharos']['use_incoming_wave_factors']:    
                fac = incoming_wave_factors(self.theta, dijknormalen)
                inner_spectrum.energy = np.einsum('ijkl,jl->ijkl', inner_spectrum.energy, fac)

            # Determine wave parameters and add to result
            results.append(np.r_[inner_spectrum.Hm0(), inner_spectrum.Tp_smooth(), inner_spectrum.Tmm10(), inner_spectrum.Theta0()].T)
            
            if progress_function is not None:
                progress_function(10)

        # Before appending the results, make sure the locations are sorted such that the
        # locations are in the order that the results are read.
        # This is the location id, not the nae, since the order does not have to be alphabetic!
        self.calculation_results['before_order'] = list(range(len(self.calculation_results)))
        self.calculation_results.sort_values(by=['HydraulicLoadId', 'LocationId'], inplace=True)
        columns = ['Hs pharos', 'Tp pharos', 'Tm-1,0 pharos', 'Wave direction pharos']
        self.calculation_results[columns] = np.vstack(results).round(3)
        self.calculation_results.sort_values(by='before_order', inplace=True)
        self.calculation_results.drop('before_order', axis=1, inplace=True)

        if np.isnan(self.calculation_results[columns].values).any():
            raise ValueError('NaN values in PHAROS-data.')
        
        # Combine results with SWAN
        self.combine_with_swan()

    def combine_with_swan(self):
        """
        Class to combine calculation results with swan output.
        Prerequisite is that the swan results are filled.
        """

        # Copy results from calculation results
        pharos_columns = ['Hs pharos', 'Tp pharos', 'Tm-1,0 pharos', 'Wave direction pharos']
        merge_columns = ['Location', 'Load combination']
        
        # Merge with swan final results
        swan_results = self.mainmodel.swan.calculation_results
        swan_columns = ['Hm0 swan', 'Tm-1,0 swan', 'Tp swan', 'Wave direction swan']
        
        if np.isnan(swan_results[swan_columns].values).any():
            raise ValueError('NaN values in SWAN-data.')

        self.calculation_results[swan_columns] = self.calculation_results[merge_columns].merge(
            swan_results[swan_columns + merge_columns],
            on=merge_columns
        ).drop(merge_columns, axis=1).values

        if np.isnan(self.calculation_results[swan_columns + pharos_columns].values).any():
            raise ValueError('NaN values in combined SWAN and PHAROS data. This indicates an error in merging the results from both models.')

        # Calculate combined energies
        f_swan, f_pharos = (self.calculation_results[['Hm0 swan', 'Hs pharos']].fillna(0.0) ** 2).values.T
        
        # Calculate combined significant wave height
        self.calculation_results['Hs totaal'] = np.hypot(*self.calculation_results[['Hm0 swan', 'Hs pharos']].values.T)

        # Calculate combined Tm -1,0
        f_total = np.sum([f_swan, f_pharos], axis=0)
        idx = f_total > 0.0
        self.calculation_results.loc[idx, 'Tm-1,0 totaal'] = np.round(
            (self.calculation_results.loc[idx, 'Tm-1,0 swan'] * f_swan[idx] + self.calculation_results.loc[idx, 'Tm-1,0 pharos'] * f_pharos[idx]) / f_total[idx], 3)
        self.calculation_results.loc[~idx, 'Tm-1,0 totaal'] = 0.0
        
        # Calculate combined Tp
        self.calculation_results['Tp totaal'] = np.round(np.nanmax(self.calculation_results[['Tp swan', 'Tp pharos']], axis=1), 3)
        
        # Calculate combined wave direction
        self.calculation_results['Wave direction totaal'] = average_angle(
            angles=self.calculation_results[['Wave direction swan', 'Wave direction pharos']].values,
            factors=np.vstack([f_swan, f_pharos]).T,
            degrees=True
        )

        # Round to three decimals
        combined_columns = ['Hs totaal', 'Tm-1,0 totaal', 'Tp totaal', 'Wave direction totaal']
        self.calculation_results[combined_columns] = self.calculation_results[combined_columns].round(3)


class WavesModel:
    """
    Solves the wave dispersion relationship via Newton-Raphson.

    .. math::
        \omega^2 = gk\tanh kh

    Parameters
    ----------
    h : array_like, str
        Water depth [m] or 'deep', 'shallow' as keywords
    T : array_like
        Wave period [s]
    L : array_like
        Wave length [m]
    thetao : array_like
        TODO
    Ho : array_like
        TODO

    Returns
    -------
    omega : array_like
            Wave frequency
    hoLo, hoL, Lo, L, k, T, Co, C, Cg, G, Ks, Kr, theta, H

    Notes
    -----
    Compare values with:
    http://www.coastal.udel.edu/faculty/rad/wavetheory.html

    Examples
    --------
    from oceans.sw_extras import Waves
    wav = Waves(h=10, T=5, L=None)
    print("ho/Lo = %s" % wav.hoLo)
    ho/Lo = 0.256195119559
    print("ho/L  = %s" % wav.hoL)
    ho/L  = 0.273273564378
    print("Lo    = %s" % wav.Lo)
    Lo    = 39.0327497933
    print("L     = %s" % wav.L)
    L     = 36.5933676122
    print("k     = %s" % wav.k)
    k     = 0.171702844454
    print("omega = %s" % wav.omega)
    omega = 1.25663706144
    print("T     = %s" % wav.T)
    T     = 5.0
    print("C     = %s" % wav.C)
    C     = 7.31867352244
    print("Cg    = %s" % wav.Cg)
    Cg    = 4.47085819307
    print("G     = %s" % wav.G)
    G     = 0.22176735425
    wav = Waves(h=10, T=None, L=100)
    print("ho/Lo = %s" % wav.hoLo)
    ho/Lo = 0.05568933069
    print("ho/L  = %s" % wav.hoL)
    ho/L  = 0.1
    print("Lo    = %s" % wav.Lo)
    Lo    = 179.56760974
    print("L     = %s" % wav.L)
    L     = 100.0
    print("k     = %s" % wav.k)
    k     = 0.0628318530718
    print("omega = %s" % wav.omega)
    omega = 0.585882379881
    print("T     = %s" % wav.T)
    T     = 10.7243117782
    print("C     = %s" % wav.C)
    C     = 9.32460768286
    print("Cg    = %s" % wav.Cg)
    Cg    = 8.29120888868
    print("G     = %s" % wav.G)
    G     = 0.778350182802
    print("  = %s" % wav.Ks)
    print("  = %s" % wav.Kr)
    print("  = %s" % wav.theta)
    print("  = %s" % wav.H)
    print("  = %s" % wav.Ks)
    print("  = %s" % wav.Kr)
    print("  = %s" % wav.theta)
    print("  = %s" % wav.H)

    """
    def __init__(self, h, T=None, L=None, thetao=None, Ho=None, lat=None):
        self.T = np.asarray(T, dtype=np.float)
        self.L = np.asarray(L, dtype=np.float)
        self.Ho = np.asarray(Ho, dtype=np.float)
        self.lat = np.asarray(lat, dtype=np.float)
        self.thetao = np.asarray(thetao, dtype=np.float)

        if isinstance(h, str):
            if L is not None:
                if h == 'deep':
                    self.h = self.L / 2.
                elif h == 'shallow':
                    self.h = self.L * 0.05
        else:
            self.h = np.asarray(h, dtype=np.float)

        if lat is None:
            g = 9.81  # Default gravity.
        else:
            g = self.grav(lat, p=0)

        if L is None:
            self.omega = 2 * np.pi / self.T
            self.Lo = (g * self.T ** 2) / 2 / np.pi
            # Returns wavenumber of the gravity wave dispersion relation using
            # newtons method. The initial guess is shallow water wavenumber.
            self.k = self.omega / np.sqrt(g)
            # TODO: May change to,
            # self.k = self.w ** 2 / (g * np.sqrt(self.w ** 2 * self.h / g))
            f = g * self.k * np.tanh(self.k * self.h) - self.omega ** 2

            while np.abs(f.max()) > 1e-10:
                dfdk = (g * self.k * self.h *
                        (1 / (np.cosh(self.k * self.h))) ** 2 +
                        g * np.tanh(self.k * self.h))
                self.k = self.k - f / dfdk
                # FIXME:
                f = g * self.k * np.tanh(self.k * self.h) - self.omega ** 2

            self.L = 2 * np.pi / self.k
            if isinstance(h, str):
                if h == 'deep':
                    self.h = self.L / 2.
                elif h == 'shallow':
                    self.h = self.L * 0.05
        else:
            self.Lo = self.L / np.tanh(2 * np.pi * self.h / self.L)
            self.k = 2 * np.pi / self.L
            self.T = np.sqrt(2 * np.pi * self.Lo / g)
            self.omega = 2 * np.pi / self.T

        self.hoL = self.h / self.L
        self.hoLo = self.h / self.Lo
        self.C = self.omega / self.k  # or L / T
        self.Co = self.Lo / self.T
        self.G = 2 * self.k * self.h / np.sinh(2 * self.k * self.h)
        self.n = (1 + self.G) / 2
        self.Cg = self.n * self.C
        self.Ks = np.sqrt(1 / (1 + self.G) / np.tanh(self.k * self.h))

        if thetao is None:
            self.theta = np.NaN
            self.Kr = np.NaN
        if thetao is not None:
            self.theta = np.rad2deg(np.asin(self.C / self.Co *
                                            np.sin(np.deg2rad(self.thetao))))
            self.Kr = np.sqrt(np.cos(np.deg2rad(self.thetao)) /
                              np.cos(np.deg2rad(self.theta)))

        if Ho is None:
            self.H = np.NaN
        if Ho is not None:
            self.H = self.Ho * self.Ks * self.Kr

    def grav(lat, p=0):
        """
        Calculates acceleration due to gravity as a function of latitude and as
        a function of pressure in the ocean.

        Parameters
        ----------
        lat : array_like
              latitude in decimal degrees north [-90...+90]
        p : number or array_like. Default p = 0
            pressure [dbar]

        Returns
        -------
        g : array_like
            gravity [m s :sup:`2`]

        Notes
        -----
        In the ocean z is negative.

        Examples
        --------
        import gsw
        lat = [-90, -60, -30, 0]
        p = 0
        self.grav(lat, p)
        array([ 9.83218621,  9.81917886,  9.79324926,  9.780327  ])
        self.grav(45)
        9.8061998770458008

        References
        ----------
        .. [1] IOC, SCOR and IAPSO, 2010: The international thermodynamic equation
           of seawater -  2010: Calculation and use of thermodynamic properties.
           Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
           UNESCO (English), 196 pp.

        .. [2] Moritz (2000) Goedetic reference system 1980. J. Geodesy, 74,
           128-133.

        .. [3] Saunders, P.M., and N.P. Fofonoff (1976) Conversion of pressure to
           depth in the ocean. Deep-Sea Res.,pp. 109 - 111.
        """

        X = np.sin(lat * DEG2RAD)
        sin2 = X ** 2
        gs = 9.780327 * (1.0 + (5.2792e-3 + (2.32e-5 * sin2)) * sin2)
        z = z_from_p(p, lat)
        # z is the height corresponding to p.
        grav = gs * (1 - gamma * z)

        return grav
