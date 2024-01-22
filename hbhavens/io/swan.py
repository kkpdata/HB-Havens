# -*- coding: utf-8 -*-
"""
Created on  : Tue Jul 11 11:34:28 2017
Author      : Johan Ansink
Project     : PR3594.10.00
Description : HB Havens

"""
import os
import re
import shutil
import numpy as np

from hbhavens.core.spectrum import Spectrum2D, jonswap, jonswap_2d

SWAN_TIME_FORMAT = '%Y%m%d.%H%M%S'

class SwanIO:

    """
    Class for SWAN io
    """
    stationary = True
    run = None
    table = None
    variables = []
    units = []

    def _makeFolder(self, folder):
        """
        Makes a folder (and its parents) if not present

        Parameters
        ----------
        folder : string
            Name of the folder
        """
        try:
            os.makedirs(folder, exist_ok=False)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

    def createSWANFolder(self, swanFolder):
        """
        Create the SWAN folder structure

        Parameters
        ----------
        swanFolder : string
            Name of the root swan folder structure
        """
        # Normalize the case of a pathname
        swanFolder = os.path.normpath(swanFolder)

        if os.path.exists(swanFolder):
            shutil.rmtree(swanFolder, ignore_errors=True)

        if os.path.exists(swanFolder): # check if it still exists
            raise OSError('Did not succeed deleting the old SWAN-folder: {}. Close the folder and try again.'.format(swanFolder))

        if not os.path.exists(swanFolder):
            swanSubFolders = ['block', 'bottom', 'error_files', 'inputs', 'inputs/masters', 'logfiles', 'points', 'print_files', 'spectra', 'table','par']
            for swanSubFolder in swanSubFolders:
                swanSubFolder = os.path.join(swanFolder,swanSubFolder)
                self._makeFolder(swanSubFolder)

    def getMaster(self):
        """
        Get SWAN master file text. Requires text to be read with readMaster

        """
        self.text = self.master_text[:]

    def check_master(self, items):
        """
        Get presence of replace keys in master
        """
        for item in items:
            if item not in self.master_text:
                raise KeyError('Key "{}" not present in master text'.format(item))

    def copyFiles(self, swan_folder, master, bathymetry, swanrun):
        """
        Copy master template, bottom file and swanrun to SWAN folder

        Parameters
        ----------
        swan_folder : string
            Name of the SWAN folder structure
        master : string
            Name of the SWAN project bottom file
        bathymetry : string
            Name of the SWAN project bathymetry file
        swanrun : string
            Name of the SWAN run file
        """

        if not os.path.exists(master):
            raise OSError('Master file not found: "{}"'.format(master))

        npath = os.path.join(swan_folder,'inputs', 'masters', 'master.swn')
        shutil.copyfile(master, npath)

        if not os.path.exists(bathymetry):
            raise OSError('Bathymetry file not found: "{}"'.format(bathymetry))

        #TODO: niet bestand hernoemen
        npath = os.path.join(swan_folder, 'bottom', 'bottom.dep')
        shutil.copyfile(bathymetry, npath)

        npath = os.path.join(swan_folder, 'swanrun.bat')
        shutil.copyfile(swanrun, npath)

    def readMaster(self, fpath):
        """
        Read SWAN master file

        Parameters
        ----------
        fpath : string
            filename of the SWAN master file
        """
        self.master = fpath
        with open(fpath, 'r') as fp:
            self.master_text = fp.read()

    def updateMaster(self, old, new):
        """
        Update SWAN master file

        Parameters
        ----------
        old : string
            old substring to be replaced
        new : string
            new substring, which would replace old substring
        """

        # Replace with regex
        self.text = re.sub(old.replace('[', '\[').replace(']', '\]'), new, self.text)

    def writeMaster(self, fpath):
        """
        Update SWAN master lines to a new file

        Parameters
        ----------
        fpath : string
            Filename of the new SWAN master file
        """
        with open(fpath, 'w') as fp:
            fp.write(self.text)

    def writeSupportLocation(self, fpath, supportlocation):
        """
        Write support locations to a file

        Parameters
        ----------
        fpath : string
            path name of the pointlist file
        supportlocation : geopandas.GeoDataFrame
            Name of the support location
        """

        pointlist = os.path.join(fpath, 'pointlist.txt')
        with open(pointlist, "w") as fp:
            fp.write(str(supportlocation['XCoordinate']) + '\t' + str(supportlocation['YCoordinate']) + '\n')

    def writeLocations(self, fpath, locations):
        """
        Write locations to a file

        Parameters
        ----------
        fpath : string
            path name of the pointlist file
        locations : geopandas.GeoDataFrame
            geodataframe with locations. Should contain at least the columns
            "XCoordinate" and "YCoordinate".
        """

        pointlist = os.path.join(fpath, 'pointlist.txt')
        with open(pointlist, "w") as fp:
            if locations is not None:
                for location in locations.itertuples():
                    fp.write(f'{location.geometry.x:.3f}\t{location.geometry.y:.3f}\n')

    def setNoReflection(self):
        """
        Remove reflection parameter with regular expression:
        - line must start with OBSTACLE
        - line must contain REFL with subsequent parameters
        - the part until LINE (longest, greedy) is removed

        """

        # Multi line version    
        # self.text = re.sub('(OBSTACLE.+)*REFL[\s&]*\d*.\d*[\s&]*LINE', r'\1 LINE', string, flags=re.MULTILINE)
        
        # Single line version
        marker = '\t \t  \t   \t'
        self.text = re.sub(marker, '&\n', re.sub('(OBSTACLE.+)REFL.+LINE', r'\1LINE', re.sub('& *\n', marker, self.text)))

    def setNoTransmision(self):
        """
        Set transmision parameter to 0 with regular expression:
        - line must start with OBSTACLE
        - line must contain DAM, TRANSm, TRANS1D, TRANS2D with subsequent parameters
        - the part until REFL or LINE (shortest, non greedy) is replaces with TRANSm 0
        """
        # Multi line version
        # self.text = re.sub('(OBSTACLE\s+)(DAM|TRANSm|TRANS1D|TRANS2D)[\s&]*\d*.\d*[\s&]*(REFL|LINE)', r'\1TRANSm 0 \3', self.text, re.MULTILINE)
        # Single line version
        marker = '\t \t  \t   \t'
        self.text = re.sub(marker, '&\n', re.sub('(OBSTACLE.+)(DAM|TRANSm|TRANS1D|TRANS2D).*?(REFL|LINE)', r'\1TRANSm 0 \3', re.sub('& *\n', marker, self.text)))

    def read_output_spectrum(self, fpath):
        """
        Load 2D spectral series file from SWAN spectral file.
        First the lines are read per swan key, to order the lines per key
        After this, the lines corresponding to specific keys are parsed.

        Parameters
        ----------
        fpath : string
            path name of the file
        """
        
        # Read data from swan file
        with open(fpath, 'r') as f:
            keydata = get_lines_per_key(f.readlines())

        # parse coordinates
        crds = np.array([line[:2] for line in keydata['LOCATIONS'][1:]], dtype=float)

        # parse directions
        if 'CDIR' in keydata.keys():
            # cartesian', 'degrees_true'  # CF convention
            dirs = np.array([line[:1] for line in keydata['CDIR'][1:]], dtype=float).squeeze()
        else:
            # 'nautical', 'degrees_north'
            dirs = np.array([line[:1] for line in keydata['NDIR'][1:]], dtype=float).squeeze()

        # parse directions
        if 'AFREQ' in keydata.keys():
            freqs = np.array([line[:1] for line in keydata['AFREQ'][1:]], dtype=float).squeeze()
        else:
            raise NotImplementedError('Only AFREQ implemented')

        if 'TIME' in keydata.keys():
            raise NotImplementedError('Time encoding is not implemented')
        timecoding = None

        # Number of quantities in table
        number_of_quantities = int(keydata['QUANT'][0][0])
        quantity_names = [keydata['QUANT'][i][0] for i in range(1, number_of_quantities*3+1, 3)]
        quantity_units = [keydata['QUANT'][i][0] for i in range(2, number_of_quantities*3+2, 3)]
        quantity_exception_values = [float(keydata['QUANT'][i][0]) for i in range(3, number_of_quantities*3+3, 3)]

        # Check energy units:
        if not quantity_units[0][0:9] == 'm2/Hz/deg':
            raise ValueError('Wrong energy units. Got {}, expected: m2/Hz/deg'.format(quantity_units[0]))

        # Create empty matrix for energy spectra
        factors = []
        spectra = []
        insert = []
        j = 0
        for i in range(len(crds)):
            key = 'energy_{:03d}'.format(i)
            lines = keydata[key]

            # In the normal case, the lines are filled with the energy spectrum
            if lines:
                factors.append(float(lines[0][0]))
                spectra.append(lines[1:])
                j += 1
            
            # If no lines belong to the energy key, we have a case of zero of nodata. We leave the energy to zero there
            else:
                factors.append(0.0)
                insert.append(j)
                
        # Convert to read spectra to an array. Also insert zeros where no data is found
        if spectra:
            energy = np.insert(np.array([spectra], dtype=float), insert, 0, axis=1)
        
        # In the special case of only no data
        else:
            energy = np.zeros((1, len(crds), len(freqs), len(dirs)))
        
        # Multiply energies by factor, also add mask for no data values
        energy = np.ma.masked_array(energy * np.array(factors)[None, :, None, None], energy == quantity_exception_values[0])

        # Create a 2D spectrum object
        spec2d = Spectrum2D(frequencies=freqs, directions=dirs, energy=energy)
        
        return spec2d, crds


def get_lines_per_key(lines):
    """
    Loop through swan file lines. If a line starts with a alphabet character
    it is a key. All the subsequent lines belong to that key, and are
    collected in a dictionary. Some keys have multiple occurences. These are
    keys that refer to spectrum values. We add those to lists.

    Parameters
    ----------
    lines : list
        list with lines

    Returns
    -------
    keydata : dictionary
        Lines per key
    """
    # Initialize dictionary for lines per key and list for lines
    keydata = {}
    loccount = 0

    key = None
    vals = []
    for line in lines:
        # Check if the first character is alphabetic
        if line[0].isalpha():
            # if so, check if it is an allcaps string
            potential_key = line.split()[0]
            if potential_key == potential_key.upper():
                # New key found, save values except when it is the first key
                if key is not None:
                    keydata[key] = [line.strip().split() for line in vals]
                # Determine key and initialize list with values
                key = potential_key[:]
                vals = []
                if key in ['FACTOR', 'ZERO', 'NODATA']:
                    key = 'energy_{:03d}'.format(loccount)
                    loccount += 1
            else:
                # Add values to list
                vals.append(line)
        else:
            # Add values to list
            vals.append(line)
    # Add to dictionary
    keydata[key] = [line.strip().split() for line in vals]

    return keydata

def read_swan_table(filelocation):
    """
    Leest SWAN-table in

    Parameters
    ----------
    filelocation : str
        Locatie waar het SWAN-bestand staat.
    
    Returns
    -------
    resultaat : pd.DataFrame
        resultaattabel
    """

    # Read text
    with open(filelocation, 'r') as f:
        lines = f.readlines()
        header = [line[1:].strip() for line in lines if line.startswith('%')]
        values = [line.split() for line in lines if not line.startswith('%')]
        
    # Find the header row
    for line in header:
        # Lin with swan version
        if 'SWAN' in line:
            run, table, version = re.findall('Run:(.*?)\s+Table:(.*?)\s+SWAN version:(.*?)$', line)[0]
        # line with variables
        elif 'xp' in line.lower():
            variables = line.split()
            break

    # Return dictionary with values
    resultaat = {key: float(vals[0]) for key, vals in zip(variables, zip(*values))}
    
    return resultaat

def generate_dummy_spectrum(filepath, Hs, richting, Tp, result_locations):
    """
    Generate dummy output spectrum based on given wave parameters and
    a JONSWAP spectrum. This function can be used for testing.
    """
    f = open(filepath, 'w')
    
    # Header
    header = ('SWAN   1                                Swan standard spectral file, version\n'
              '$   Data produced by SWAN version 41.10AB\n'             
              '$   Project: generated from simple method ;  run number: 0001\n')
    f.write(header)
    
    # Locations
    f.write('LOCATIONS\n   {}\n'.format(len(result_locations)))
    for loc in result_locations.itertuples():
        f.write('   {:.4f}   {:.4f}\n'.format(*loc.geometry.coords[0]))
        
    
    theta = np.arange(0, 360, 5)
    Tp_max, Tp_min = 6, 2
    freqs = np.linspace(0.7 * (1. / Tp_max), 3.0 * (1. / Tp_min), 38)
    
    # Frequencies
    f.write('AFREQ\n    {}\n'.format(len(freqs)))
    f.write('\n'.join(['{:10.4f}'.format(fi) for fi in freqs]))
    
    # Directions
    f.write('\nNDIR\n    {}\n'.format(len(theta)))
    f.write('\n'.join(['{:10.4f}'.format(th) for th in theta]))
    
    # Meta
    meta = ('\nQUANT\n'
            '     1                                  number of quantities in table\n'
            'VaDens                                  variance densities in m2/Hz/degr\n'
            'm2/Hz/degr                              unit\n'
            '   -0.9900E+02                          exception value\n')
    f.write(meta)
    
    # Write all spectra
    for i, (h, t, th) in enumerate(zip(Hs, Tp, richting)):
        
        if h == 0.0:
            f.write('\nZERO')

        else:
            E_1d = jonswap(freqs, h, t)

            E_2d = jonswap_2d(
                f=freqs,
                theta=theta,
                S_1d=E_1d,
                Hm0=h,
                Tp=t,
                Theta=th,
                spread=10.0
            )

            # Scale to 9901
            factor = E_2d.max() / 9901
            E_2d = np.round(E_2d / factor).astype(int)

            # Write
            f.write('\nFACTOR\n    {:14.8e}\n'.format(factor))
            f.write('\n'.join([''.join(['{:5d}'.format(e) for e in erow]) for erow in E_2d]))

    f.close()

def generate_swan_table(filepath, xs, ys, Hs, Tp, wlevs):
    """
    Generate dummy output table based on given wave parameters.
    This function can be used for testing.
    """
    
    f = open(filepath, 'w')
    # Write header
    f.write('%\n%\n% Run:0001    Table:plist           SWAN version:41.10A\n%\n')
    f.write('%{:>11s}{:>13s}{:>13s}{:>13s}{:>13s}{:>13s}{:>13s}\n'.format('Xp', 'Yp', 'Depth', 'Watlev', 'Botlev', 'Hsig', 'TPsmoo'))
    f.write('%{:>11s}{:>13s}{:>13s}{:>13s}{:>13s}{:>13s}{:>13s}\n%\n'.format('[m]', '[m]', '[m]', '[m]', '[m]', '[m]', '[s]'))

    
    # Iterate through arrays/lists
    for x, y, wl, hs, tp in zip(xs, ys, wlevs, Hs, Tp):
        f.write('{:12.3f} {:12.3f} {:12.5f} {:12.5f} {:12.5f} {:12.5f} {:12.5f}\n'.format(x, y, wl, wl, 0, hs, tp))
        
    f.close()
    