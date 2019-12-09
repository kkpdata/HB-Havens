# -*- coding: utf-8 -*-
"""
Created on  : Tue Jul 11 11:34:36 2017
Author      : Guus Rongen
Project     : PR3594.10.00
Description :

"""
import os
import shutil
import errno
import json
import itertools

import numpy as np

class PharosIO:

    """
    Class for Pharos io
    """
    def __init__(self, parent):
        self.pharos = parent
        self.settings = parent.mainmodel.project.settings

    def create_pharos_folder(self, src, dest):
        """
        Create the Pharos folder structure

        Parameters
        ----------
        src : string
            Name of the root pharos folder structure
        dest : string
            Name of the new pharos folder structure
        """
        # Normalize the case of a pathname
        src = os.path.normpath(src)
        dest = os.path.normpath(dest)

        # Remove old folder
        if os.path.exists(dest):
            shutil.rmtree(dest, ignore_errors=True)

        # Check if it still exists
        if os.path.exists(dest):
            raise OSError('Did not succeed deleting the old PHAROS-folder: {}. Close the folder and try again.'.format(dest))

        if not os.path.exists(dest):
            # Copy PHAROS folder
            shutil.copytree(src, dest)

    def getSchematisations(self, schematisationsfolder):
        """
        Get a list of the Pharos schematisations

        Parameters
        ----------
        schematisationsfolder : string
            Name of the pharos folder with all schematisations
        """
        schematisations = [name for name in os.listdir(schematisationsfolder) if os.path.isdir(os.path.join(schematisationsfolder, name))]
        return schematisations

    def saveData(self, fname, data):
        if os.path.exists(fname):
            os.remove(fname)

        with open(fname, 'w') as outfile:
            json.dump(data, outfile, indent=2)

    def loadData(self,fname):
        if os.path.exists(fname):
            with open(fname, 'r') as data:
                return json.load(data)
        else:
            return None


    def write_bat_files(self, schematisation):
        
        # Get iterables
        iterables = self.pharos.get_iterables()
        frequencies = iterables[0]
        water_levels = iterables[1]        
        directions = self.settings['pharos']['schematisations'][schematisation]
        combinations = set(itertools.product(frequencies, directions, water_levels))
        ncombs = len(combinations)

        # Destination folder
        dst_folder = os.path.join(self.settings['pharos']['paths']['pharos folder'], 'calculations', schematisation)

        # Open pharcone bat
        f_pharcone = open(os.path.join(dst_folder, 'launch_pharcone.bat'), 'w')
        f_pharcone.write('echo off\n')

        # Open pharos bat
        f_pharos = open(os.path.join(dst_folder, 'launch_pharos.bat'), 'w')
        f_pharos.write('echo off\n')

        # Open rDPRA bat
        f_rDPRA = open(os.path.join(dst_folder, 'launch_rDPRA.bat'), 'w')
        f_rDPRA.write('echo off\n')

        for i, (frequency, direction, water_level) in enumerate(combinations):
            run = 'T{:.3f}D{:05.1f}H{:.3f}'.format(1./frequency, direction, water_level)
            project = self.pharos.project_names[schematisation]
            grid = self.pharos.grid_names[schematisation]
            outfile = '{}_pharcon_{}_{}.out'.format(project, grid, run)

            # Write lines to pharcone
            f_pharcone.write('echo {} our of {}\n'.format(i+1, ncombs))
            f_pharcone.write('if not exist {} (\npharcone.exe {} {} {} )\n'.format(outfile, project, grid, run))

            # Write lines to pharos
            f_pharos.write('echo Run: {} started\n'.format(run))
            f_pharos.write('echo {} our of {}\n'.format(i+1, ncombs))
            f_pharos.write('if not exist {} (\n[PHAROSEXE] {} {} {} )\n'.format(outfile, project, grid, run))

            # Write lines to rDPRA
            f_rDPRA.write('rDPRA_batch.exe input_{}.txt\n'.format(run))

        # Close files
        f_pharcone.close()
        f_pharos.close()
        f_rDPRA.close()

    def write_location_file(self, schematisation):
        """
        Write file with PHAROS locations to schematisation folder
        The locations are the HB Havens location, transformed with
        the user specified offset.

        Parameters
        ----------
        schematisation : str
            Name of the schematisation
        """
        # Destination folder
        dst_folder = os.path.join(self.settings['pharos']['paths']['pharos folder'], 'calculations', schematisation)

        # Get transformation
        dx = self.settings['pharos']['transformation']['dx']
        dy = self.settings['pharos']['transformation']['dy']

        # Open file and write
        with open(os.path.join(dst_folder, 'coordinates.csv'), 'w') as f:
            for pt in self.pharos.mainmodel.schematisation.result_locations['geometry']:
                xcrd, ycrd = round(pt.coords[0][0] + dx, 3), round(pt.coords[0][1] + dy, 3)
                # Convert to formatted string, max 1 decimal
                xcrd = '{:.0f}'.format(xcrd) if xcrd.is_integer() else '{:.1f}'.format(xcrd)
                ycrd = '{:.0f}'.format(ycrd) if ycrd.is_integer() else '{:.1f}'.format(ycrd)
                # Write to file
                f.write(xcrd+','+ycrd+'\n')

def generate_dummy_json(wave_direction, locations, normalen, f_in=0.6, f_out=0.2):
    """
    Function to generate a dummy input file for PHAROS
    """

    # Bpeaal H
    inkomend = np.maximum(np.random.rand(len(locations)) * (f_in + 0.1) - 0.1, 0)
    reflecterend = np.maximum(np.random.rand(len(locations)) * (f_out + 0.1) - 0.1, 0)

    # Bereken uitvalhoek
    uitval_hoek = (normalen + (normalen - wave_direction)) % 360

    # Genereer json
    output = []
    for i, location in enumerate(locations):
        loc_data = {}
        x, y = round(location[0], 1), round(location[1], 1)
        loc_data['input_coordinates'] = {'x': int(x) if x.is_integer() else x, 'y': int(y) if y.is_integer() else y}
        loc_data['wave_heights'] = [
            {
                'naut': int((wave_direction + np.random.randn(1)[0]*10)) % 360,
                'H': round(inkomend[i], 3)
            },
            {
                'naut': int((uitval_hoek[i] + np.random.randn(1)[0]*10)) % 360,
                'H': round(reflecterend[i], 3)
            }
        ]
        output.append(loc_data)
        
    return output
