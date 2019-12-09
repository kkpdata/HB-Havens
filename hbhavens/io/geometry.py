# -*- coding: utf-8 -*-
"""
Created on  : Wed Sep 27 15:43:09 2017
Author      : Guus Rongen, HKV Lijn in Water
Project     : PR0000.00
Description : 
    
"""

import os
import fiona
import numpy as np
from shapely.geometry import LineString, shape

DATADIR = os.path.join(os.path.dirname(__file__), '..', 'data')

def get_datadir(datadir):
    if datadir is None:
        datadir = DATADIR
    return datadir

def import_section_ids(datadir=None):
    """
    Load traject ids from appended shapefile
    """
    with fiona.open(os.path.join(get_datadir(datadir), 'dijktrajecten.shp')) as src:
        sectionids = [entry['properties']['traject_id'] for entry in src]
            
    return sectionids


def import_section_geometry(sectionid, datadir=None):
    """
    Load traject ids from appended shapefile
    
    Parameters
    ----------
    sectionid : str
        section id for which the section geometry is loaded
    """
    
    with fiona.open(os.path.join(get_datadir(datadir), 'dijktrajecten.shp')) as src:
        for entry in src:
            if entry['properties']['traject_id'] == sectionid:
                geometry = LineString([crd[:2] for crd in entry['geometry']['coordinates']])
                return geometry
        else:
            return None

def read_landboundary(datadir=None):
    """
    Load land boundary for background plotting
    """
    with open(os.path.join(get_datadir(datadir), 'landboundary.txt'),'r') as src:
        return list(zip(*(map(float, line.split()) for line in src.readlines())))