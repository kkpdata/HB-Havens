# -*- coding: utf-8 -*-
"""
Created on  : Wed Aug  2 11:13:42 2017
Author      : Guus Rongen, HKV Lijn in Water
Project     : PR3594.10.00
Description : 
    
"""

from pycallgraph import PyCallGraph
from pycallgraph import Config
from pycallgraph.output import GraphvizOutput

import os

if not 'Graphviz2.38' in os.environ['PATH']:
    os.environ['PATH'] = os.environ['PATH']+r';C:\Program Files (x86)\Graphviz2.38\bin'

config = Config(max_depth=40)
graphviz = GraphvizOutput(output_file='structure_withExecfile.png')

import sys
sys.path.append(r'..')

#    import os
#from hbhavens import simple
#from hbhavens import advanced
from hbhavens import geometry
import matplotlib.pyplot as plt
#    import sqlite3

from hbhavens import io
import sqlite3

#import pandas as pd
import imp
imp.reload(geometry)
imp.reload(io)
#from hbhavens import sqlite

with PyCallGraph(output=graphviz, config=config):
        
        

    harbor = geometry.Harbor()
    
    # Import breakwaters
    shapefileloc = r'd:\Documents\3594.10 Software tool voor (zee)havens\GIS\vlissingen'
    
    harborarealoc = os.path.join(shapefileloc, 'haventerrein.shp')
    harborarea = harbor.add_geometry('harborarea', harborarealoc)
    
    breakwaterloc = os.path.join(shapefileloc, 'havendammen.shp')
    breakwaters = harbor.add_geometry('breakwater', breakwaterloc)
    
    # Import levees
    flooddefenceloc = os.path.join(shapefileloc, 'waterkeringlijn.shp')
    flooddefence = harbor.add_geometry('flooddefence', flooddefenceloc)
    
    
    # Add database
    database = os.path.join(shapefileloc, '..', 'WBI2017_Westerschelde_29-3_29-4_v01.sqlite')
    conn = sqlite3.connect(database)
    
    # Get locations
    locations = io.database.read_HRD_locations(conn)
    
    supportlocationname = 'WS_1_29-3_dk_00024'
    supportlocation = locations.where(locations['Name'] == supportlocationname).dropna()
    harbor.add_support_location(supportlocation)
    
    #HRDlocations = harbor.add_HRDlocations(os.path.join(shapefileloc, 'uitvoerlocaties.shp'))
    
    
    harbor.generate_HRDLocations(distance=50, interval=100)
    
    plt.close('all')
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    harbor.plot(ax=ax)
    #locations.plot(ax=ax, color='k')
    #for _, row in locations.iterrows():
    #    ax.text(row['geometry'].x, row['geometry'].y, row['Name'])
    
    harbor.HRDLocations.plot(ax=ax, color='k', ls='')
    harbor.support_location.plot(ax=ax, color='k', marker='x', markersize=5, markeredgewidth=2)
    
    ax.plot(*zip(*geometry.extend_point_to_linestring(harbor.breakwater.iloc[0]['geometry'].coords[0], 235, -1000).coords[:]))
    #ax.plot(*zip(*geometry.extend_point_to_linestring(harbor.breakwater.iloc[0]['geometry'].coords[0], 215, -1000).coords[:]))
    #ax.plot(*zip(*geometry.extend_point_to_linestring(harbor.breakwater.iloc[0]['geometry'].coords[0], 190, -1000).coords[:]))
    
    #ax.legend()
    
    #url = 'http://geodata.nationaalgeoregister.nl/tiles/service'
    #
    #wmts = WMTS(url, 'EPSG:28992')
    #bbox = (ax.get_xlim()[0], ax.get_ylim()[0], ax.get_xlim()[1], ax.get_ylim()[1])
    #layer = 'brtachtergrondkaartgrijs'
    ##layer = '2016_ortho25'
    #wmts.plot(bbox, layer=layer, level=10, ax=ax, clip=True, fmt='image/png')
        
    # Read conditions
    #
    hydraulic_loads = io.database.read_HydroDynamicData(conn, supportlocation.index[0])
    #hydraulic_loads = pd.read_excel(os.path.join(shapefileloc, 'HydraulicLoads.xlsx'))
    #
    #
    #st = timeit.default_timer()
    
    # Calculate diffraction coefficient
    Kd = simple.calc_diffraction(breakwater=harbor.breakwater,
                                 hrdlocations=harbor.HRDLocations,
                                 bedlevel = -20.0,
                                 hydraulic_loads = hydraulic_loads,
                                 )
    
    #print(timeit.default_timer() - st)
    conn.close()