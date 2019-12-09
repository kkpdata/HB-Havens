# -*- coding: utf-8 -*-
"""
Created on  : Mon Jul 10 14:52:43 2017
Author      : Guus Rongen
Project     : PR0000.00
Description :

"""

import sys
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
if '..' not in sys.path:
    sys.path.append('..')

import geopandas as gpd
import matplotlib.pyplot as plt
import sqlite3
import shutil
import pandas as pd
from descartes import PolygonPatch

import timeit

pd.options.mode.chained_assignment = None

from hbhavens import core
from hbhavens import io
import pandas as pd
import numpy as np

from matplotlib.collections import PatchCollection

#import pandas as pd
import imp

#from hbhavens import sqlite


#==============================================================================
# import harbor geometry
#==============================================================================

mainmodel = core.models.MainModel()

# Import levees
flooddefence = mainmodel.schematisation.add_flooddefence('28-1')

harborarealoc = os.path.join('colijnsplaat', 'haventerrein.shp')
harborarea = mainmodel.schematisation.add_harborarea(harborarealoc)

breakwaterloc = os.path.join('colijnsplaat', 'havendammen.shp')
breakwaters = mainmodel.schematisation.add_breakwater(breakwaterloc)





#diff = chmin.exterior.difference(harbor.geometries.harborarea.iloc[0]['geometry'])
mainmodel.schematisation.set_bedlevel(-10.0)

# Add database
database = os.path.join('colijnsplaat', 'WBI2023_Oosterschelde_28-1_v00_selectie.sqlite')
dstdb = os.path.join('colijnsplaat', 'WBI2023_Oosterschelde_28-1_v00_tevullen.sqlite')
shutil.copy2(database, dstdb)
# Get locations
mainmodel.input_databases.add_HRD(database)

#supportlocation = locations.where(locations['Name'] == supportlocationname).dropna()
#harbor.add_support_location(supportlocation)


mainmodel.schematisation.add_result_locations(os.path.join('colijnsplaat', 'uitvoerlocaties.shp'))



plt.close('all')
fig, ax = plt.subplots()
ax.set_aspect('equal')

polygons = []
for i, geometry in enumerate(mainmodel.schematisation.harborarea['geometry'].values.tolist()):
    patch = PolygonPatch(geometry)
    polygons.append(patch)

collection = ax.add_collection(PatchCollection(polygons, label='test'))
#harbor.geometries.harborarea.plot(ax=ax, color='C0')
mainmodel.schematisation.flooddefence.plot(ax=ax, color='C3', label='fd')
mainmodel.schematisation.breakwaters.plot(ax=ax, color='C1')

mainmodel.schematisation.generate_result_locations(50, 100, 20)

#print(b)
ax.plot(*np.vstack(mainmodel.schematisation.entrance.coords[:]).T, ls='--')
#locations.plot(ax=ax, color='k')
#for _, row in locations.iterrows():
#    ax.text(row['geometry'].x, row['geometry'].y, row['Name'])

ax.add_patch(PolygonPatch(mainmodel.schematisation.inner, alpha=0.3, color='grey'))
mainmodel.schematisation.support_locations.plot(ax=ax, marker='x', mew=8, ms=1, color='k')

ax.legend()

#harbor.plot(ax=ax)
#locations.plot(ax=ax, color='k')
#for _, row in locations.iterrows():
#    ax.text(row['geometry'].x, row['geometry'].y, row['Name'])

#ax.add_patch(PolygonPatch(harbor.inner, alpha=0.3, color='grey'))
#harbor.plot_support_locations(ax=ax)

supportlocationname = 'OS_1_28-1_dk00102'
mainmodel.schematisation.set_selected_support_location(supportlocationname)

#harbor.plot_result_locations(ax=ax)
#ax.plot(*harbor.support_locations.reset_index().set_index('Name').loc[supportlocationname, 'geometry'].coords[0], marker='o', ms=10, mew=1, mfc='none', color='k')

#ax.plot(*zip(*geometry.extend_point_to_linestring(harbor.breakwater.iloc[0]['geometry'].coords[0], 205, -1000).coords[:]))
#ax.plot(*zip(*geometry.extend_point_to_linestring(harbor.breakwater.iloc[0]['geometry'].coords[-1], 175, -1000).coords[:]))
#ax.plot(*zip(*geometry.extend_point_to_linestring(harbor.breakwater.iloc[0]['geometry'].coords[0], 190, -1000).coords[:]))

#ax.legend()

# Read conditions
#
#hydraulic_loads = pd.read_excel(os.path.join(shapefileloc, 'HydraulicLoads.xlsx'))#.iloc[[2]]

mainmodel.project.settings['calculation_method']['method'] = 'simple'
#def show_progess(percentage, message):
#    print(percentage, message)

mainmodel.simple_calculation.run_all()

mainmodel.modeluncertainties.add_result_locations()
tab = mainmodel.modeluncertainties.table
tab.loc[:, tab.columns[1:]] = ['Steunpunt'] + mainmodel.modeluncertainties.supportloc_unc.stack().values.tolist()

mainmodel.export.add_result_locations()
mainmodel.project.settings['export']['export_HLCD_and_config'] = False
mainmodel.export.export_dataframe['Exportnaam'] = mainmodel.export.export_dataframe['Naam']
mainmodel.export.export_dataframe['SQLite-database'] = dstdb

mainmodel.export.export_output_to_database()
