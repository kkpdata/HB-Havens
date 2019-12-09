# -*- coding: utf-8 -*-
"""
Created on  : Mon Jul 10 14:52:43 2017
Author      : Guus Rongen
Project     : PR0000.00
Description :

"""

import sys
sys.path.append(r'..')

import os
from hbhavens import core
import geopandas as gpd
import matplotlib.pyplot as plt
import sqlite3
import pandas as pd
from descartes import PolygonPatch

import timeit

pd.options.mode.chained_assignment = None

from hbhavens import io
import pandas as pd

#import pandas as pd
import imp
imp.reload(core)
imp.reload(io)

#from hbhavens import sqlite


#==============================================================================
# import harbor geometry
#==============================================================================

harbor = core.general.Harbor()


harborarealoc = os.path.join('vlissingen', 'haventerrein.shp')
harborarea = harbor.add_harborarea(harborarealoc)

breakwaterloc = os.path.join('vlissingen', 'havendammen.shp')
breakwaters = harbor.add_breakwater(breakwaterloc)

# Import levees
flooddefenceloc = os.path.join('vlissingen', 'waterkeringlijn.shp')
flooddefence = harbor.add_flooddefence(flooddefenceloc)

bedlevel = -20.0

# Add database
database = os.path.join('vlissingen', 'WBI2017_Westerschelde_29-3_29-4_v01.sqlite')
# Get locations
harbor.add_HRD(database)

#supportlocation = locations.where(locations['Name'] == supportlocationname).dropna()
#harbor.add_support_location(supportlocation)


harbor.add_result_locations(os.path.join('vlissingen', 'uitvoerlocaties.shp'))


#harbor.generate_result_locations(distance=50, interval=100)



plt.close('all')
fig, ax = plt.subplots()
ax.set_aspect('equal')
harbor.plot(ax=ax)
#locations.plot(ax=ax, color='k')
#for _, row in locations.iterrows():
#    ax.text(row['geometry'].x, row['geometry'].y, row['Name'])

ax.add_patch(PolygonPatch(harbor.inner, alpha=0.3, color='grey'))
harbor.plot_support_locations(ax=ax)

supportlocationname = 'WS_1_29-3_dk_00024'
harbor.add_location_links({resultloc : supportlocationname for resultloc in harbor.result_locations['Name']})

harbor.plot_result_locations(ax=ax)
ax.plot(*harbor.support_locations.reset_index().set_index('Name').loc[supportlocationname, 'geometry'].coords[0], marker='o', ms=10, mew=1, mfc='none', color='k')

#ax.plot(*zip(*geometry.extend_point_to_linestring(harbor.breakwater.iloc[0]['geometry'].coords[0], 205, -1000).coords[:]))
#ax.plot(*zip(*geometry.extend_point_to_linestring(harbor.breakwater.iloc[0]['geometry'].coords[-1], 175, -1000).coords[:]))
#ax.plot(*zip(*geometry.extend_point_to_linestring(harbor.breakwater.iloc[0]['geometry'].coords[0], 190, -1000).coords[:]))

#ax.legend()

# Read conditions
#
#hydraulic_loads = pd.read_excel(os.path.join(shapefileloc, 'HydraulicLoads.xlsx'))#.iloc[[2]]
hydraulic_loads = harbor.hydraulic_loads
#.

st1 = timeit.default_timer()

# Calculate diffraction coefficient
diffraction = core.simple.Diffraction(breakwaters=harbor.breakwaters,
                                      hrdlocations=harbor.result_locations,
                                      bedlevel = bedlevel,
                                      hydraulic_loads = hydraulic_loads,
                                      )

diffraction.run()
print('Diffraction took: {:5.2f}'.format(timeit.default_timer() - st1))
st = timeit.default_timer()

transmission = core.simple.Transmission(breakwaters=harbor.breakwaters, 
                                        hrdlocations=harbor.result_locations, 
                                        hydraulic_loads=hydraulic_loads,
                                        )
transmission.run()

print('Transmission took: {:5.2f}'.format(timeit.default_timer() - st))
st = timeit.default_timer()

fetch = core.simple.LocalWaveGrowth(
        harbor.result_locations,
        harbor.breakwaters,
        harbor.flooddefence, 
        harbor.innerbound,
        hydraulic_loads)
fetch.run()

print('Local wave growth took: {:5.2f}'.format(timeit.default_timer() - st))
st = timeit.default_timer()

wavebreaking = core.simple.WaveBreaking(hrdlocations=harbor.result_locations,
                                        harborarea=harbor.harborarea,
                                        hydraulic_loads=hydraulic_loads,
                                        bedlevel=bedlevel,
                                        breakfraction=0.7,
                                        limitlength=50.0,
                                        )

wavebreaking.run()

print('Wave breaking took: {:5.2f}'.format(timeit.default_timer() - st))
st = timeit.default_timer()

combining_results = core.simple.CombineResults(
                                      hydraulic_loads=hydraulic_loads,
                                      )

combining_results.combine_output(diffraction=diffraction.output,
                                 transmission=transmission.output,
                                 wavegrowth=fetch.output,
                                 wavebreaking=wavebreaking.output,
                                      )
print('Combining took: {:5.2f}'.format(timeit.default_timer() - st))
st = timeit.default_timer()

print('Total time: {:5.2f}'.format(timeit.default_timer() - st1))

#print(combined.output)


#ax.plot(*zip(*fetch.inner.coords[:]))
#ax.plot(*zip(*fetch.inner[1].coords[:]), color='k', ls='--')

#tz.plot(ax=ax, color='grey', alpha=0.2)

#conn.close()

#==============================================================================
# simple method
#==============================================================================

#    simple.process_diffraction(harbor)
#    simple.process_transmission(harbor)
#    simple.process_diffraction_transmission(harbor)
#    simple.process_wavegrowth(harbor)
#    simple.process_foreshore(harbor)


"""
#==============================================================================
# advanced method
#==============================================================================
elif method == 'complex':
    # Choose SWAN or PHAROS
    model = advanced.choose_model()
    if model.type == 'SWAN':
        advanced.swan.load_master(model)        
        advanced.swan.select_conditions(model, loads)
        advanced.swan.iterate_initial_conditions(model, loads)
        
        io.swan.generate_input(model, loads)
        io.swan.read_output(model, loads)
        
    elif model.type == 'PHAROS':
        advanced.pharos.load_master(model)        
        advanced.pharos.select_conditions(model, loads)
        advanced.pharos.iterate_initial_conditions(model, loads)
        
        io.pharos.generate_input(model, loads)
        io.pharos.read_output(model, loads)
    
    else:
        raise ValueError('Model type "{}" is not implemented'.format(model.type))

else:
    raise ValueError('Method "{}" not implemented'.format(method))
    
#==============================================================================
# Add to database
#==============================================================================
sqlite.export_hrd(loads, harbor)
sqlite.export_hlcd(harbor)
sqlite.export_config(harbor)


"""