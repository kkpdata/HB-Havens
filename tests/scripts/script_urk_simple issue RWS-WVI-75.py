# -*- coding: utf-8 -*-
"""
Created on  : Mon Jul 10 14:52:43 2017
Author      : Guus Rongen
Project     : PR0000.00
Description :

"""

import sys
import os
import shutil

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from common import set_paths

case = 'urk'
trunkdir, tempdir, datadir = set_paths(__file__, case)
from hbhavens.core.models import MainModel
from hbhavens.io.database import HRDio

#==============================================================================
# import harbor geometry
#==============================================================================

harbor = MainModel()
harbor.project.settings['project']['name'] = case
harbor.project.settings['project']['user']['email'] = 'benit@hkv.nl'
harbor.project.settings['project']['user']['name'] = 'Matthijs Bénit'

harbor.project.save_as(os.path.join(tempdir, case, case) + '.json')

# Import levees
trajecten = ['7-2']
for trajectid in trajecten:
    harbor.schematisation.add_flooddefence(trajectid)

# Harbourarea
harborarealoc = os.path.join(datadir, 'shapes', 'haventerrein.shp')
harbor.schematisation.add_harborarea(harborarealoc)

# Breakwaters
breakwaterloc = os.path.join(datadir, 'shapes', 'havendammen.shp')
harbor.schematisation.add_breakwater(breakwaterloc)

# Harbour boundary
harbor.schematisation.generate_harbor_bound()

# Output locations
harbor.schematisation.add_result_locations(os.path.join(datadir, 'shapes', 'uitvoerlocaties.shp'))

# Bed level
bedlevel = -5.0
harbor.schematisation.set_bedlevel(bedlevel)
harbor.schematisation.check_bedlevel(bedlevel)

# Add database
database = os.path.join(datadir, 'databases', 'WBI2017_IJsselmeer_7-2_v02.sqlite')
# database = os.path.join(datadir, 'databases', 'WBI2017_IJsselmeer_7-2_v02_steunpunt.sqlite')
harbor.hydraulic_loads.add_HRD(database)
supportlocation = 'YM_3_7-2_ha02_00001'
harbor.schematisation.set_selected_support_location(supportlocation)

#==============================================================================
# run simple calculation
#==============================================================================
harbor.project.settings['calculation_method']['method'] = 'simple'
harbor.simple_calculation.initialize()
harbor.simple_calculation.diffraction.run()
harbor.simple_calculation.transmission.run()
harbor.simple_calculation.wavegrowth.run()
harbor.simple_calculation.wavebreaking.run()
harbor.simple_calculation.combinedresults.run(processes=['Diffractie', 'Transmissie', 'Lokale golfgroei'])

harbor.project.settings['simple']['finished'] = True
harbor.simple_calculation.save_results()

#==============================================================================
# Uncertainties
#==============================================================================
harbor.modeluncertainties.add_result_locations()
harbor.modeluncertainties.load_modeluncertainties()

uncertainties = harbor.modeluncertainties.supportloc_unc.unstack()
uncertainties.index = [' '.join(vals) for vals in uncertainties.index]
harbor.modeluncertainties.table.loc[:, uncertainties.index] = uncertainties.values
harbor.modeluncertainties.save_tables()

#==============================================================================
# Export
#==============================================================================
harbor.project.settings["export"]["export_HLCD_and_config"] = True
harbor.project.settings["export"]["HLCD"] = os.path.join(tempdir, case, case, 'hlcd.sqlite')
shutil.copy2(os.path.join(datadir, 'databases', 'WBI2017_IJsselmeer_7-2_v02.sqlite'), os.path.join(tempdir, case, case, 'WBI2017_IJsselmeer_7-2_v02_withoriginal.sqlite'))
shutil.copy2(os.path.join(datadir, 'databases', 'WBI2017_IJsselmeer_7-2_v02.sqlite'), os.path.join(tempdir, case, case, 'WBI2017_IJsselmeer_7-2_v02_nooriginal.sqlite'))
shutil.copy2(os.path.join(datadir, 'databases', 'WBI2017_IJsselmeer_7-2_v02.config.sqlite'), os.path.join(tempdir, case, case, 'WBI2017_IJsselmeer_7-2_v02_withoriginal.config.sqlite'))
shutil.copy2(os.path.join(datadir, 'databases', 'WBI2017_IJsselmeer_7-2_v02.config.sqlite'), os.path.join(tempdir, case, case, 'WBI2017_IJsselmeer_7-2_v02_nooriginal.config.sqlite'))
shutil.copy2(os.path.join(datadir, '..', 'hlcd.sqlite'), os.path.join(tempdir, case, case, 'hlcd.sqlite'))

# Default method including the original database locations
harbor.export.add_result_locations()
harbor.export.export_dataframe['Exportnaam'] = case + '_' + harbor.export.export_dataframe['Naam'] # Hier is de mogelijkheid om de namen aan de juiste conventie te laten voldoen. In de GUI gaat dit via een excel file
harbor.project.settings['export']['HRD'] = os.path.join(tempdir, case, case, 'WBI2017_IJsselmeer_7-2_v02_withoriginal.sqlite')
harbor.export.settings['export']['remove_original_HRD_loc'] = False
harbor.export.export_output_to_database()

HRD_withoriginal = []
# for database in harbor.export.export_dataframe['SQLite-database'].unique():
hrd_connection = HRDio(harbor.project.settings['export']['HRD'])

# Get locations
hrd_connection._connect()
remove_ids = []
sql = f'SELECT HRDLocationId, Name, XCoordinate, YCoordinate FROM HRDLocations;'
locations = hrd_connection.conn.execute(sql)
[HRD_withoriginal.append(loc) for loc in locations]
hrd_connection._close()

# Alternate method excluding the original database locations
harbor.export.add_result_locations()
harbor.export.export_dataframe['Exportnaam'] = case + '_' + harbor.export.export_dataframe['Naam'] # Hier is de mogelijkheid om de namen aan de juiste conventie te laten voldoen. In de GUI gaat dit via een excel file
harbor.project.settings['export']['HRD'] = os.path.join(tempdir, case, case, 'WBI2017_IJsselmeer_7-2_v02_nooriginal.sqlite')
harbor.export.settings['export']['remove_original_HRD_loc'] = True
harbor.export.export_output_to_database()

HRD_nooriginal = []
# for database in harbor.export.export_dataframe['SQLite-database'].unique():
hrd_connection = HRDio(harbor.project.settings['export']['HRD'])

# Get locations
hrd_connection._connect()
remove_ids = []
sql = f'SELECT HRDLocationId, Name, XCoordinate, YCoordinate FROM HRDLocations;'
locations = hrd_connection.conn.execute(sql)
[HRD_nooriginal.append(loc) for loc in locations]
hrd_connection._close()
    
harbor.project.settings['export']['export_succeeded'] = True

harbor.save_tables()
harbor.project.save()

#==============================================================================
# plot schematisation
#==============================================================================

fig, ax = harbor.schematisation.plot_schematisation(buffer=500, locations=False)

[ax.plot(HRD[2], HRD[3], markerfacecolor='None', markeredgecolor='b', marker='o', linestyle='None', markersize=10, zorder=3, label='HRD + steunpunt + uitvoer') for HRD in HRD_withoriginal[0:1]]
[ax.plot(HRD[2], HRD[3], markerfacecolor='None', markeredgecolor='b', marker='o', linestyle='None', markersize=10, zorder=3) for HRD in HRD_withoriginal]
[ax.plot(HRD[2], HRD[3], markerfacecolor='None', markeredgecolor='r', marker='x', linestyle='None', markersize=10, zorder=3, label='steunpunt + uitvoer') for HRD in HRD_withoriginal[0:1]]
[ax.plot(HRD[2], HRD[3], markerfacecolor='None', markeredgecolor='r', marker='x', linestyle='None', markersize=10, zorder=3) for HRD in HRD_nooriginal]
ax.legend(loc='lower right')

fig.savefig(os.path.join(tempdir, case, case, case+'.png'), dpi=150)