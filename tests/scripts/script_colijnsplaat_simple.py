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

case = 'colijnsplaat'
trunkdir, tempdir, datadir = set_paths(__file__, case)
from hbhavens.core.models import MainModel

#==============================================================================
# import harbor geometry
#==============================================================================

harbor = MainModel()
harbor.project.settings['project']['name'] = case
harbor.project.settings['project']['user']['email'] = 'benit@hkv.nl'
harbor.project.settings['project']['user']['name'] = 'Matthijs Bénit'

harbor.project.save_as(os.path.join(tempdir, case, case) + '.json')

# Import levees
trajecten = ['28-1']
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
bedlevel = -10.0
harbor.schematisation.set_bedlevel(bedlevel)
harbor.schematisation.check_bedlevel(bedlevel)

# Add database
database = os.path.join(datadir, 'databases', 'WBI2023_Oosterschelde_28-1_v00_selectie.sqlite')
harbor.hydraulic_loads.add_HRD(database)
supportlocation = 'OS_1_28-1_dk00102'
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
harbor.project.settings['export']["export_HLCD_and_config"] = False
harbor.project.settings['export']['HRD'] = os.path.join(tempdir, case, case, 'WBI2023_Oosterschelde_28-1_v00_selectie.sqlite')
shutil.copy2(os.path.join(datadir, 'databases', 'WBI2023_Oosterschelde_28-1_v00_selectie.sqlite'), os.path.join(tempdir, case, case))

harbor.export.add_result_locations()
harbor.export.export_dataframe['Exportnaam'] = case + '_' + harbor.export.export_dataframe['Naam'] # Hier is de mogelijkheid om de namen aan de juiste conventie te laten voldoen. In de GUI gaat dit via een excel file
harbor.export.export_output_to_database()
harbor.project.settings['export']['export_succeeded'] = True

harbor.save_tables()
harbor.project.save()

#==============================================================================
# plot schematisation
#==============================================================================
harbor.schematisation.plot_schematisation(buffer=10, save=True, path=os.path.join(tempdir, case), close=True)