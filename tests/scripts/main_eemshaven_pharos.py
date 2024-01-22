# -*- coding: utf-8 -*-
"""
Created on  : Mon Jul 10 14:52:43 2017
Author      : Matthijs Benit
Project     : PR4982.10
Description :

"""

import sys
import os
import shutil

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import common
import zipfile

trunkdir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(trunkdir)
from hbhavens.core.models import MainModel

#==============================================================================
# import harbor geometry
#==============================================================================
tempdir = os.path.join(os.path.dirname(trunkdir),"temp")
if not os.path.exists(tempdir):
    os.mkdir(tempdir)

case = 'eemshaven'
if os.path.isdir(os.path.join(tempdir, case)):
    shutil.rmtree(os.path.join(tempdir, case), ignore_errors=True)

datadir = os.path.join(trunkdir, 'tests', 'data', case)

harbor = MainModel()
harbor.project.settings['project']['name'] = case
harbor.project.settings['project']['user']['email'] = 'benit@hkv.nl'
harbor.project.settings['project']['user']['name'] = 'Matthijs Bénit'

harbor.project.save_as(os.path.join(tempdir, case) + '.json')

# Import levees
trajecten = ['6-6']
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
dbname = 'WBI2017_Waddenzee_Oost_6-6_v03_selectie'
database = os.path.join(datadir, 'databases', dbname + '.sqlite')
harbor.hydraulic_loads.add_HRD(database)
supportlocation = 'WZ_1_6-6_dk_00178'
harbor.schematisation.set_selected_support_location(supportlocation)

# #==============================================================================
# # run simple calculation
# #==============================================================================
# harbor.project.settings['calculation_method']['method'] = 'simple'
# harbor.simple_calculation.initialize()
# harbor.simple_calculation.diffraction.run()
# harbor.simple_calculation.transmission.run()
# harbor.simple_calculation.wavegrowth.run()
# harbor.simple_calculation.wavebreaking.run()
# harbor.simple_calculation.combinedresults.run(processes=['Diffractie', 'Transmissie', 'Lokale golfgroei'])

# harbor.project.settings['simple']['finished'] = True
# harbor.simple_calculation.save_results()

#==============================================================================
# run advanced swan calculation
#==============================================================================
runswan = False

harbor.project.settings["calculation_method"]["method"] = "advanced"
harbor.project.settings["calculation_method"]["include_pharos"] = True
harbor.project.settings["calculation_method"]["include_hares"] = False

harbor.project.settings["swan"]["swanfolder"] = os.path.join(tempdir, case, 'swan')
harbor.project.settings["swan"]["mastertemplate"] = os.path.join(datadir, 'swan', 'A2_template_production2.swn')
harbor.project.settings["swan"]["depthfile"] = os.path.join(datadir, 'swan', 'A2.dep')
harbor.project.settings["swan"]["use_incoming_wave_factors"] = True

# for step in ["I1", "I2", "I3"]:
#     # Genereer invoerbestanden
#     harbor.swan.generate(step=step)
#     swan_dst = os.path.join(harbor.project.settings["swan"]["swanfolder"],'iterations',step)

#     if runswan:
#         # Run SWAN
#         owd = os.getcwd()
#         os.chdir(swan_dst)
#         os.system('runcases.bat')
#         os.chdir(owd)
#     else:
#         # Copy testresults
#         common.copy_swan_files(datadir, harbor.project.settings["swan"]["swanfolder"], step)

#     harbor.swan.iteration_results.read_results(step=step)

harbor.swan.iteration_finished()
for step in ["D", "TR", "W"]:
    # Genereer invoerbestanden
    harbor.swan.generate(step=step)
    swan_dst = os.path.join(harbor.project.settings["swan"]["swanfolder"],'calculations',step)

    if runswan:
        # Run SWAN
        owd = os.getcwd()
        os.chdir(swan_dst)
        os.system('runcases.bat')
        os.chdir(owd)
    else:
        # Copy testresults
        common.unzip_swan_files(datadir, harbor.project.settings["swan"]["swanfolder"], step)

    harbor.swan.calculation_results.read_results(step=step)
#     harbor.swan.save_tables()

#==============================================================================
# run advanced pharos calculation
#==============================================================================
runpharos = False

harbor.project.settings["pharos"]["hydraulic loads"]["water depth for wave length"] = 10
harbor.project.settings["pharos"]["hydraulic loads"]["Hs_max"] = 5.2 # doet niks?

harbor.project.settings["pharos"]["frequencies"]["lowest"] = 0.082
harbor.project.settings["pharos"]["frequencies"]["highest"] = 0.25
harbor.project.settings["pharos"]["frequencies"]["number of bins"] = 17
harbor.project.settings["pharos"]["frequencies"]["scale"] = "logaritmisch"
harbor.project.settings["pharos"]["frequencies"]["checked"] = [
    0.082,
    0.0879,
    0.0942,
    0.1010,
    0.1083,
    0.1161,
    0.1245,
    0.1335,
    0.1431,
    0.1535,
    0.1645,
    0.1764,
    0.1891,
    0.2028,
    0.2174,
    0.2331,
    0.25,
]

harbor.project.settings["pharos"]["2d wave spectrum"]["gamma"] = 2
harbor.project.settings["pharos"]["2d wave spectrum"]["min energy"] = 0.01
harbor.project.settings["pharos"]["2d wave spectrum"]["spread"] = 30.0

harbor.project.settings["pharos"]["paths"]["pharos folder"] = os.path.join(tempdir, case, 'pharos')
harbor.project.settings["pharos"]["paths"]["schematisation folder"] = os.path.join(datadir, 'pharos', 'schematisations')

harbor.project.settings["pharos"]["water levels"]["checked"] = [4.0]

harbor.project.settings["pharos"]["transformation"]["dx"] = -240000
harbor.project.settings["pharos"]["transformation"]["dy"] = -600000

harbor.project.settings["pharos"]["schematisations"] = {
    "Grid_c02": [
        120.0,
        290.0,
        295.0,
        300.0,
        305.0,
        310.0,
        315.0,
        320.0,
        325.0,
        330.0,
        335.0,
        340.0,
        345.0,
        350.0,
        355.0,
        0.0,
        5.0,
        10.0,
        15.0,
        20.0,
        25.0,
        30.0,
        35.0,
        40.0,
        45.0,
        50.0,
        55.0,
        60.0,
        65.0,
        70.0,
        75.0,
        80.0,
        85.0,
        90.0,
        95.0,
        100.0,
        105.0,
        110.0,
        115.0,
    ]
}

# Genereer invoerbestanden
shutil.copytree(os.path.join(datadir, 'pharos', 'schematisations'), os.path.join(tempdir, case, 'pharos', 'schematisations'))
os.mkdir(os.path.join(tempdir, case, 'pharos', 'calculations'))
harbor.pharos.initialize()
harbor.pharos.fill_spectrum_table()
harbor.pharos.generate()

if runpharos:
    # Run PHAROS
    for grid,name in harbor.pharos.grid_names:
        owd = os.getcwd()
        os.chdir(os.path.join(tempdir, 'pharos', 'calculations', grid))
        os.system('launch_pharcone.bat')
        os.system('launch_pharos.bat')
        os.system('launch_rDPRA.bat')
        os.chdir(owd)
else:
    # Copy results
    with zipfile.ZipFile(os.path.join(datadir, 'pharos', 'json_output.zip')) as zip_ref:
        zip_ref.extractall(os.path.join(tempdir, case, 'pharos', 'calculations', 'Grid_c02', 'output'))

harbor.pharos.read_calculation_results()
harbor.pharos.assign_energies()

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
shutil.copy2(os.path.join(datadir, 'databases', dbname + '.sqlite'), os.path.join(tempdir, case))
shutil.copy2(os.path.join(datadir, 'databases', dbname + '.config.sqlite'), os.path.join(tempdir, case))
shutil.copy2(os.path.join(trunkdir, 'tests', 'data', 'hlcd.sqlite'), os.path.join(tempdir, case))

harbor.export.add_result_locations()
harbor.export.export_dataframe['Exportnaam'] = case + '_' + harbor.export.export_dataframe['Naam'] # Hier is de mogelijkheid om de namen aan de juiste conventie te laten voldoen. In de GUI gaat dit via een excel file
harbor.export.export_dataframe['SQLite-database'] = os.path.join(tempdir, case, dbname + '.sqlite')
harbor.export.add_HLCD(os.path.join(tempdir, case, 'hlcd.sqlite'))
harbor.export.export_output_to_database()
harbor.project.settings['export']['export_succeeded'] = True

harbor.save_tables()
harbor.project.save()

#==============================================================================
# plot schematisation
#==============================================================================
fig, ax = harbor.schematisation.plot_schematisation(buffer=10)
ax.legend(loc='upper right')
# fig.show()
fig.savefig(os.path.join(tempdir,case,case+'.png'), dpi=150)