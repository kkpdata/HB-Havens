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
import matplotlib.pyplot as plt
from shapely.plotting import patch_from_polygon as PolygonPatch
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from common import set_paths

case = 'ketelhaven'
trunkdir, tempdir, datadir = set_paths(__file__, case)
from hbhavens.core.models import MainModel

for dam in ['havendam','havendam_los1','havendam_los2','havendam_fd1','havendam_fd2',]:
    for terrein in ['haventerrein','haventerrein_los','haventerrein_nb1','haventerrein_nb2','haventerrein_buiten','haventerrein_buitenlos']:

        try:
            casename = f'ketelhaven_{dam}_{terrein}'

            harbor = MainModel()
            harbor.project.settings['project']['name'] = casename
            harbor.project.settings['project']['user']['email'] = 'benit@hkv.nl'
            harbor.project.settings['project']['user']['name'] = 'Matthijs Bénit'
            harbor.project.save_as(os.path.join(tempdir, case, casename) + '.json')

            harbor.schematisation.add_flooddefence('8-4')

            harbor.schematisation.add_harborarea(os.path.join(datadir, 'shapes', f'{terrein}.shp'))
            harbor.schematisation.add_breakwater(os.path.join(datadir, 'shapes', f'{dam}.shp'))
            harbor.schematisation.generate_harbor_bound()

            harbor.schematisation.add_result_locations(os.path.join(datadir, 'shapes', 'uitvoerlocaties.shp'))
            harbor.schematisation.set_bedlevel(-5.0)
            harbor.hydraulic_loads.add_HRD(os.path.join(datadir, 'databases', 'WBI2017_IJsselmeer_8-4_v02_steunpunt.sqlite'))
            harbor.schematisation.set_selected_support_location('KM_2_8-4_dk_00062')

            harbor.project.settings['calculation_method']['method'] = 'simple'
            harbor.simple_calculation.run_all()
            harbor.modeluncertainties.add_result_locations()
            harbor.modeluncertainties.load_modeluncertainties()
            harbor.modeluncertainties.table.loc[:, [' '.join(vals) for vals in harbor.modeluncertainties.supportloc_unc.unstack().index]] = harbor.modeluncertainties.supportloc_unc.unstack().values

            harbor.project.settings["export"]["export_HLCD_and_config"] = False
            shutil.copy2(os.path.join(datadir, 'databases', f'WBI2017_IJsselmeer_8-4_v02_steunpunt.sqlite'), os.path.join(tempdir, case, casename ,f'WBI2017_IJsselmeer_8-4_v02_steunpunt_{casename}.sqlite'))

            harbor.export.add_result_locations()
            harbor.export.export_dataframe['Exportnaam'] = casename + '_' + harbor.export.export_dataframe['Naam'] 
            harbor.project.settings['export']['HRD'] = os.path.join(tempdir, case, casename, f'WBI2017_IJsselmeer_8-4_v02_steunpunt_{casename}.sqlite')
            harbor.export.export_output_to_database()

            harbor.schematisation.plot_schematisation(save=True, path=f'{tempdir}\\{case}', close=True)

            harbor.project.save()
            print(f'{casename} - Geldig')
        except:
            print(f'{casename} - Ongeldig')

            buffer=200
            locations=True
            figsize=(15,15)

            fig, ax = plt.subplots(figsize=figsize)
            ax.set_aspect(1.0)

            harbor.schematisation.harborarea.read_file(os.path.join(datadir, 'shapes', f'{terrein}.shp'))
            harbor.schematisation.breakwaters.read_file(os.path.join(datadir, 'shapes', f'{dam}.shp'))

            # plot harborarea
            if len(harbor.schematisation.harborarea) != 0:
                ax.add_patch(PolygonPatch(harbor.schematisation.harborarea.geometry[0], facecolor='C0', edgecolor=None, zorder=2, alpha=0.5, label='Haventerrein'))
                if len(harbor.schematisation.harborarea.geometry) > 1:
                    for geom in harbor.schematisation.harborarea.geometry[1:]:
                        ax.add_patch(PolygonPatch(geom, facecolor='C0', edgecolor=None, zorder=2, alpha=0.5))

            # Plot breakwaters
            if len(harbor.schematisation.breakwaters) != 0:
                for breakwater in harbor.schematisation.breakwaters.itertuples():
                    if breakwater.Index == 0:
                        ax.plot(*breakwater.geometry.xy, color='C1', label='Golfbreker')
                    else:
                        ax.plot(*breakwater.geometry.xy, color='C1')

            # Set limits (without flooddefence)
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()

            # Plot result locations with names
            if locations and len(harbor.schematisation.result_locations) != 0:
                harbor.schematisation.result_locations.plot(ax=ax, color='k', marker='.', zorder=10, label='Uitvoerlocaties')

            # Plot support location with names
            if locations and len(harbor.schematisation.support_locations) != 0:
                harbor.schematisation.support_locations.plot(ax=ax, color='k', marker='x', zorder=10, label='Steunpuntlocatie')
                ax.plot(harbor.schematisation.support_location.XCoordinate, harbor.schematisation.support_location.YCoordinate, markeredgecolor='k', marker='x', zorder=10)

            # Plot flood defence
            if len(harbor.schematisation.flooddefence) != 0:
                harbor.schematisation.flooddefence.plot(ax=ax, color='C3', label='Normtraject')

            # Update limits for locations
            if locations and len(harbor.schematisation.support_locations) != 0:
                ax.set_xlim(xmin=np.nanmin([harbor.schematisation.support_location.XCoordinate,xlim[0]]) - buffer, xmax=np.nanmax([harbor.schematisation.support_location.XCoordinate,xlim[1]]) + buffer)
                ax.set_ylim(ymin=np.nanmin([harbor.schematisation.support_location.YCoordinate,ylim[0]]) - buffer, ymax=np.nanmax([harbor.schematisation.support_location.YCoordinate,ylim[1]]) + buffer)
            else:
                ax.set_xlim(xmin=xlim[0] - buffer, xmax=xlim[1] + buffer)
                ax.set_ylim(ymin=ylim[0] - buffer, ymax=ylim[1] + buffer)
           
            fig.savefig(f'{tempdir}\\{case}\\{casename}_Ongeldig.png')
            plt.close(fig)
