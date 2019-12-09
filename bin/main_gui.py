# -*- coding: utf-8 -*-
"""
Created on  : Thu Aug 24 16:54:58 2017
Author      : Guus Rongen, HKV Lijn in Water
Project     : PR3594.10.00
Description :

"""

import os
import sys

import numpy as np
import sqlite3
import matplotlib.pyplot as plt
import osgeo
import fiona
import pyproj
import pandas as pd
import geopandas as gpd

from PyQt5 import QtWidgets

# Toegevoegd Svasek 29/10/2018 - Importeren van module om warnings te onderdrukken
import warnings

os.chdir(os.path.dirname(os.path.abspath(__file__)))
if '..' not in sys.path:
    sys.path.append('..')

from hbhavens.ui.main import MainWindow
from hbhavens.core.models import MainModel

# Aangepast Svasek 03/10/18 - Andere manier van opstarten van de app
# from hbhavens import core

#app = QtWidgets.QApplication(sys.argv)
#app.setApplicationVersion('0.1')
#
## Create main window
#ex = MainWindow()
#
## Open project
## fname = os.path.join('colijnsplaat', 'advanced_swan_pharos.json')
## fname = os.path.join('ijmuiden', 'simple.json')
## ex.mainmodel = MainModel()
## ex.mainmodel.project.open_from_file(fname)
## ex.step = ex.mainmodel.project.settings['project']['progress']
#
## ex.initMain()
#
## ex.clearDirty()
## ex.updateStatus('')
#
#sys.exit(app.exec_())

# Toegevoegd Svasek 29/10/2018 - Importeren van module om warnings te onderdrukken
if not sys.warnoptions:
    warnings.simplefilter("ignore")
    
if not QtWidgets.QApplication.instance():
    app = QtWidgets.QApplication(sys.argv)
else:
    app = QtWidgets.QApplication.instance()
app.setApplicationVersion('0.1')

ex = MainWindow()

if len(sys.argv) > 1 and sys.argv[1].endswith('.json'):
    ex.open_project(fname=sys.argv[1])


app.exec_()