# -*- coding: utf-8 -*-
"""
Created on  : Mon Jul 24 16:54:58 2017
Author      : Matthijs Benit, HKV Lijn in Water
Project     : PR4982.10.00
Description :

Please note that the selected steunpuntlocatie is far from the project area to the north. As such, it will not be visible in the overview unless the user extends the view far beyond Urk harbour.

"""

import os
import sys
import warnings
import shutil
from PyQt5 import QtWidgets

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from common import set_paths

case = 'urk'
trunkdir, tempdir, datadir = set_paths(__file__, case)
from hbhavens.ui.main import MainWindow

if not sys.warnoptions:
    warnings.simplefilter("ignore")
    
if not QtWidgets.QApplication.instance():
    app = QtWidgets.QApplication(sys.argv)
else:
    app = QtWidgets.QApplication.instance()

if os.path.exists(os.path.join(tempdir, case, case)):
    shutil.rmtree(os.path.join(tempdir, case, case), ignore_errors=True)
shutil.copy2(os.path.join(datadir, case + '.json'), os.path.join(tempdir, case))
shutil.copytree(os.path.join(datadir, case), os.path.join(tempdir, case, case))
shutil.copy2(os.path.join(datadir, 'databases', 'WBI2017_IJsselmeer_7-2_v02' + '.sqlite'), os.path.join(tempdir, case))

ex = MainWindow()
ex.open_project(fname=os.path.join(tempdir, case, case + '.json'))


app.exec_()