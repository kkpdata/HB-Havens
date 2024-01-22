# -*- coding: utf-8 -*-
"""
Created on  : Mon Jul 24 16:54:58 2017
Author      : Matthijs Benit, HKV Lijn in Water
Project     : PR4982.10.00
Description :

"""

import os
import sys
import warnings
import shutil
from PyQt5 import QtWidgets

trunkdir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if trunkdir not in sys.path:
    sys.path.append(trunkdir)
from hbhavens.ui.main import MainWindow

if not sys.warnoptions:
    warnings.simplefilter("ignore")
    
if not QtWidgets.QApplication.instance():
    app = QtWidgets.QApplication(sys.argv)
else:
    app = QtWidgets.QApplication.instance()
app.setApplicationVersion('23.1.1')

ex = MainWindow()

datadir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
tempdir = os.path.join(os.path.dirname(trunkdir),'temp')
case = 'lauwersoog'

if not os.path.exists(tempdir):
    os.mkdir(tempdir)

if os.path.exists(os.path.join(tempdir, case)):
    shutil.rmtree(os.path.join(tempdir, case), ignore_errors=True)
if os.path.isfile(os.path.join(tempdir, case + '.json')):
    os.remove(os.path.join(tempdir, case + '.json'))

shutil.copy2(os.path.join(datadir, case, case + '.json'), tempdir)
shutil.copytree(os.path.join(datadir, case, case), os.path.join(tempdir, case))
ex.open_project(fname=os.path.join(tempdir, case + '.json'))

app.exec_()