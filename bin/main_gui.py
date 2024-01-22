# -*- coding: utf-8 -*-
"""
Created on  : Thu Aug 24 16:54:58 2017
Author      : Guus Rongen, HKV Lijn in Water
Project     : PR3594.10.00
Description :

"""

import os
import sys
import warnings
from PyQt5 import QtWidgets

hbhdir = os.path.dirname(os.path.dirname(__file__))
if hbhdir not in sys.path:
    sys.path.append(hbhdir)
from hbhavens.ui.main import MainWindow

if not sys.warnoptions:
    warnings.simplefilter("ignore")
    
if not QtWidgets.QApplication.instance():
    app = QtWidgets.QApplication(sys.argv)
else:
    app = QtWidgets.QApplication.instance()
app.setApplicationVersion('23.1.1')

ex = MainWindow()

if len(sys.argv) > 1 and sys.argv[1].endswith('.json'):
    ex.open_project(fname=sys.argv[1])

app.exec_()