# -*- coding: utf-8 -*-
"""
Created on  : Mon Sep 18 15:14:10 2017
Author      : Guus Rongen, HKV Lijn in Water
Project     : PR3594.10.00
Description : 
    
"""

import sqlite3
import shutil
import os
import numpy as np
import pandas as pd

# Make copy
src = r'd:\Documents\3556.10 Productiesommen Oosterschelde WBI2017\Werkmap\PR3556.10.03_WAQUA\WBI2023_Oosterschelde_28-1_v00.sqlite'
dst = r'WBI2023_Oosterschelde_28-1_v00_selectie.sqlite'
if os.path.exists(dst):
    os.remove(dst)

shutil.copy2(src, dst)

# Connect
conn = sqlite3.connect(dst)

# Loop trough tables and drop content of specific tables
for table in ['HRDLocations', 'HydroDynamicInputData', 'HydroDynamicResultData', 'HydroDynamicData', 'UncertaintyModelFactor']:
    conn.execute('DELETE FROM {};'.format(table))

# Close and connect to new db
conn.commit()
conn.close()

conn = sqlite3.connect(src)

# Attach old full db
conn.execute("ATTACH DATABASE ? AS dstdb;", (dst,))

hrdlocationids = ', '.join([str(i) for i in [136, 137, 138, 139, 140]])
winddirectionids = ', '.join([str(i) for i in [1, 15, 16]])
closingsituationids = ', '.join([str(i) for i in [1, 4]])

# Select load ids from data
query = 'SELECT HydraulicLoadId FROM HydroDynamicData WHERE HRDWindDirectionId IN ({});'.format(winddirectionids)
hydraulicloadids = np.unique(np.hstack(conn.execute(query).fetchall()))
# Select load ids from HydroDynamicInputData
hdid = pd.read_sql('SELECT * FROM HydroDynamicInputData WHERE HRDInputColumnId IN (1, 3);', conn, index_col=['HydraulicLoadId', 'HRDInputColumnId']).unstack()
hdid.columns = hdid.columns.get_level_values(1)
for col, selection in zip([1, 3], [[2.5, 3.5], [20]]):
    hdid = hdid.loc[np.isin(hdid[col], selection), :]

# Combine selections
hydraulicloadids = ', '.join(hydraulicloadids[np.isin(hydraulicloadids, hdid.index.values)].astype(str).tolist())

# Todo: add variableids
query = """
    INSERT INTO dstdb.HydroDynamicResultData
    SELECT *
    FROM HydroDynamicResultData
    WHERE HRDLocationId IN ({}) AND HydraulicLoadId IN ({});""".format(hrdlocationids, hydraulicloadids)
conn.execute(query)

query = """
    INSERT INTO dstdb.HRDLocations
    SELECT *
    FROM HRDLocations
    WHERE HRDLocationId IN ({});""".format(hrdlocationids)
conn.execute(query)
        
query = """
    INSERT INTO dstdb.HydroDynamicData
    SELECT *
    FROM HydroDynamicData
    WHERE HydraulicLoadId IN ({}) AND HRDWindDirectionId IN ({});""".format(hydraulicloadids, winddirectionids)
conn.execute(query)

query = """
    INSERT INTO dstdb.HydroDynamicInputData
    SELECT *
    FROM HydroDynamicInputData
    WHERE HydraulicLoadId IN ({});""".format(hydraulicloadids)
conn.execute(query)

query = """
    INSERT INTO dstdb.UncertaintyModelFactor
    SELECT *
    FROM UncertaintyModelFactor
    WHERE HRDLocationId IN ({}) AND ClosingSituationId IN ({});""".format(hrdlocationids, closingsituationids)
conn.execute(query)

conn.commit()
conn.close()

conn = sqlite3.connect(dst)
conn.execute('VACUUM;')
conn.commit()
conn.close()

