# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 10:35:04 2018

@author: billg_000
"""

### Test for Torch
import torch
import sqlite3

print("Torch version:", torch.__version__)
print("Torch path:", torch.__path__)
print("Has cuda?:", torch.cuda.is_available())

print("Number of cuda devices:", torch.cuda.device_count())

db = sqlite3.connect('run_results.sqlite')
# Get a cursor object
cursor = db.cursor()
cursor.execute('''
    CREATE TABLE IF NOT EXISTS run_results(id INTEGER PRIMARY KEY, run_name TEXT,
                       layers TEXT, transfer_functions TEXT, EPOCHS INTEGER,
                       net_description TEXT, PREDICTIONS TEXT, TIME REAL)
''')
db.commit()
db.close()