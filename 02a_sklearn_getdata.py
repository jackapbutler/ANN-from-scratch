import pandas as pd
import os 
os.system('kaggle datasets download -d uciml/pima-indians-diabetes-database')
os.system('move pima-indians* ./data')

import zipfile
with zipfile.ZipFile("./data/pima-indians-diabetes-database.zip","r") as zip_ref:
    zip_ref.extractall("./data/")

os.system('del .\data\pima-indians-diabetes-database.zip*')
