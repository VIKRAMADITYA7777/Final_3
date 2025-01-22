# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 22:10:40 2025

@author: HP
"""

import pandas as pd

def load_data():
    # Load the dataset from the CSV file
    df = pd.read_csv("re_cricketdata.csv")
    return df
