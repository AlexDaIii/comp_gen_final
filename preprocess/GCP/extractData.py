#!/usr/bin/env python3

import sys
import os
import argparse
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser(description='Extract relevant data from csv files')
parser.add_argument('--X', dest='feats', help='Features to go into the model')
parser.add_argument('--Y', dest='lab', help='Labels corresponding to the features')
parser.add_argument('--outputY', dest='outputY', help='csv file to contain labels')
parser.add_argument('--outputX', dest='outputX', help='csv file to contain features')
args = parser.parse_args()

df_X = pd.read_csv(args.feats)
df_Y = pd.read_csv(args.lab)

labs = df_Y[df_Y.Compound == 'PD-0325901'][['CCLE Cell Line Name', 'Primary Cell Line Name','EC50 (uM)', 'IC50 (uM)','Amax','ActArea']]
filtered_X = df_X[df_X['CellLineName'].isin(labs['CCLE Cell Line Name'])]
filtered_Y = labs[labs['CCLE Cell Line Name'].isin(df_X['CellLineName'])]

labels_sorted = filtered_Y.sort_values('CCLE Cell Line Name')
features_sorted = filtered_X.sort_values('CellLineName')

labels_sorted.to_csv(args.outputY, sep=',', header=True, index=False)
features_sorted.to_csv(args.outputX, sep=',', header=True, index=False)
