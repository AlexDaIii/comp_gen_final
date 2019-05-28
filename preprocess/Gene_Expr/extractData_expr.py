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

df_X = pd.read_csv(args.feats, sep = '\t', index_col = [0, 1], skiprows=[0, 1])
df_Y = pd.read_csv(args.lab)

labs = df_Y[df_Y.Compound == 'PD-0325901'][['CCLE Cell Line Name', 'Primary Cell Line Name','EC50 (uM)', 'IC50 (uM)','Amax','ActArea']]
# Filter only for relevant cell lines in both features and label datasets
filtered_Y = labs[labs['CCLE Cell Line Name'].isin(df_X.columns)]
filtered_gene_expr = df_X.T[df_X.columns.isin(filtered_Y['CCLE Cell Line Name'])]

# Sort the datasets in alphabetical order
features_sorted = filtered_gene_expr.sort_index()
labels_sorted = filtered_Y.sort_values('CCLE Cell Line Name')

labels_sorted.to_csv(args.outputY, sep=',', header=True, index=False)
features_sorted.to_csv(args.outputX, sep=',', header=True, index=True)
