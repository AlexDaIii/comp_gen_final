# Our Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import ElasticNet
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.decomposition import PCA
from matplotlib.pyplot import plot, show, savefig, xlim, figure, ylim, legend, boxplot, setp, axes, title, ylabel
import argparse
import os
import sys


def get_args():
    parser = argparse.ArgumentParser(description="This is the main test harness for your algorithms.")

    parser.add_argument("--features", type=str, required=True, help="The data to use for training.")
    parser.add_argument("--labels", type=str, required=True, help="The labels to use for training")
    parser.add_argument("--output_dir", type=str, required=True, help="The output directory")
    
    args = parser.parse_args()

    return args

def get_data(feature_file, label_file):

    features = pd.read_csv(feature_file)
    labels = pd.read_csv(label_file)
    # Sort that data
    labels_sorted = labels.sort_values('CCLE Cell Line Name')
    features_sorted = features.sort_values('CellLineName')

    labels_the_label = labels_sorted.iloc[:,3:].values

    features_mean_all_datatable = features_sorted.copy()
    missing_feats = ['H3K4me0', 'H3K4ac1', 'H3K18ac0K23ub1', 'H3K27ac1K36me0', 'H3K27ac1K36me1', 'H3K27ac1K36me2', 
                    'H3K27ac1K36me3', 'H3K56me1', 'H3K79me1', 'H3K79me2']
    for i in missing_feats:
        features_mean_all_datatable[i] = features_mean_all_datatable[i].fillna(features_sorted[i].mean())

    # Extracting features into array
    features_mean_all_nstand = features_mean_all_datatable.values[:,2:].astype(float)

    return features_mean_all_nstand, labels_the_label


def run_pca(features_mean_all_nstand, labels, dir):

    # Look at top 10 PCA components for all standardized data (without imputated missing data)
    # features_mean_all is a 458x42 matrix (samples x features)
    pca = PCA(n_components=10)
    princComp = pca.fit_transform(features_mean_all_nstand)

    # Setting up variables
    df_pca10 = pd.DataFrame(princComp,columns=(range(10)+np.ones(10)))
    y = pca.explained_variance_ratio_
    N = len(y)
    x = np.arange(N)

    # Plotting % explained variance from top 10 principal components
    mpl_fig = plt.figure()
    ax = mpl_fig.add_subplot(111)

    width = 0.5
    ax.bar(x, y*100, width, color="blue")
    ax.set_ylabel('% Explained Variance')
    ax.set_xlabel('Principal Components')
    ax.set_title('% Explained Variance from the top 10 principal components')

    ax.set_xticks(x) #+ width/3.)
    #ax.set_yticks(np.arange(0, 0.12, 10))
    ax.set_xticklabels(["{:02d}".format(x) for x in range(1,N+1)])

    mpl_fig.savefig(os.path.join(dir, '21_top_10_pc_exp_var.png'))

    # PCA for log(ActArea)

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.set_xlabel('Principal Component 1', fontsize=15)
    ax.set_ylabel('Principal Component 2', fontsize=15)
    ax.set_title('Top 2 principal components for log(ActArea)', fontsize=15)
    #targets = [0, 1]
    #markers = ['+', 'x']
    #for target, m in zip(targets,markers):
    #    indicesToKeep = df_Final['disease'] == target
    #    ax.scatter(df_Final.loc[indicesToKeep, 'principal component 1']
    #               , df_Final.loc[indicesToKeep, 'principal component 2']
    #               , marker = m
    #               , s = 50)
        
    a = ax.scatter(df_pca10[1],df_pca10[2], alpha=0.8, c = np.log(labels[:,2]), cmap = 'seismic')
    #ax.legend(['Control','Parkinson\'s'])
    ax.grid()
    cbar = fig.colorbar(a)
    fig.savefig(os.path.join(dir, '22_pc2_vs_pc1.png.png'))


def main():

    args = get_args()

    features_mean_all_nstand, labels = get_data(args.features, args.labels)

    run_pca(features_mean_all_nstand, labels, args.output_dir)

if __name__ == "__main__":
    main()