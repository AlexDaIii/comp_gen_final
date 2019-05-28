import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso
from sklearn.preprocessing import scale
from matplotlib.pyplot import plot, show, savefig, xlim, figure, ylim, legend, boxplot, setp, axes, title, ylabel
import matplotlib.pyplot as plt
import argparse
import sys
import os
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
from hyperparameter import *


def get_args():
    parser = argparse.ArgumentParser(description="This is the main test harness for your algorithms.")

    parser.add_argument("--features", type=str, required=True, help="The data to use for training.")
    parser.add_argument("--labels", type=str, required=True, help="The labels to use for training")
    parser.add_argument("--output_dir", type=str, required=True, help="The output directory")
    
    args = parser.parse_args()

    return args


def get_data(feature_file, label_file):

    # loading in the data
    features = pd.read_csv(feature_file)
    labels = pd.read_csv(label_file)

    # sort the data
    labels_sorted = labels.sort_values('CCLE Cell Line Name')
    features_sorted = features.sort_values('CellLineName')

    col_to_drp = ['H3K4ac1', 'H3K18ac0K23ub1', 'H3K56me1']
    feats = features.drop(columns=col_to_drp)

    # fill the na with mean of the whole entire column
    # fill_feat_zero = feats.fillna(0)
    fill_feat_mean = feats.fillna(feats.mean())

    # extract data into arrays
    Y = labels_sorted.iloc[:,3:].values
    # X_zero_ns = fill_feat_zero.iloc[:, 2:].values
    X_mean_ns = fill_feat_mean.iloc[:, 2:].values

    # Standardize Data
    # X_zero_ni = scale(X_zero_ns)
    X_mean_ni = scale(X_mean_ns)

    # add intercept
    # X_zero = np.append(X_zero_ni, np.ones((458,1)), 1)
    X_mean = np.append(X_mean_ni, np.ones((458,1)), 1)

    return X_mean_ni, X_mean, Y


def histone_regression(X_mean, Y, dir):

    X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(X_mean, Y, test_size=0.2, random_state=42)
    kf = KFold(n_splits=10, shuffle=True, random_state=42)

    alphas = np.arange(0.2, 1.05, 0.05)
    lambdas = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]

    mse_m_ic50, mse_train_m_ic50, rsquared_m_ic50 = hyperparameter(lambdas, alphas, X_train_m, y_train_m[:, 0], kf)
    mse_m_amax, mse_train_m_amax, rsquared_m_amax = hyperparameter(lambdas, alphas, X_train_m, y_train_m[:, 1], kf)
    mse_m_actarea, mse_train_m_actarea, rsquared_m_actarea = hyperparameter(lambdas, alphas, X_train_m, y_train_m[:, 2], kf)


    compare_ic50 = mse2df(mse_m_ic50, lambdas, alphas, rsquared_m_ic50, mse_train_m_ic50)
    compare_amax = mse2df(mse_m_amax, lambdas, alphas, rsquared_m_amax, mse_train_m_amax)
    compare_act = mse2df(mse_m_actarea, lambdas, alphas, rsquared_m_actarea, mse_train_m_actarea)


    compare_ic50.nsmallest(10, 'MSE Test').to_csv(os.path.join(dir, '00_ic50_mse_smallest.csv'))
    compare_ic50.nlargest(10, 'R2').to_csv(os.path.join(dir, '01_ic50_r2_largest.csv'))

    compare_amax.nsmallest(10, 'MSE Test').to_csv(os.path.join(dir, '02_amax_mse_smallest.csv'))
    compare_amax.nlargest(10, 'R2').to_csv(os.path.join(dir, '03_amax_r2_largest.csv'))

    compare_act.nsmallest(10, 'MSE Test').to_csv(os.path.join(dir, '04_actarea_mse_smallest.csv'))
    compare_act.nlargest(10, 'R2').to_csv(os.path.join(dir, '05_actarea_r2_largest.csv'))    


def spearman(X_mean_ni, output):

    df_corr_mean_ni = pd.DataFrame(X_mean_ni)  
    s_corr_mean_ni = df_corr_mean_ni.corr(method='spearman')

    ax3 = sns.heatmap(s_corr_mean_ni, cmap='coolwarm')
    ax3.set(title='Spearman Correlation Features - Mean, Standardized')
    plt.savefig(os.path.join(output, '06_Spearman.png'))


def fdr_test(X, Y):
    """
    X - the x data - n * m (examples vs features) matrix
    Y - the y data - n * 1 vector of labels
    """
    
    peas = np.ones((X.shape[1],))
    for i in range(X.shape[1]):
        row_int = np.append(np.ones((X.shape[0],1)), X[:,i].reshape(X.shape[0], 1), axis=1)
        # 0 = ic50, 1 = amax, 2 = act area
        model = sm.OLS(Y, row_int)
        results = model.fit()
        peas[i] = results.pvalues[1]
        
    reject, _, _, _ = multipletests(peas, method='fdr_bh')
    
    return reject, peas


def output_file(output, results_d_ic50, results_d_amax, results_d_acta):
    
    file = open(os.path.join(output, '07_indv_lin_regression.txt'), 'w')
    
    string = 'IC50 signifigant features: ' + str(np.sum(results_d_ic50)) + '\n'
    file.write(string)
    string = 'A_max signifigant features: ' + str(np.sum(results_d_amax)) + '\n'
    file.write(string)
    string = 'Activity Area signifigant features: ' + str(np.sum(results_d_acta)) + '\n'
    file.write(string)
    
    file.close()


def test_linear(X_mean_ni, Y, output):

    # Multiple Hypothesis IC50
    results_d_ic50, _ = fdr_test(X_mean_ni, Y[:,0])
    # Multiple Hypothesis AMax
    results_d_amax, _ = fdr_test(X_mean_ni, Y[:,1])
    # Multiple Hypothesis ActArea
    results_d_acta, _ = fdr_test(X_mean_ni, Y[:,2])

    output_file(output, results_d_ic50, results_d_amax, results_d_acta)


def main():

    args = get_args()
    X_mean_ni, X_mean, Y = get_data(args.features, args.labels)

    histone_regression(X_mean, Y, args.output_dir)

    spearman(X_mean_ni, args.output_dir)

    test_linear(X_mean_ni, Y, args.output_dir)

if __name__ == "__main__":
    main()