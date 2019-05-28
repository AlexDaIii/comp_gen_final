import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso
from sklearn.preprocessing import scale
import matplotlib.pyplot as plt
import argparse
import sys
import os
from hyperparameter import *
from sklearn.utils import resample
from decimal import Decimal


def get_args():
    parser = argparse.ArgumentParser(description="This is the main test harness for your algorithms.")

    parser.add_argument("--features", type=str, required=True, help="The data to use for training.")
    parser.add_argument("--labels", type=str, required=True, help="The labels to use for training")
    parser.add_argument("--output_dir", type=str, required=True, help="The output directory")
    parser.add_argument("--hyperparam", type=int, required=True, help="Tells if run the hyperparameter search")
    parser.add_argument("--load_new", type=int, required=True, help="Tells if load new data format or not")
    
    args = parser.parse_args()

    return args


def get_data(feature_file, label_file):

    # loading in the data
    features = pd.read_csv(feature_file)
    labels = pd.read_csv(label_file)

    # remove first row in features since it is not a name
    # then sort values

    labels_sorted = labels.sort_values('CCLE Cell Line Name')
    features_sorted = features.drop([0]).sort_values('Name')

    # get col names
    columns = features_sorted.columns[1:]

    # Data cleaning and data splitting

    # extract data points
    Y = labels_sorted.iloc[:,4:].values.astype(float)
    X_ns = features_sorted.iloc[:,1:].values.astype(float)

    # standardize values
    X_ni = scale(X_ns)

    # add intercept
    X = np.append(X_ni, np.ones((491,1)),1)

    return columns, X, Y


def get_data_new(feature_file, label_file):

    # loading in the data
    features = pd.read_csv(feature_file, skiprows=[1])
    labels = pd.read_csv(label_file)

    # remove first row in features since it is not a name
    # then sort values

    labels_sorted = labels.sort_values('CCLE Cell Line Name')
    features_sorted = features.sort_values('Name')

    # get col names
    columns = features_sorted.columns[1:]

    # Data cleaning and data splitting

    # extract data points
    Y = labels_sorted.iloc[:,3:].values.astype(float)
    X_ns = features_sorted.iloc[:,1:].values.astype(float)

    # standardize values
    X_ni = scale(X_ns)

    # add intercept
    X = np.append(X_ni, np.ones((491,1)),1)

    return columns, X, Y


def bootstrap(fin_lambda, fin_alpha, X_train, Y_train, bag_size=250, num_bags=200):
    """
    fin_lambda = the lambda to use for elastic net
    fin_alpha = the alpha to use for elastic net
    X_train = training x data
    Y_train = training y data
    bag_size = the number examples to be sampled
    num_bags = the number of resampled datasets
    """
    cost_train = 0
    rsquared = []
    weights = []
    
    # create the resampled datasets
    for i in range(num_bags):
        
        boot_X, boot_Y = resample(X_train, Y_train)
        
        # do elastic net on the resampled data
        regr = ElasticNet(alpha=fin_lambda, l1_ratio=fin_alpha, fit_intercept=False, tol = 0.01)
        regr.fit(boot_X, boot_Y)
        
        # TODO: save the weights of the trained model
        weights.append(regr.coef_)
        
        # compute mse for training
        yhat_train = regr.predict(boot_X)
        cost_train += mean_squared_error(boot_Y, yhat_train)
        
        # didn't compute out of sample (val) error because the paper doesn't mention it
        
        # save all r squared
        rsquared.append(regr.score(boot_X, boot_Y))
        
    
    # average cost of training across all the boostrapped examples
    cost_train /= num_bags
    
    return np.array(weights), cost_train


def gene_regression(X_train, y_train, dir):

    kf5 = KFold(n_splits=5, shuffle=True, random_state=42)

    alphas = np.arange(0.2, 1.05, 0.05)
    lambdas = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]

    mse_m_ic50_5, mse_train_m_ic50_5, rsquared_m_ic50_5 = hyperparameter(lambdas, alphas, X_train, y_train[:, 0], kf5)
    mse_m_amax_5, mse_train_m_amax_5, rsquared_m_amax_5 = hyperparameter(lambdas, alphas, X_train, y_train[:, 1], kf5)
    mse_m_actarea_5, mse_train_m_actarea_5, rsquared_m_actarea_5 = hyperparameter(lambdas, alphas, X_train, np.log1p(y_train[:, 2]), kf5)

    compare_ic50_5 = mse2df(mse_m_ic50_5, lambdas, alphas, rsquared_m_ic50_5, mse_train_m_ic50_5)
    compare_amax_5 = mse2df(mse_m_amax_5, lambdas, alphas, rsquared_m_amax_5, mse_train_m_amax_5)
    compare_act_5 = mse2df(mse_m_actarea_5, lambdas, alphas, rsquared_m_actarea_5, mse_train_m_actarea_5)

    compare_ic50_5.nsmallest(10, 'MSE Test').to_csv(os.path.join(dir, '08_ic50_mse_smallest_gene.csv'))
    compare_ic50_5.nlargest(10, 'R2').to_csv(os.path.join(dir, '09_ic50_r2_largest_gene.csv'))

    compare_amax_5.nsmallest(10, 'MSE Test').to_csv(os.path.join(dir, '10_amax_mse_smallest_gene.csv'))
    compare_amax_5.nlargest(10, 'R2').to_csv(os.path.join(dir, '11_amax_r2_largest_gene.csv'))

    compare_act_5.nsmallest(10, 'MSE Test').to_csv(os.path.join(dir, '12_actarea_mse_smallest_gene.csv'))
    compare_act_5.nlargest(10, 'R2').to_csv(os.path.join(dir, '13_actarea_r2_largest_gene.csv'))  


def good_features_df(columns, percentages):
    # creates a dataframe that shows each of the the percentage of bootstrap datasets in which 
    # the feature was inferred as significant 
    feat_names = []
    table = []
    
    for col, pct in zip(columns, percentages):
        table.append(pct)
        feat_names.append(col)

    features = np.array(feat_names)
    data = np.array(table)
    compare = pd.DataFrame(features, columns=['Feature'])
    compare['Pct Sigf'] = data
    
    return compare


def output_mean_weights(output_file, idx, mean):

    file = open(output_file, 'w')

    weights = mean[idx]

    for s in weights:
        string = str(s) + '\t'
        file.write(string)
    
    file.close()


def calculate_mse(beta, x, y_train):
    """
    beta - [# bags, n feats]
    x = [# examples, # feats]
    """
    
    y_squigle_hat = np.mean(np.dot(beta, x.T), axis=0)
    cost = mean_squared_error(y_squigle_hat, y_train)
    
    return cost


def output_mse(output, c_act, cost_act, c_amax, cost_amax, c_ic50, cost_ic50):

    file = open(output, 'w')

    string = "Train Cost Activity Area: " + str('%.3F' % Decimal(c_act)) + "\tTest Cost Activity Area: " + str('%.2F' % Decimal(cost_act)) + '\n'
    file.write(string)

    string = "Train Cost A_max: " + str('%.2F' % Decimal(c_amax)) + "\tTest Cost Activity Area: " + str('%.2F' % Decimal(cost_amax)) + '\n'
    file.write(string)

    string = "Train Cost Activity Area: " + str('%.2F' % Decimal(c_ic50)) + "\tTest Cost Activity Area: " + str('%.2F' % Decimal(cost_ic50)) + '\n'
    file.write(string)

    file.close()

    pass


def perform_boot(X_train, X_test, y_train, y_test, columns, dir):

    # the "best" lambdas and alphas
    ic50_la = [0.3, 1.00]
    amax_la = [1.0, 1.00]
    act_la = [0.03, 0.80] # this is the best one

    w_act, c_act = bootstrap(act_la[0], act_la[1], X_train, y_train[:,2])
    w_amax, c_amax = bootstrap(amax_la[0], amax_la[1], X_train, y_train[:,1])
    w_ic50, c_ic50 = bootstrap(ic50_la[0], ic50_la[1], X_train, y_train[:,0])

    # the percentage of bootstrap datasets in which it was inferred as significant
    good_act = np.sum(w_act > 0, axis=0)/200
    good_amax = np.sum(w_amax > 0, axis=0)/200
    good_ic50 = np.sum(w_ic50 > 0, axis=0)/200

    # creates a dataframe that shows each of the the percentage of bootstrap datasets in which 
    # the feature was inferred as significant 
    feats_act = good_features_df(columns, good_act)
    feats_amax = good_features_df(columns, good_amax)
    feats_ic50 = good_features_df(columns, good_ic50)

    # get top 10 features
    feats_act.nlargest(10, 'Pct Sigf').to_csv(os.path.join(dir, '14_actarea_sig_feats.csv'))
    feats_amax.nlargest(10, 'Pct Sigf').to_csv(os.path.join(dir, '15_amax_sig_feats.csv'))
    feats_ic50.nlargest(10, 'Pct Sigf').to_csv(os.path.join(dir, '16_ic50_sig_feats.csv'))

    # report the mean weights
    mean_act = np.mean(w_act, axis=0)
    mean_amax = np.mean(w_amax, axis=0)
    mean_ic50 = np.mean(w_ic50, axis=0)

    # we sadly hardcoded this because we didnt want to write more code and this was faster
    good_act_idx = [1218, 2061, 6772, 4862, 2216, 2488, 3254, 5003, 2129, 7908]
    good_amax_idx = [15505, 7924, 3837, 836, 14195, 8055, 4361, 1957, 2261, 10794]
    good_ic50_idx = [3837, 7924, 16646, 2261, 9033, 10794, 15343, 16306, 9232, 8637]

    output_mean_weights(os.path.join(dir, '17_actarea_weights.txt'), good_act_idx, mean_act)
    output_mean_weights(os.path.join(dir, '18_amax_weights.txt'), good_amax_idx, mean_amax)
    output_mean_weights(os.path.join(dir, '19_ic50_weights.txt'), good_ic50_idx, mean_ic50)

    # get the test mse
    cost_act = calculate_mse(w_act, X_test, y_test[:,2])
    cost_amax = calculate_mse(w_amax, X_test, y_test[:,1])
    cost_ic50 = calculate_mse(w_ic50, X_test, y_test[:,0])

    output_mse(os.path.join(dir, '20_mse.txt'), c_act, cost_act, c_amax, cost_amax, c_ic50, cost_ic50)


def main():

    args = get_args()

    if args.load_new:
        columns, X, Y = get_data_new(args.features, args.labels)
    else:
        columns, X, Y = get_data(args.features, args.labels)

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    if args.hyperparam:
        gene_regression(X_train, y_train, args.output_dir)

    perform_boot(X_train, X_test, y_train, y_test, columns, args.output_dir)


if __name__ == "__main__":
    main()