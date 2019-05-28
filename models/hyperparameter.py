import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error
import pandas as pd

def hyperparameter(lambdas, alphas, X_train, y_train, kf):
    # lambda is alpha in sklearn's elastic net
    # alpha is l1 ratio in sklearn's elastic net
    
    mse = np.zeros((len(lambdas)*len(alphas),))
    mse_train = np.zeros((len(lambdas)*len(alphas)))
    rsquared = np.zeros((len(lambdas)*len(alphas), kf.n_splits))
    counter = 0
    
    for lam in lambdas:
        for alp in alphas:
            cost = 0
            cost_train = 0
            r = 0
            
            for train_index, test_index in kf.split(X_train):
                
                # get the train test split
                kf_X_train, kf_X_test = X_train[train_index], X_train[test_index]
                kf_y_train, kf_y_test = y_train[train_index], y_train[test_index]
                
                # do elastic net
                regr = ElasticNet(alpha=lam, l1_ratio=alp, fit_intercept=False)
                regr.fit(kf_X_train, kf_y_train)
                
                # compute mse
                yhat = regr.predict(kf_X_test)
                cost += mean_squared_error(kf_y_test, yhat)
                
                # compute mse for training
                yhat_train = regr.predict(kf_X_train)
                cost_train += mean_squared_error(kf_y_train, yhat_train)
                
                # save all r squared
                rsquared[counter, r] = regr.score(kf_X_train, kf_y_train)
                r += 1
                
            # average mse from kfolds
            mse[counter] = cost/kf.n_splits
            mse_train[counter] = cost_train/kf.n_splits
            counter += 1
            
    return mse, mse_train, rsquared


def mse2df(mse, lambdas, alphas, rsquared, mse_train):
    # creates a dataframe that shows the mse and average r^2 for each lambda and alpha value
    # mse is from out of set examples in 10-folds cv
    # r^2 is the score from how well the model fits the training data
    counter = 0
    table = []
    
    avgr = np.average(rsquared, axis=1)
    for lam in lambdas:
        for alp in alphas:
            table.append([lam, alp, mse[counter], mse_train[counter], avgr[counter]])
            counter += 1

    data = np.array(table)
    compare = pd.DataFrame(data, columns=['Lambda', 'Alpha', 'MSE Test', 'MSE Train', 'R2'])
    
    return compare

