# -*- coding: utf-8 -*-

import numpy as np
from scipy.stats import multivariate_normal
import seaborn as sns
import esda
import geopandas as gpd
from libpysal.weights import Queen
from pykrige.ok import OrdinaryKriging
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from loess.loess_2d import loess_2d # https://pypi.org/project/loess/#toc-entry-1
from statsmodels.nonparametric.kernel_regression import KernelReg
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV

def make_data(grid_size, true_price_per_isq, lv_center, lv_decay, lv_noise_sd, isq_noise_sd, iv_noise_factor, iv_noise_sd, tv_obs_noise, isq_from_lv):
    x, y = np.mgrid[0:grid_size, 0:grid_size]
    pos = np.empty(x.shape + (2,))
    pos[:, :, 0] = x
    pos[:, :, 1] = y

    center    = (grid_size / 2, grid_size / 2)
    distances = np.sqrt((pos[:, :, 0] - center[0])**2 + (pos[:, :, 1] - center[1])**2)
    distances = np.random.normal(distances, lv_noise_sd, pos.shape[:2]) # there used to be (accidentally) a magic number rescaling lv_noise_sd...i've removed it and rescaled lv_noise_sd

    lv_true = lv_center * np.exp(-lv_decay*distances)

    isq_true = np.random.normal(lv_true, abs(lv_true*isq_noise_sd), size=lv_true.shape) # i think the noise in square footage should also be proportional to land value. you see bigger variations in isq and other improvements downtown (skyscraper next to vacant) than in rural (sfh next to vacant)
    np.clip(isq_true, 0, None, out=isq_true) # these are effectively vacant lots.

    iv_raw  = true_price_per_isq * isq_true
    iv_true = np.random.normal(iv_raw, abs(iv_raw * iv_noise_factor))
    iv_true = np.clip(iv_true, 0, None)
    
    tv_true  = lv_true + iv_true

    row_indices = np.repeat(np.arange(tv_true.shape[0]), tv_true.shape[1])
    col_indices = np.tile(np.arange(tv_true.shape[1]), tv_true.shape[0])
    lv_values   = lv_true.flatten()
    isq_values  = isq_true.flatten()
    iv_values   = iv_true.flatten()
    tv_values   = tv_true.flatten()

    # Create a Pandas DataFrame
    df_true = pd.DataFrame({
        'y': row_indices,
        'x': col_indices,
        'lv_true':  lv_values,
        'isq_true': isq_values,
        'iv_true':  iv_values,
        'tv_true':  tv_values
    })

    df_true['tv_observed'] = np.random.normal(df_true['tv_true'], abs(df_true['tv_true'] * tv_obs_noise), size=len(df_true))
    df_true['tv_observed'] = np.clip(df_true['tv_observed'], 0, None)
    return df_true

def calculate_spatial_autocorr(df_true):
    gdf_true = gpd.GeoDataFrame(df_true, geometry=gpd.points_from_xy(df_true.x, df_true.y))
    w = Queen.from_dataframe(gdf_true)
    w.transform = 'r'
    morans = []
    for var in ['lv_true', 'iv_true', 'tv_true']:
        moran = esda.Moran(gdf_true[var], w)
        morans.append(moran.I)
    morans = pd.Series(morans, index=['lv_true','iv_true','tv_true'])
    return morans


def fit_data(df_true, train_n):

    train_idx = np.logical_and(df_true['x'] % train_n == 0, df_true['y'] % train_n == 0).squeeze().values

    df_train = df_true[train_idx]
    df_test  = df_true[~train_idx]

    df_train_diff = df_train[['isq_true', 'tv_observed']].diff().dropna()

    lr = LinearRegression(fit_intercept=False)
    lr.fit(y=df_train_diff['tv_observed'],
           X=df_train_diff['isq_true'].values.reshape(-1,1))

    iv_pred  = lr.predict(df_train['isq_true'].values.reshape(-1,1))
    df_train = df_train.assign(iv_pred=iv_pred,
                               lv_pred=np.clip(df_train['tv_observed'] - iv_pred, 0, None))

    # knn regression
    zout_knn = fit_knn(df_train, df_test)

    # kernel_regression
    zout_kr, _ = KernelReg(endog=df_train['lv_pred'],
                           exog=[df_train['x'], df_train['y']],
                           var_type='cc').fit([df_test['x'], df_test['y']])

    # 2d LOESS Regression

    zoutdeg1, _ = loess_2d(x=df_train['x'].values, y=df_train['y'].values, z=df_train['lv_pred'].values,
                            xnew=df_test['x'].values, ynew=df_test['y'].values,
                            degree=1, frac=0.5, npoints=None, rescale=False, sigz=None)
    zoutdeg2, _ = loess_2d(x=df_train['x'].values, y=df_train['y'].values, z=df_train['lv_pred'].values,
                            xnew=df_test['x'].values, ynew=df_test['y'].values,
                            degree=2, frac=0.5, npoints=None, rescale=False, sigz=None)

    # kriging (gaussian process regression?)
    zout_krig = fit_kriging(df_train, df_test)

    # Extrapolate improved and total land values, put into df_test
    iv_pred = lr.predict(df_test['isq_true'].values.reshape(-1,1))

    df_test = df_test.assign(iv_pred          =iv_pred,
                             lv_pred_knn      =zout_knn,
                             lv_pred_loessdeg1=zoutdeg1,
                             lv_pred_loessdeg2=zoutdeg2,
                             lv_pred_kr       =zout_kr,
                             lv_pred_krig     =zout_krig,
                             tv_pred_knn      =iv_pred + zout_knn,
                             tv_pred_loessdeg1=iv_pred + zoutdeg1,
                             tv_pred_loessdeg2=iv_pred + zoutdeg2,
                             tv_pred_kr       =iv_pred + zout_kr,
                             tv_pred_krig     =iv_pred + zout_krig,
                             )
    return df_test

def fit_knn(df_train, df_test):
    X_train = df_train[['x', 'y']]
    y_train = df_train['lv_pred']

    # Create KNN regressor and define parameter grid for k
    param_grid = {'n_neighbors': range(1, 100)}
    knn = KNeighborsRegressor()

    # Use GridSearchCV for cross-validation
    grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)

    # Get the best k value
    best_k = grid_search.best_params_['n_neighbors']

    # Fit the KNN model with the best k
    best_knn = KNeighborsRegressor(n_neighbors=best_k)
    best_knn.fit(X_train, y_train)
    
    zout_knn = best_knn.predict(df_test[['x', 'y']])
    return zout_knn

def fit_kriging(df_train, df_test):
    
    def fit_kriging_and_evaluate(x_tr, y_tr, z_tr, x_te, y_te, z_te, variogram_model):
        # Create the Kriging model
        OK = OrdinaryKriging(
            x_tr, y_tr, z_tr,
            variogram_model=variogram_model,
            verbose=False,
            enable_plotting=False
        )

        # Perform kriging on test points
        zout_krig, _ = OK.execute('points', x_te, y_te)

        # Evaluate the prediction using mean squared error
        mse = mean_squared_error(z_te, zout_krig)

        return mse, zout_krig

    # Split the training data into a smaller training subset and a validation set
    train_sub_idx, val_idx = train_test_split(np.arange(len(df_train)), test_size=0.2)
    df_train_sub = df_train.iloc[train_sub_idx]
    df_val       = df_train.iloc[val_idx]

    # Extract x, y, and z values for training and validation
    x_train_sub = df_train_sub['x'].values.astype(float)
    y_train_sub = df_train_sub['y'].values.astype(float)
    z_train_sub = df_train_sub['lv_pred'].values.astype(float)

    x_val = df_val['x'].values.astype(float)
    y_val = df_val['y'].values.astype(float)
    z_val = df_val['lv_pred'].values.astype(float)

    # List of variogram models to test
    variogram_models = ['linear', 'power', 'gaussian', 'spherical', 'exponential']

    # Dictionary to store MSE for each model
    mse_dict  = {}
    # Fit and evaluate each model
    for model in variogram_models:
        mse, _ = fit_kriging_and_evaluate(x_train_sub, y_train_sub, z_train_sub, x_val, y_val, z_val, model)
        mse_dict[model] = mse
        print(f'Variogram Model: {model}, MSE: {mse}')

    # Find the best model based on the lowest MSE
    best_model = min(mse_dict, key=mse_dict.get)
    print(f'Best Variogram Model: {best_model} with MSE: {mse_dict[best_model]}')

    x_train = df_train['x'].values.astype(float)
    y_train = df_train['y'].values.astype(float)
    z_train = df_train['lv_pred'].values.astype(float)

    # Test data
    x_test = np.array(df_test['x']).astype(float)
    y_test = np.array(df_test['y']).astype(float)
    z_test = np.array(df_test['lv_true']).astype(float)

    # If desired, you can save the predictions using the best model
    _, zout_krig = fit_kriging_and_evaluate(x_train, y_train, z_train, x_test, y_test, z_test, best_model)
    
    return zout_krig

def calculate_COD(assessed_values, sale_prices):
    """
    https://github.com/PhilaController/OPA-analysis/blob/master/sales_ratio_study.ipynb
    Compute the coefficient of dispersion, which gives the mean
    absolute deviation (as a percent) from the median ratio.

    Here, ratio refers to the ratio of assessed value to sale price.

    Parameters
    ----------
    assessed_values : array_like
        the array of assessed values
    sale_prices : array_like
        the array of sale prices
    """
    ratios = assessed_values / sale_prices
    median = np.median(ratios)
    return np.mean(abs(ratios-median)) / median * 100

def calculate_PRD(assessed_values, sale_prices):
    """
    Compute the price-related differential, equal to the mean ratio divided by
    the mean ratio weighted by sale price.

    Here, ratio refers to the ratio of assessed value to sale price.

    Parameters
    ----------
    assessed_values : array_like
        the array of assessed values
    sale_prices : array_like
        the array of sale prices
    """
    ratios = assessed_values / sale_prices
    weighted_mean = np.average(ratios, weights=sale_prices)
    return np.mean(ratios) / weighted_mean

def evaluate_model(df_test):

    true_pred_pairs = [('iv_true', 'iv_pred'),
                       ('lv_true', 'lv_pred_knn'),
                       ('lv_true', 'lv_pred_loessdeg1'),
                       ('lv_true', 'lv_pred_loessdeg2'),
                       ('lv_true', 'lv_pred_kr'),
                       ('lv_true', 'lv_pred_krig'),
                       ('tv_true', 'tv_pred_knn'),
                       ('tv_true', 'tv_pred_loessdeg1'),
                       ('tv_true', 'tv_pred_loessdeg2'),
                       ('tv_true', 'tv_pred_kr'),
                       ('tv_true', 'tv_pred_krig'),
                       ]

    results = []
    for true_name, pred_name in true_pred_pairs:
      true = df_test[true_name]
      pred = df_test[pred_name]

      r2 = r2_score(y_true=true, y_pred=pred)

      median_ratio = np.median(pred/true)
      COD = calculate_COD(assessed_values=pred, sale_prices=true)
      PRD = calculate_PRD(assessed_values=pred, sale_prices=true)
      mape = np.mean(np.abs((true - pred) / true)) * 100
      
      results.append([r2, mape, median_ratio, COD, PRD])

    results_df = pd.DataFrame(results,
                              index=['Improved Values', 
                                     'Land Values (kNN)', 
                                     'Land Values (LoessDeg1)', 
                                     'Land Values (LoessDeg2)', 
                                     'Land Values (KernelReg)', 
                                     'Land Values (Kriging)', 
                                     'Total Values (kNN)', 
                                     'Total Values (LoessDeg1)', 
                                     'Total Values (LoessDeg2)',
                                     'Total Values (KernelReg)', 
                                     'Total Values (Kriging)'], 
                              columns=['R-squared', 'MAPE', 'Median Ratio', 'Coefficient of Disperion', 'Price-related Differential',
                                       ])
    return results_df
 
if __name__ == "__main__":
     
    grid_size          = 100   # number of properties in the x and y dimension, grid_size = 100 makes 100 by 100 grid
    true_price_per_isq = 4     # "dollars" per improved square foot
    isq_from_lv        = True  # whether improved square footage should be generated from land values, or generated independently (with same process as land values)
    
    lv_center        = 100000
    lv_decay         = .03
    lv_noise_sd      = 1       # should really be called lv_noise_factor but keeping name for legacy reasons
    isq_noise_sd     = 5e-1    # should really be called isq_noise_factor but keeping name for legacy reasons
    iv_noise_factor  = .3
    iv_noise_sd      = None    # just a placeholder
    tv_obs_noise     = 5e-2
    train_n          = 3
    
    df_true = make_data(grid_size, true_price_per_isq, lv_center, lv_decay, lv_noise_sd,
                        isq_noise_sd, iv_noise_factor, iv_noise_sd, tv_obs_noise, isq_from_lv)
    morans = calculate_spatial_autocorr(df_true)
    print(morans)    

    df_test = fit_data(df_true, train_n)
    results_df = evaluate_model(df_test)
    print(results_df)

