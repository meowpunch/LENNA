<<<<<<< HEAD:NASR/model/train_mixed.py
# ----------- math tools
import numpy as np # linear algebra
import matplotlib.pyplot as plt
import pandas as pd
from joblib import load

# ----------- torch things
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision

# ----------- sklearn things
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
import xgboost as xgb
import lightgbm as lgb
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import GridSearchCV

# ----------- Validation function
def inverse_latency(X):
    robust, quantile = load("robust.pkl"), load("quantile.pkl")
    if isinstance(X, pd.Series):
        X = X.values.reshape(-1, 1)
    else:
        X = X.reshape(-1, 1)
    return robust.inverse_transform(quantile.inverse_transform(X)).reshape(-1)

# Kfold -> cross_val_score에서 inverse latency 처리해줘야 하나???
n_folds = 10
def rmsle_cv(model, X, Y):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits()
    # 전처리도 여기에 parameter로 넣으면 되고 fit, predict 모두 처리하는 게 아래 줄.
    rmsle = np.sqrt(-cross_val_score(model, X, Y, scoring="neg_mean_squared_error", cv = kf))
    return rmsle

def rmse(model, x_test, y_test):
    y_pred = model.predict(x_test)
    y_test, y_pred = inverse_latency(y_test), inverse_latency(y_pred)
    return np.sqrt(mean_squared_error(y_test, y_pred))

# ------------ get data
from data_pipeline.data_preprocessor import PreProcessor
PP = PreProcessor()
X_train, Y_train, X_test, Y_test = PP.process()
X, Y = PP.split_xy(PP.preprocess())
# x_train = Variable(X_train)
# Y_train = Variable(Y_train)

# ----------- regressions
lasso = Lasso(alpha =0.0005, random_state=3)
ENet = ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3)
KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5) #kernel = 'rbf' , 'sigmoid'
lasso.fit(X_train, Y_train)
ENet.fit(X_train, Y_train)
KRR.fit(X_train, Y_train)

# ---------- Gradient Boosting regression
GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10,
                                   loss='huber', random_state =5)

# --------- LGB Regressor
model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,
                              learning_rate=0.05, n_estimators=720,
                              max_bin = 55, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)


# ------------ XGBoost with GridSearch
# A parameter grid for XGBoost
params = {'min_child_weight':[i/10.0 for i in range(5,18)], 'gamma':[i/100.0 for i in range(3,6)],
'subsample':[i/10.0 for i in range(4,9)], 'colsample_bytree':[i/10.0 for i in range(4,8)], 'max_depth': [2,3,4]}
xgb = XGBRegressor(nthread=-1)
grid_xgb = GridSearchCV(xgb, params)
grid_xgb.fit(X_train, Y_train)
xgb = grid_xgb.best_estimator_

# ------------ SVR with GridSearch
params = {'gamma' :[i/100.0 for i in range(0,11)],
          'coef0':[0, 0.1, 0.5, 1], 'C' :[0.1, 0.2, 0.5, 1], 'epsilon':[i/10.0 for i in range(0,6)]}
model_svr = SVR()
grid_svr = GridSearchCV(model_svr, params, cv=10, scoring='neg_mean_squared_error')
grid_svr.fit(X_train, Y_train)
model_svr = grid_svr.best_estimator_

# ------------ Random Forest Regressor with GridSearch
param_grid = [
    {'n_estimators': [3, 10, 30, 60, 90], 'max_features': [50,100,150,200,250,300]},
    {'bootstrap': [True], 'n_estimators': [3, 10, 30, 60, 90], 'max_features': [50,100,150,200,250]},
]
forest_reg = RandomForestRegressor()
grid_rf = GridSearchCV(forest_reg, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_rf.fit(X_train, Y_train)
forest_reg = grid_rf.best_estimator_

# ------- scores
# score = rmsle_cv(lasso)
# print("\nLasso score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
# score = rmsle_cv(ENet)
# print("ElasticNet score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
# score = rmsle_cv(KRR)
# print("Kernel Ridge score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
# score = rmsle_cv(GBoost)
# print("Gradient Boosting score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
# score = rmsle_cv(model_lgb)
# print("LGBM score: {:.4f} ({:.4f})\n" .format(score.mean(), score.std()))
# score = rmsle_cv(xgb)
# print("Xgboost score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
# score = rmsle_cv(model_svr)
# print("SVR score: {:.4f} ({:.4f})\n" .format(score.mean(), score.std()))
# score = rmsle_cv(forest_reg)
# print("Random Forest score: {:.4f} ({:.4f})\n" .format(score.mean(), score.std()))

print("Lasso rmse: {}".format(rmse(lasso, X_test, Y_test)))
print("ElasticNet rmse: {}".format(rmse(ENet, X_test, Y_test)))
print("Kernel Ridge rmse: {}".format(rmse(KRR, X_test, Y_test)))
print("Gradient Boosting rmse: {}".format(rmse(GBoost, X_test, Y_test)))
print("LGBM rmse : {}".format(rmse(model_lgb, X_test, Y_test)))
print("Xgboost rmse : {}".format(xgb, X_test, Y_test))
print("SVR rmse : {}".format(model_svr, X_test, Y_test))
print("Random Forest rmse : {}".format(forest_reg, X_test, Y_test))

# 여긴 뭐하는건가 싶네
# # XGBoost
# xgb.fit(X_train, Y_train)
# xgb_train_pred = xgb.predict(X_train)
# xgb_pred = np.expm1(xgb.predict(X_test))
# print(rmse(Y_test, xgb_train_pred)) <- 이건 내가 바꾼 거임
# 
# # LightGBM
# model_lgb.fit(X_train, Y_train)
# lgb_train_pred = model_lgb.predict(X_train)
# lgb_pred = np.expm1(model_lgb.predict(X_test))
# print(rmse(Y_test, lgb_train_pred))
# 
# # Gradient Boost
# GBoost.fit(X_train,Y_train)
# GB_train_pred = GBoost.predict(X_train)
# GB_pred = np.expm1(GBoost.predict(X_test))
# print(rmse(Y_test, GB_train_pred))

# ensemble
ensemble = xgb.predict(X_test)*0.25 + lgb.predict(X_test)*0.25 + GBoost.predict(X_test)*0.5

=======
# ----------- math tools
import numpy as np # linear algebra
import matplotlib.pyplot as plt
import pandas as pd
from joblib import load

# ----------- torch things
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision

# ----------- sklearn things
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
import xgboost as xgb
import lightgbm as lgb
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import GridSearchCV

# ----------- Validation function
def inverse_latency(X):
    robust, quantile = load("robust.pkl"), load("quantile.pkl")
    if isinstance(X, pd.Series):
        X = X.values.reshape(-1, 1)
    else:
        X = X.reshape(-1, 1)
    return robust.inverse_transform(quantile.inverse_transform(X)).reshape(-1)

# Kfold -> cross_val_score에서 inverse latency 처리해줘야 하나???
n_folds = 10
def rmsle_cv(model, X, Y):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits()
    # 전처리도 여기에 parameter로 넣으면 되고 fit, predict 모두 처리하는 게 아래 줄.
    rmsle = np.sqrt(-cross_val_score(model, X, Y, scoring="neg_mean_squared_error", cv = kf))
    return rmsle

def rmse(model, x_test, y_test):
    y_pred = model.predict(x_test)
    y_test, y_pred = inverse_latency(y_test), inverse_latency(y_pred)
    return np.sqrt(mean_squared_error(y_test, y_pred))

# ------------ get data
from data_pipeline.data_preprocessor import PreProcessor
PP = PreProcessor()
X_train, Y_train, X_test, Y_test = PP.process()
X, Y = PP.split_xy(PP.preprocess())
# x_train = Variable(X_train)
# Y_train = Variable(Y_train)

# ----------- regressions
lasso = Lasso(alpha =0.0005, random_state=3)
ENet = ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3)
KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5) #kernel = 'rbf' , 'sigmoid'
lasso.fit(X_train, Y_train)
ENet.fit(X_train, Y_train)
KRR.fit(X_train, Y_train)

# ---------- Gradient Boosting regression
GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10,
                                   loss='huber', random_state =5)

# --------- LGB Regressor
model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,
                              learning_rate=0.05, n_estimators=720,
                              max_bin = 55, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)


# ------------ XGBoost with GridSearch
# A parameter grid for XGBoost
params = {'min_child_weight':[i/10.0 for i in range(5,18)], 'gamma':[i/100.0 for i in range(3,6)],
'subsample':[i/10.0 for i in range(4,9)], 'colsample_bytree':[i/10.0 for i in range(4,8)], 'max_depth': [2,3,4]}
xgb = XGBRegressor(nthread=-1)
grid_xgb = GridSearchCV(xgb, params)
grid_xgb.fit(X_train, Y_train)
xgb = grid_xgb.best_estimator_

# ------------ SVR with GridSearch
params = {'gamma' :[i/100.0 for i in range(0,11)],
          'coef0':[0, 0.1, 0.5, 1], 'C' :[0.1, 0.2, 0.5, 1], 'epsilon':[i/10.0 for i in range(0,6)]}
model_svr = SVR()
grid_svr = GridSearchCV(model_svr, params, cv=10, scoring='neg_mean_squared_error')
grid_svr.fit(X_train, Y_train)
model_svr = grid_svr.best_estimator_

# ------------ Random Forest Regressor with GridSearch
param_grid = [
    {'n_estimators': [3, 10, 30, 60, 90], 'max_features': [50,100,150,200,250,300]},
    {'bootstrap': [True], 'n_estimators': [3, 10, 30, 60, 90], 'max_features': [50,100,150,200,250]},
]
forest_reg = RandomForestRegressor()
grid_rf = GridSearchCV(forest_reg, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_rf.fit(X_train, Y_train)
forest_reg = grid_rf.best_estimator_

# ------- scores
# score = rmsle_cv(lasso)
# print("\nLasso score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
# score = rmsle_cv(ENet)
# print("ElasticNet score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
# score = rmsle_cv(KRR)
# print("Kernel Ridge score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
# score = rmsle_cv(GBoost)
# print("Gradient Boosting score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
# score = rmsle_cv(model_lgb)
# print("LGBM score: {:.4f} ({:.4f})\n" .format(score.mean(), score.std()))
# score = rmsle_cv(xgb)
# print("Xgboost score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
# score = rmsle_cv(model_svr)
# print("SVR score: {:.4f} ({:.4f})\n" .format(score.mean(), score.std()))
# score = rmsle_cv(forest_reg)
# print("Random Forest score: {:.4f} ({:.4f})\n" .format(score.mean(), score.std()))

print("Lasso rmse: {}".format(rmse(lasso, X_test, Y_test)))
print("ElasticNet rmse: {}".format(rmse(ENet, X_test, Y_test)))
print("Kernel Ridge rmse: {}".format(rmse(KRR, X_test, Y_test)))
print("Gradient Boosting rmse: {}".format(rmse(GBoost, X_test, Y_test)))
print("LGBM rmse : {}".format(rmse(model_lgb, X_test, Y_test)))
print("Xgboost rmse : {}".format(xgb, X_test, Y_test))
print("SVR rmse : {}".format(model_svr, X_test, Y_test))
print("Random Forest rmse : {}".format(forest_reg, X_test, Y_test))

# 여긴 뭐하는건가 싶네
# # XGBoost
# xgb.fit(X_train, Y_train)
# xgb_train_pred = xgb.predict(X_train)
# xgb_pred = np.expm1(xgb.predict(X_test))
# print(rmse(Y_test, xgb_train_pred)) <- 이건 내가 바꾼 거임
# 
# # LightGBM
# model_lgb.fit(X_train, Y_train)
# lgb_train_pred = model_lgb.predict(X_train)
# lgb_pred = np.expm1(model_lgb.predict(X_test))
# print(rmse(Y_test, lgb_train_pred))
# 
# # Gradient Boost
# GBoost.fit(X_train,Y_train)
# GB_train_pred = GBoost.predict(X_train)
# GB_pred = np.expm1(GBoost.predict(X_test))
# print(rmse(Y_test, GB_train_pred))

# ensemble
ensemble = xgb.predict(X_test)*0.25 + lgb.predict(X_test)*0.25 + GBoost.predict(X_test)*0.5
>>>>>>> d50f9753dc48a23bdbfd960c25c7eb23fe3c6040:NASR/train_mixed.py
