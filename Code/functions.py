'''
Functions for project 1, FYS-STK4155

Authors:
- Kjersti Stangeland
- Ingvild Olden Bjerklund
- Jenny Guldvog
- Sverre Johansen

September 2025
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Lasso

np.random.seed(2018)

# --- General functions ---
#Definere runge
def f_true(x):   
    return 1.0 / (1.0 + 25.0 * x**2)  # Runge-lignende funksjon

def make_data(n, seed=seed):        
    x = np.linspace(-1, 1, n)   

    y_clean = f_true(x)    
    y = y_clean + np.random.normal(0, 1)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=seed, shuffle=True
    )

    train = (x_train, y_train)
    test  = (x_test, y_test)
    full  = (x, y, y_clean)
    return train, test, full

"""
How to use:
(train, test, full) = make_dataset(n_points)
x_train, y_train = train
x_test, y_test = test
x_all, y_all, y_all_clean = full 
"""

def polynomial_features(x, p, intercept=True):
    '''
    Generate a design matrix X.

    Parameters:
        x: dataset
        p: number of features
        intercept: adds a column of intercept if set to True

    Returns:
        X: design matrix with dimensions
             (n, p) if intercept=False or (n, p+1) if intercept=True
    '''
    n = len(x)

    if intercept:
        X = np.zeros((n, p + 1))
        X[:, 0] = 1  # adds intercept
        for i in range(1, p + 1):
            X[:, i] = x ** i
    else:
        X = np.zeros((n, p))
        for i in range(0, p):
            X[:, i] = x ** (i + 1)

    return X 

def standardize(X, y):
    '''
    Scale and standardize a design matrix X and data y.

    Parameters:
        X: np.ndarray
        y: np.ndarray
    
    Returns:
        X_norm: np.ndarray
        y_centered: np.ndarray
        
    '''
    X_mean = X.mean(axis=0) # The mean of each column/feature
    X_std = X.std(axis=0)
    X_std[X_std == 0] = 1  # safeguard to avoid division by zero for constant features
    X_norm = (X - X_mean) / X_std

    # Center the target to zero mean
    y_mean = y.mean()
    y_centered = y - y_mean

    return X_norm, y_centered

def MSE(y_data, y_pred):
    ''' 
    Mean square error
    '''
    mse = np.mean((y_data - y_pred)**2)

    return mse

def R2(y_data, y_pred):
    '''
    R^2 score
    '''
    numerator = np.sum((y_data - y_pred)**2)
    denumerator = np.sum((y_data - np.mean(y_data))**2)

    if denumerator == 0:
        r2 = np.nan
    else:
        r2 = 1 - numerator/denumerator
    
    return r2
                   

# --- Part a) ---
def OLS_parameters(X, y):
    ''' 
    Find the OLS parameters
    '''
    return np.linalg.pinv(X) @ y


# --- Part b) ---
def Ridge_parameters(X, y, lamb=0.01):
    '''
    Doc string kommer...
    '''
    # Assumes X is scaled and has no intercept column
    I = np.eye(np.shape(X.T @ X)[0])
    
    return np.linalg.inv(X.T @ X + lamb*I) @ X.T @ y


# --- Part c) ---

def ols_gradient(X, y, beta):
    return (2/len(y)) * (X.T @ (X @ beta - y))

def ridge_gradient(X, y, beta, lambda_):
    return (2/len(y)) * (X.T @ (X @ beta - y)) + 2 * lambda_ * beta

def gradient_descent_ols(X, y, learning_rate=0.01, n_iterations=1000, tol=1e-6, use_tol=False):
    n_samples, n_features = X.shape
    theta = np.zeros(n_features)
    cost_history = []
    for i in range(n_iterations):
        gradient = ols_gradient(X, y, theta)
        theta -= learning_rate * gradient
        # cost is the OLS cost function
        cost = (1/n_samples) * np.sum((X @ theta - y)**2)
        cost_history.append(cost)
        if use_tol and i > 0 and abs(cost_history[-2] - cost) < tol:
            print(f"Converged after {i} iterations.")
            break
    return theta, cost_history

def gradient_descent_ridge(X, y, alpha, learning_rate=0.01, n_iterations=1000, tol=1e-6, use_tol=False):
    n_samples, n_features = X.shape
    theta = np.zeros(n_features)
    cost_history = []
    for i in range(n_iterations):
        gradient = ridge_gradient(X, y, theta, alpha)
        theta -= learning_rate * gradient
        # cost is the Ridge cost function, including the regularization term
        cost = (1/n_samples) * np.sum((X @ theta - y)**2) + alpha * np.sum(theta**2)
        cost_history.append(cost)
        if use_tol and i > 0 and abs(cost_history[-2] - cost) < tol:
            print(f"Converged after {i} iterations.")
            break
    return theta, cost_history

# --- Part d) ---

def gradient_descent_advanced(X, y, method='gd', lr_method='ols', learning_rate=0.01, n_iterations=1000, tol=1e-6, use_tol=False, beta=0.9, epsilon=1e-8, lambda_=0.01):
    n_samples, n_features = X.shape
    theta = np.zeros(n_features)
    cost_history = []   
    m = np.zeros(n_features)  # For momentum and Adam
    v = np.zeros(n_features)  # For Adam 
    for i in range(n_iterations):
        if lr_method == 'ols':
            gradient = ols_gradient(X, y, theta)
        elif lr_method == 'ridge':
            gradient = ridge_gradient(X, y, theta, lambda_=lambda_)
        else:
            raise ValueError("Unknown linear regression method")
        if method == 'momentum':
            m = beta * m + (1 - beta) * gradient
            gradient = m
        elif method == 'adagrad':
            v += gradient**2
            adjusted_lr = learning_rate / (np.sqrt(v) + epsilon)
            gradient = adjusted_lr * gradient
        elif method == 'rmsprop':
            v = beta * v + (1 - beta) * gradient**2
            adjusted_lr = learning_rate / (np.sqrt(v) + epsilon)
            gradient = adjusted_lr * gradient
        elif method == 'adam':
            m = beta * m + (1 - beta) * gradient
            v = beta * v + (1 - beta) * (gradient**2)
            m_hat = m / (1 - beta**(i+1))
            v_hat = v / (1 - beta**(i+1))
            adjusted_lr = learning_rate / (np.sqrt(v_hat) + epsilon)
            gradient = adjusted_lr * m_hat
        elif method == 'gd':
            # For GD, do nothing
            pass
        else:
            raise ValueError("Unknown optimization method")
        theta -= learning_rate * gradient
        if lr_method == 'ols':
            cost = (1/n_samples) * np.sum((X @ theta - y)**2)
        elif lr_method == 'ridge':
            cost = (1/n_samples) * np.sum((X @ theta - y)**2) + lambda_ * np.sum(theta**2)
        cost_history.append(cost)
        if use_tol and i > 0 and abs(cost_history[-2] - cost) < tol:
            print(f"{method} converged after {i} iterations.")
            break
    return theta, cost_history

# --- Part e) ---


# --- Part f) ---


# --- Part g) ---

def ols(x_train, y_train, x_eval, degree):    
    model = make_pipeline(
        PolynomialFeatures(degree=degree, include_bias=True), 
        LinearRegression(fit_intercept=False))    

    model.fit(x_train.reshape(-1, 1), y_train)  
    y_predicted = model.predict(x_eval.reshape(-1, 1)).ravel()    

    return y_predicted, model

def bootstrap(degrees, x_train, x_test, y_train, y_test, x_grid, boots_reps, seed=seed):

    rng = np.random.default_rng(seed)
    y_true = f_true(x_grid)

    boot_results = {
        "degree": [], 
        "bias2_boots": [],
        "var_boots": [], 
        "mse_boots": [], 
        "test_mse": [], 
        "train_mse": []}

    for d in degrees:
        #Direct-fit (regular OLS without bootstrapping) on train/test MSE
        y_predicted_train, model = ols(x_train, y_train, x_train, degree=d)
        y_predicted_test = model.predict(x_test.reshape(-1,1)).ravel()

        train_mse = mse(y_train, y_predicted_train)
        test_mse = mse(y_test, y_predicted_test)

        boots_predicted = np.empty((boots_reps, x_grid.size), dtype=np.float64)

        #Bootstrap, and OLS on the bootstrap values
        for b in range(boots_reps):
            idx_b = rng.choice(x_train.size, size=x_train.size, replace=True)
            xb, yb = x_train[idx_b], y_train[idx_b]
            y_pred_boot, _ = ols(xb, yb, x_grid, degree=d)
            boots_predicted[b] = y_pred_boot

        mean_boots = boots_predicted.mean(axis=0)
        var_boots = boots_predicted.var(axis=0, ddof=1)
        bias2_boots = (mean_boots - y_true)**2
        mse_boots = ((boots_predicted - y_true[None,:])**2).mean(axis=0)

        boot_results["degree"].append(d)
        boot_results["bias2_boots"].append(bias2_boots.mean())
        boot_results["var_boots"].append(var_boots.mean())
        boot_results["mse_boots"].append(mse_boots.mean())
        boot_results["train_mse"].append(train_mse)
        boot_results["test_mse"].append(test_mse)

    for k in list(boot_results.keys()):
        boot_results[k] = np.array(boot_results[k])

    return boot_results


# --- Part h) ---

#K-fold cross-validation for OLS

def kfold_cv_mse_ols(degree, k, x, y, seed=seed):
    kf = KFold(n_splits=k, shuffle=True, random_state=seed)
    kfold_mses = []
    for train_idx, val_idx in kf.split(x):
        x_train_cv, x_val = x[train_idx], x[val_idx]
        y_train_cv, y_val = y[train_idx], y[val_idx]

        predicted_val, _ = ols(x_train_cv, y_train_cv, x_val, degree)
        kfold_mses.append(mse(y_val, predicted_val))

    return np.mean(kfold_mses)

#K-fold cross-validation for ridge, lasso

    def cv_for_methods(method, degree, lambdas, k, x, y, seed=seed):
    x = np.asarray(x); y = np.asarray(y)

    kf = KFold(n_splits=k, shuffle=True, random_state=seed)
    cv_means = []

    for lam in lambdas:
        kfold_mses = []
        for train_idx, val_idx in kf.split(x):
            x_train_cv, x_val = x[train_idx], x[val_idx]
            y_train_cv, y_val = y[train_idx], y[val_idx]

            if method == 'ridge':
                base = Ridge(alpha=lam, fit_intercept=True, 
                max_iter=300_000, 
                tol=1e-3)
            elif method == "lasso":
                base = Lasso(alpha=lam, 
                fit_intercept=True, 
                max_iter=2_000_000, 
                tol=5e-3, 
                selection="cyclic")
            else:
                raise ValueError("method must be 'ridge' or 'lasso'")
            
            model = make_pipeline(PolynomialFeatures(degree, include_bias=False), 
            StandardScaler(),
            base)

            model.fit(x_train_cv.reshape(-1,1), y_train_cv)
            pred_vals = model.predict(x_val.reshape(-1,1)).ravel()
            kfold_mses.append(mse(y_val, pred_vals))

        cv_means.append(np.mean(kfold_mses))

    cv_means = np.array(cv_means)
    best_idx = int(np.argmin(cv_means))

    return {
        "lambdas": np.array(lambdas, dtype=float),
        "cv_mse": cv_means,
        "best_lambda": float(lambdas[best_idx]),
        "best_mse": float(cv_means[best_idx])
    }
