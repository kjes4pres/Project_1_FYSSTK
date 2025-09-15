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
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# --- General functions ---
def make_data(n):
    '''
    Make a dataset with a given number (n) datapoints
    of the Runge function.
    '''
    x = np.linspace(-1, 1, n)
    y = 1/(1 + 25*x**2) + np.random.normal(0, 0.1)

    return x, y

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
        X[:, 0] = 1
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

def split_n_train(X, y, size):
    '''
    ...
    '''
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=size)

    return X_train, X_test, y_train, y_test

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


# --- Part b) ---


# --- Part c) ---


# --- Part d) ---


# --- Part e) ---


# --- Part f) ---


# --- Part g) ---


# --- Part h) ---