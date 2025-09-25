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

np.random.seed(2018)

# --- General functions ---
def make_data(n):
    '''
    Make a dataset with a given number (n) datapoints
    of the Runge function.
    '''
    x = np.linspace(-1, 1, n)
    y = 1/(1 + 25*x**2) + np.random.normal(0, 1)

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


# --- Part h) ---