"""
Functions for project 1, FYS-STK4155

Authors:
- Kjersti Stangeland
- Ingvild Olden Bjerkelund
- Jenny Guldvog
- Sverre Johansen

September 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split, KFold 
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error as mse
import matplotlib.style as mplstyle

mplstyle.use(["ggplot", "fast"])


# For reproducibility
np.random.seed(2018)
seed = np.random.seed(2018)

# --- General functions ---

def f_true(x):
    """
    Return 1D Runge function
    """
    return 1.0 / (1.0 + 25.0 * x**2)


def make_data(n, seed=seed):
    """
    Makes a data set of length n over the Runge function
    for x in (-1, 1). Includes stochastic noise.

    Creates train and test data sets
    """
    x = np.linspace(-1, 1, n)
    x = x.reshape(-1, 1)

    scaler = StandardScaler(with_std=False)
    scaler.fit(x)
    x_s = scaler.transform(x)

    y_clean = f_true(x_s)
    y = y_clean + np.random.normal(0, 0.1, n)

    x_train, x_test, y_train, y_test = train_test_split(
        x_s, y, test_size=0.2, random_state=seed, shuffle=True
    )

    train = (x_train, y_train)
    test = (x_test, y_test)
    full = (x_s, y, y_clean)
    return train, test, full


def MSE(y_data, y_pred):
    """
    Mean square error
    """
    n = np.size(y_pred)
    mse = np.mean((y_data - y_pred) ** 2)/n

    return mse


def R2(y_data, y_pred):
    """
    R^2 score
    """
    numerator = np.sum((y_data - y_pred) ** 2)
    denumerator = np.sum((y_data - np.mean(y_data)) ** 2)

    if denumerator == 0:
        r2 = np.nan
    else:
        r2 = 1 - numerator / denumerator

    return r2


# --- Part a) ---

def make_clean_data(n, seed=seed):
    """
    Makes a data set of length n over the Runge function
    for x in (-1, 1), with no stochastic noise.

    Creates train and test data sets
    """
    x = np.linspace(-1, 1, n)

    x = x.reshape(-1, 1)

    scaler = StandardScaler(with_std=False)
    scaler.fit(x)
    x_s = scaler.transform(x)

    y = f_true(x_s)
    
    x_train, x_test, y_train, y_test = train_test_split(
        x_s, y, test_size=0.2, random_state=seed, shuffle=True
    )

    train = (x_train, y_train)
    test = (x_test, y_test)
    full = (x_s, y)
    return train, test, full


def polynomial_features(x, p, intercept=True):
    """
    Creates a design matrix X with p features and n samples.
    Option to add an intercept (column of ones).

    Inputs:
        x (np.ndarray): Input data of shape (n, 1), n is the number of samples.
        p (int): Number of features.
        intercept (boolean): Optional, if True adds a column of ones.

    Returns:
        X (np.ndarray): Design matrix of shape (n, p+1) if 'intercept=True',
        otherwise shape (n, p).
    """
    n = len(x[:, 0])

    if intercept:
        X = np.zeros((n, p + 1))
        X[:, 0] = 1
        for i in range(1, p + 1):
            X[:, i] = x[:, 0] ** i
    else:
        X = np.zeros((n, p))
        for i in range(0, p):
            X[:, i] = x[:, 0] ** (i + 1)

    return X


def OLS_parameters(X, y):
    """
    Computes the optimal parameters for 
    Ordinary Least Squares (OLS).

    Inputs:
        X (np.ndarray): Design matrix of shape (n, p), where `n` is the
        number of samples and `p` is the number of features.
        y (np.ndarray): Target vector of shape (n).

    Returns:
        (np.ndarray): Optimal parameters vector of shape (p)
    """
    return np.linalg.pinv(X) @ y

def OLS_various_poly_deg(n, p_vals, noise=True):
    """
    Performs Ordinary Least Square analysis on
    the Runge function for a given number of samples
    and a range of polynomial degrees (features).

    Inputs:
        n (int): Number of samples
        p (np.ndarray): List of polynomial degrees.

    Returns:
        df_OLS (pd.DataFrame): Dataframe containing all results.
    """
    results = []
    # Making the data and splitting into test/train
    if noise:
        train, test, full = make_data(n)  # making a dataset with size n
        x_train, y_train = train  # training data
        x_test, y_test = test  # test data
        x_all, y_all, y_all_clean = full  # full data

    else:
        train, test, full = make_clean_data(n) # dataset without noise
        x_train, y_train = train  # training data
        x_test, y_test = test  # test data
        x_all, y_all = full  # full data

    x_train = x_train.reshape(-1, 1)
    x_test = x_test.reshape(-1, 1)
    x_all = x_all.reshape(-1, 1)

    for p in p_vals:
        # Making a design matrix based of the scaled data
        X_train = polynomial_features(x_train, p, intercept=True)
        X_test = polynomial_features(x_test, p, intercept=True)

        # Finding the OLS parameters from the training data
        theta = OLS_parameters(X_train, y_train)

        # Prediciting
        y_pred_test = X_test @ theta
        y_pred_train = X_train @ theta

        # assessing the model with scores
        mse_test = MSE(y_test, y_pred_test)
        r2_test = R2(y_test, y_pred_test)

        mse_train = MSE(y_train, y_pred_train)
        r2_train = R2(y_train, y_pred_train)

        # saving the results in a pandas dataframe
        results.append({
            'p': p,
            'theta': theta,
            'MSE_test': mse_test,
            'R2_test': r2_test,
            'MSE_train': mse_train,
            'R2_train': r2_train,
            'y_pred_test': y_pred_test,
            'y_pred_train': y_pred_train,
            'y_test': y_test,
            'y_train': y_train,
            'y_all': y_all,
            'x_test': x_test,
            'x_train': x_train,
            'x_all': x_all
            })

    df_OLS = pd.DataFrame(results)

    return df_OLS

def plot_OLS_results(df_OLS, p, n):
    """
    Plot the OLS results for a specific number of datapoints 'n' and polynomial degree `p`.
    """
    row = df_OLS[(df_OLS['p'] == p)].iloc[0]

    x_train = row['x_train']
    y_train = row['y_train']
    x_test = row['x_test']
    y_test = row['y_test']
    y_pred_test = row['y_pred_test']
    y_pred_train = row['y_pred_train']

    plt.figure(figsize=(8, 5))

    # Plot training data
    plt.scatter(x_train, y_train, s=6, label='Training data')

    # Plot test data
    plt.scatter(x_test, y_test, s=6, label='Test data')

    # Plot model prediction on test data
    plt.scatter(x_test, y_pred_test, s=6, label='Predicted (test)')

    # Plot model prediction on test data
    plt.scatter(x_train, y_pred_train, s=6, label='Predicted (train)')

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'OLS Polynomial Regression (n={n}, p={p})')
    plt.legend()
    plt.show()

def OLS_various_n_data(p, n_vals):
    """
    Performs Ordinary Least Square analysis on
    the Runge function for a given polynomial degree
    and a range of samples (data points).

    Inputs:
        p (int): Polynomial degree.
        p (np.ndarray): List of number of samples.

    Returns:
        df_OLS (pd.DataFrame): Dataframe containing all results.
    """
    results = []

    for n in n_vals:
        # Making the data and splitting into test/train
        train, test, full = make_data(n)  # making a dataset with size n
        x_train, y_train = train  # training data
        x_test, y_test = test  # test data
        x_all, y_all, y_all_clean = full  # full data

        x_train = x_train.reshape(-1, 1)
        x_test = x_test.reshape(-1, 1)
        x_all = x_all.reshape(-1, 1)
        
        # Making a design matrix based of the scaled data
        X_train = polynomial_features(x_train, p, intercept=True)
        X_test = polynomial_features(x_test, p, intercept=True)


        # Finding the OLS parameters from the training data
        theta = OLS_parameters(X_train, y_train)

        # Prediciting
        y_pred_test = X_test @ theta
        y_pred_train = X_train @ theta

        # assessing the model with scores
        mse_test = MSE(y_test, y_pred_test)
        r2_test = R2(y_test, y_pred_test)

        mse_train = MSE(y_train, y_pred_train)
        r2_train = R2(y_train, y_pred_train)
        
        # saving the results in a pandas dataframe
        results.append({
            'n': n,
            'theta': theta,
            'MSE_test': mse_test,
            'R2_test': r2_test,
            'MSE_train': mse_train,
            'R2_train': r2_train,
            'y_pred_test': y_pred_test,
            'y_pred_train': y_pred_train,
            'y_test': y_test,
            'y_train': y_train,
            'y_all': y_all,
            'x_test': x_test,
            'x_train': x_train,
            'x_all': x_all
            })

    df_OLS = pd.DataFrame(results)

    return df_OLS

# --- Part b) ---
def Ridge_parameters(X, y, lamb, intercept=True):
    """
    Computes the optimal parameters for 
    Ridge regression.

    Inputs:
        X (np.ndarray): Design matrix of shape (n, p), where `n` is the
        number of samples and `p` is the number of features.
        y (np.ndarray): Target vector of shape (n).
        lamb (float): Penalization parameter.
        intercept (boolean): If True (default), the intercept/bias term
        (first column of `X`) is not penalized.

    Returns:
        (np.ndarray): Optimal parameters vector of shape (p)
    """
    if intercept:
        n_features = X.shape[1]
        I = np.eye(n_features)
        I[0, 0] = 0
        params = np.linalg.inv(X.T @ X + lamb * I) @ X.T @ y

    else:
        I = np.eye(np.shape(X.T @ X)[0])
        params = np.linalg.inv(X.T @ X + lamb * I) @ X.T @ y

    return params

def Ridge_various_poly_deg(n, lamb, p_vals):
    """
    Performs Ridge regression analysis on
    the Runge function for a given number of data points,
    a given hyperparameter and for a range of samples (data points).

    Inputs:
        n (int): Number of samples.
        lamb (float): Hyperparameter
        p_vals (np.ndarray): List of polynomial degrees.

    Returns:
        df_Ridge (pd.DataFrame): Dataframe containing all results.
    """
    results = []
    # Making the data and splitting into test/train
    train, test, full = make_data(n)  # making a dataset with size n
    x_train, y_train = train  # training data
    x_test, y_test = test  # test data
    x_all, y_all, y_all_clean = full  # full data

    x_train = x_train.reshape(-1, 1)
    x_test = x_test.reshape(-1, 1)
    x_all = x_all.reshape(-1, 1)

    for p in p_vals:
        # Making a design matrix based of the scaled data
        X_train = polynomial_features(x_train, p, intercept=True)
        X_test = polynomial_features(x_test, p, intercept=True)

        # Finding the OLS parameters from the training data
        theta = Ridge_parameters(X_train, y_train, lamb, intercept=True)

        # Prediciting
        y_pred_test = X_test @ theta
        y_pred_train = X_train @ theta

        # assessing the model with scores
        mse_test = MSE(y_test, y_pred_test)
        r2_test = R2(y_test, y_pred_test)

        mse_train = MSE(y_train, y_pred_train)
        r2_train = R2(y_train, y_pred_train)

        # saving the results in a pandas dataframe
        results.append({
            'p': p,
            'theta': theta,
            'MSE_test': mse_test,
            'R2_test': r2_test,
            'MSE_train': mse_train,
            'R2_train': r2_train,
            'y_pred_test': y_pred_test,
            'y_pred_train': y_pred_train,
            'y_test': y_test,
            'y_train': y_train,
            'y_all': y_all,
            'x_test': x_test,
            'x_train': x_train,
            'x_all': x_all
            })

    df_Ridge = pd.DataFrame(results)

    return df_Ridge

def plot_Ridge_results(df_Ridge, p, n, lamb):
    """
    Plot the Ridge results for a specific number of datapoints 'n' and polynomial degree `p`.
    """
    row = df_Ridge[(df_Ridge['p'] == p)].iloc[0]

    x_train = row['x_train']
    y_train = row['y_train']
    x_test = row['x_test']
    y_test = row['y_test']
    y_pred_test = row['y_pred_test']
    y_pred_train = row['y_pred_train']

    plt.figure(figsize=(8, 5))

    # Plot training data
    plt.scatter(x_train, y_train, s=6, label='Training data')

    # Plot test data
    plt.scatter(x_test, y_test, s=6, label='Test data')

    # Plot model prediction on test data
    plt.scatter(x_test, y_pred_test, s=6, label='Predicted (test)')

    # Plot model prediction on test data
    plt.scatter(x_train, y_pred_train, s=6, label='Predicted (train)')

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(rf'Ridge Polynomial Regression (n={n}, $\lambda$ = {lamb}, p={p})')
    plt.legend()
    plt.show()

def Ridge_various_lambs(n, p, lambs):
    """
    Performs Ridge regression analysis on
    the Runge function for a given polynomial degree,
    a given number of samples and for a range of hyperparameters.

    Inputs:
        n (int): Number of samples.
        p (int): Polynomial degree.
        lambs (np.ndarray): List of hyperparameters.

    Returns:
        df_Ridge (pd.DataFrame): Dataframe containing all results.
    """
    results = []

    # Making the data and splitting into test/train
    train, test, full = make_data(n)  # making a dataset with size n
    x_train, y_train = train  # training data
    x_test, y_test = test  # test data
    x_all, y_all, y_all_clean = full  # full data

    x_train = x_train.reshape(-1, 1)
    x_test = x_test.reshape(-1, 1)
    x_all = x_all.reshape(-1, 1)

    # Making a design matrix based of the scaled data
    X_train = polynomial_features(x_train, p, intercept=True)
    X_test = polynomial_features(x_test, p, intercept=True)

    for l in lambs:
        # Finding the OLS parameters from the training data
        theta = Ridge_parameters(X_train, y_train, l, intercept=True)

        # Prediciting
        y_pred_test = X_test @ theta
        y_pred_train = X_train @ theta

        # assessing the model with scores
        mse_test = MSE(y_test, y_pred_test)
        r2_test = R2(y_test, y_pred_test)

        mse_train = MSE(y_train, y_pred_train)
        r2_train = R2(y_train, y_pred_train)

        # saving the results in a pandas dataframe
        results.append({
            'lambda': l,
            'theta': theta,
            'MSE_test': mse_test,
            'R2_test': r2_test,
            'MSE_train': mse_train,
            'R2_train': r2_train,
            'y_pred_test': y_pred_test,
            'y_pred_train': y_pred_train,
            'y_test': y_test,
            'y_train': y_train,
            'y_all': y_all,
            'x_test': x_test,
            'x_train': x_train,
            'x_all': x_all
            })

    df_Ridge = pd.DataFrame(results)

    return df_Ridge

def Ridge_various_n_data(p, lamb, n_vals):
    """
    Performs Ordinary Least Square analysis on
    the Runge function for a given polynomial degree
    and a range of samples (data points).

    Inputs:
        p (int): Polynomial degree.
        p (np.ndarray): List of number of samples.

    Returns:
        df_Ridge (pd.DataFrame): Dataframe containing all results.
    """
    results = []

    for n in n_vals:
        # Making the data and splitting into test/train
        train, test, full = make_data(n)  # making a dataset with size n
        x_train, y_train = train  # training data
        x_test, y_test = test  # test data
        x_all, y_all, y_all_clean = full  # full data

        x_train = x_train.reshape(-1, 1)
        x_test = x_test.reshape(-1, 1)
        x_all = x_all.reshape(-1, 1)

        # Making a design matrix based of the scaled data
        X_train = polynomial_features(x_train, p, intercept=True)
        X_test = polynomial_features(x_test, p, intercept=True)

        # Finding the Ridge parameters from the training data
        theta = Ridge_parameters(X_train, y_train, lamb)

        # Prediciting
        y_pred_test = X_test @ theta
        y_pred_train = X_train @ theta

        # assessing the model with scores
        mse_test = MSE(y_test, y_pred_test)
        r2_test = R2(y_test, y_pred_test)

        mse_train = MSE(y_train, y_pred_train)
        r2_train = R2(y_train, y_pred_train)
        
        # saving the results in a pandas dataframe
        results.append({
            'n': n,
            'theta': theta,
            'MSE_test': mse_test,
            'R2_test': r2_test,
            'MSE_train': mse_train,
            'R2_train': r2_train,
            'y_pred_test': y_pred_test,
            'y_pred_train': y_pred_train,
            'y_test': y_test,
            'y_train': y_train,
            'y_all': y_all,
            'x_test': x_test,
            'x_train': x_train,
            'x_all': x_all
            })

    df_Ridge = pd.DataFrame(results)

    return df_Ridge

# --- Part c) ---

def ols_gradient(X: np.ndarray, y: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """
    Calculate the gradient of the Ordinary Least Squares cost function.

    Args:
        X (np.ndarray)    : The input feature matrix of shape (n_samples, n_features).
        y (np.ndarray)    : The target values of shape (n_samples,).
        beta (np.ndarray) : The current coefficients of shape (n_features,).

    Returns:
        np.ndarray: The gradient of the OLS cost function with respect to beta.
    """
    return (2 / len(y)) * (X.T @ (X @ beta - y))

def ridge_gradient(X: np.ndarray, y: np.ndarray, beta: np.ndarray, lambda_: float) -> np.ndarray:
    """
    Calculate the gradient of the Ridge regression cost function.

    Args:
        X (np.ndarray)    : The input feature matrix of shape (n_samples, n_features).
        y (np.ndarray)    : The target values of shape (n_samples,).
        beta (np.ndarray) : The current coefficients of shape (n_features,).
        lambda_ (float)   : The regularization strength parameter.

    Returns:
        np.ndarray: The gradient of the Ridge cost function with respect to beta.
    """
    return (2 / len(y)) * (X.T @ (X @ beta - y)) + 2 * lambda_ * beta

def gradient_descent_ols(
    X: np.ndarray, 
    y: np.ndarray, 
    learning_rate: float = 0.01, 
    n_iterations: int = 1000, 
    tol: float = 1e-6, 
    use_tol: bool = False
) -> tuple[np.ndarray, list[float]]:
    """
    Perform gradient descent for Ordinary Least Squares regression.

    Args:
        X (np.ndarray)        : The input feature matrix of shape (n_samples, n_features).
        y (np.ndarray)        : The target values of shape (n_samples,).
        learning_rate (float) : The learning rate for gradient descent.
        n_iterations (int)    : The maximum number of iterations.
        tol (float)           : The tolerance for convergence.
        use_tol (bool)        : Whether to use tolerance for convergence.

    Returns:
        tuple[np.ndarray, list[float]]: The estimated coefficients and the cost history.
    """
    n_samples, n_features = X.shape
    theta =  np.zeros(n_features)
    cost_history = []
    for i in range(n_iterations):
        gradient = ols_gradient(X, y, theta)
        theta -= learning_rate * gradient
        # cost is the OLS cost function
        cost = (1 / n_samples) * np.sum((X @ theta - y) ** 2)
        cost_history.append(cost)
        if use_tol and i > 0 and abs(cost_history[-2] - cost) < tol:
            print(f"Converged after {i} iterations.")
            break
    return theta, cost_history

def gradient_descent_ridge(
    X: np.ndarray, 
    y: np.ndarray, 
    alpha: float, 
    learning_rate: float = 0.01, 
    n_iterations: int = 1000, 
    tol: float = 1e-6, 
    use_tol: bool = False
) -> tuple[np.ndarray, list[float]]:
    """
    Perform gradient descent for Ridge regression.

    Args:
        X (np.ndarray)        : The input feature matrix of shape (n_samples, n_features).
        y (np.ndarray)        : The target values of shape (n_samples,).
        alpha (float)         : The regularization strength parameter.
        learning_rate (float) : The learning rate for gradient descent.
        n_iterations (int)    : The maximum number of iterations.
        tol (float)           : The tolerance for convergence.
        use_tol (bool)        : Whether to use tolerance for convergence.

    Returns:
        tuple[np.ndarray, list[float]]: The estimated coefficients and the cost history.
    """
    n_samples, n_features = X.shape
    theta = np.zeros(n_features)
    cost_history = []
    for i in range(n_iterations):
        gradient = ridge_gradient(X, y, theta, alpha)
        theta -= learning_rate * gradient
        # cost is the Ridge cost function, including the regularization term
        cost = (1 / n_samples) * np.sum((X @ theta - y) ** 2) + alpha * np.sum(theta**2)
        cost_history.append(cost)
        if use_tol and i > 0 and abs(cost_history[-2] - cost) < tol:
            print(f"Converged after {i} iterations.")
            break
    return theta, cost_history

# --- Part d) ---

def gradient_descent_advanced(
    X: np.ndarray,
    y: np.ndarray,
    method: str = "gd",
    lr_method: str = "ols",
    learning_rate: float = 0.01,
    n_iterations: int = 1000,
    tol: float = 1e-6,
    use_tol: bool = False,
    beta: float = 0.9,
    beta1: float = 0.8,
    beta2: float = 0.6,
    epsilon: float = 1e-8,
    lambda_: float = 0.01,
) -> tuple[np.ndarray, list[float]]:
    """
    Perform advanced gradient descent with various optimization methods.

    Args:
        X (np.ndarray)        : The input feature matrix of shape (n_samples, n_features).
        y (np.ndarray)        : The target values of shape (n_samples,).
        method (str)          : The optimization method to use ('gd', 'momentum', 'adagrad', 'rmsprop', 'adam').
        lr_method (str)       : The linear regression method ('ols', 'ridge', 'lasso').
        learning_rate (float) : The learning rate for gradient descent.
        n_iterations (int)    : The maximum number of iterations.
        tol (float)           : The tolerance for convergence.
        use_tol (bool)        : Whether to use tolerance for convergence.
        beta (float)          : The momentum factor for 'momentum' method and 'rmsprop'.
        beta1 (float)         : The exponential decay rate for the first moment estimate in 'adam'.
        beta2 (float)         : The exponential decay rate for the second moment estimate in 'adam'.
        epsilon (float)       : A small constant to prevent division by zero in adaptive learning methods.
        lambda_ (float)       : The regularization strength parameter for ridge and lasso methods.

    Returns:
        tuple[np.ndarray, list[float]]: The estimated coefficients and the cost history.
    """
    n_samples, n_features = X.shape
    theta = np.zeros(n_features)
    cost_history = []
    m = np.zeros(n_features)  # For momentum and Adam
    v = np.zeros(n_features)  # For Adam
    for i in range(n_iterations):
        if lr_method == "ols":
            gradient = ols_gradient(X, y, theta)
        elif lr_method == "ridge":
            gradient = ridge_gradient(X, y, theta, lambda_=lambda_)
        elif lr_method == "lasso":
            gradient = lasso_gradient(X, y, theta, lambda_)
        else:
            raise ValueError("Unknown linear regression method")
        if method == "momentum":
            m = beta * m + (1 - beta) * gradient
            gradient = m
        elif method == "adagrad":
            v += gradient**2
            adjusted_lr = learning_rate / (np.sqrt(v) + epsilon)
            gradient = adjusted_lr * gradient
        elif method == "rmsprop":
            v = beta * v + (1 - beta) * gradient**2
            adjusted_lr = learning_rate / (np.sqrt(v) + epsilon)
            gradient = adjusted_lr * gradient
        elif method == "adam":
            m = beta1 * m + (1 - beta1) * gradient
            v = beta2 * v + (1 - beta2) * (gradient**2)
            m_hat = m / (1 - beta1 ** (i + 1))
            v_hat = v / (1 - beta2 ** (i + 1))
            adjusted_lr = learning_rate / (np.sqrt(v_hat) + epsilon)
            gradient = adjusted_lr * m_hat
        elif method == "gd":
            # For GD, do nothing
            pass
        else:
            raise ValueError("Unknown optimization method")
        theta -= learning_rate * gradient
        if lr_method == "ols":
            cost = (1 / n_samples) * np.sum((X @ theta - y) ** 2)
        elif lr_method == "ridge":
            cost = (1 / n_samples) * np.sum((X @ theta - y) ** 2) + lambda_ * np.sum(
                theta**2
            )
        elif lr_method == "lasso":
            cost = (1 / n_samples) * np.sum((X @ theta - y) ** 2) + lambda_ * np.sum(
                np.abs(theta)
            )
        cost_history.append(cost)
        if use_tol and i > 0 and abs(cost_history[-2] - cost) < tol:
            print(f"{method} converged after {i} iterations.")
            break
    return theta, cost_history

# --- Part e) ---

def lasso_gradient(X, y, beta, lmbd):
    return (2 / len(y)) * X.T @ (X @ beta - y) + np.sign(beta) * lmbd

def gradient_descent_lasso(
    X, y, lmbd, learning_rate=0.0001, n_iterations=1000, tol=1e-6, use_tol=False
):
    n_samples, n_features = X.shape
    theta = np.random.random(n_features)
    cost_history = []
    for i in range(n_iterations):
        gradient = lasso_gradient(X, y, theta, lmbd)
        theta -= learning_rate * gradient
        # cost is the Lasso cost function, including the regularization term
        cost = (1 / n_samples) * np.sum((X @ theta - y) ** 2) + lmbd * np.sum(
            np.abs(theta)
        )
        cost_history.append(cost)
        if use_tol and i > 0 and abs(cost_history[-2] - cost) < tol:
            print(f"Converged after {i} iterations.")
            break
    return theta, cost_history


# --- Part f) ---
def stochastic_gradient_descent_advanced(
    X,
    y,
    method="gd",
    lr_method="ols",
    learning_rate=0.01,
    tol=1e-6,
    use_tol=False,
    beta=0.9,
    beta1=0.9,
    beta2=0.999,
    epsilon=1e-8,
    lambda_=0.001,
    n_epochs = 100
):
    n_samples, n_features = X.shape
    seed=np.random.seed(2018)
    theta = np.random.random(n_features)
    batch_size = 20
    mini_batches = int(n_samples/batch_size)
    cost_history = []
    m = np.zeros(n_features)  # For momentum and Adam
    v = np.zeros(n_features)  # For Adam
    for j in range(n_epochs):
        for i in range(mini_batches):
            init_pos = np.random.choice(np.linspace(0, n_samples - batch_size,mini_batches), 1)
            
            X_, y_ = (
                X[int(init_pos[0]) : int(init_pos[0]) + batch_size],
                y[int(init_pos[0]) : int(init_pos[0]) + batch_size],
            )
            if lr_method == "ols":
                gradient = ols_gradient(X_, y_, theta)
            elif lr_method == "ridge":
                gradient = ridge_gradient(X_, y_, theta, lambda_=lambda_)
            elif lr_method == "lasso":
                gradient = lasso_gradient(X_, y_, theta, lambda_)
            else:
                raise ValueError("Unknown linear regression method")
            if method == "momentum":
                m = beta * m + (1 - beta) * gradient
                gradient = m
            elif method == "adagrad":
                v += gradient**2
                adjusted_lr = learning_rate / (np.sqrt(v) + epsilon)
                gradient = adjusted_lr * gradient
            elif method == "rmsprop":
                v = beta * v + (1 - beta) * gradient**2
                adjusted_lr = learning_rate / (np.sqrt(v) + epsilon)
                gradient = adjusted_lr * gradient
            elif method == "adam":
                m = beta1 * m + (1 - beta1) * gradient
                v = beta2 * v + (1 - beta2) * (gradient**2)
                m_hat = m / (1 - beta1 ** (i + 1))
                v_hat = v / (1 - beta2 ** (i + 1))
                adjusted_lr = learning_rate / (np.sqrt(v_hat) + epsilon)
                gradient = adjusted_lr * m_hat
            elif method == "gd":
                # For GD, do nothing
                pass
            else:
                raise ValueError("Unknown optimization method")
            theta -= learning_rate * gradient
        if lr_method == "ols":
            cost = (1 / n_samples) * np.sum((X_ @ theta - y_) ** 2)
        elif lr_method == "ridge":
            cost = (1 / n_samples) * np.sum((X_ @ theta - y_) ** 2) + lambda_ * np.sum(
                theta**2
            )
        elif lr_method == "lasso":
            cost = (1 / n_samples) * np.sum((X_ @ theta - y_) ** 2) + lambda_ * np.sum(
                np.abs(theta)
            )
        cost_history.append(cost)
        if use_tol and j > 100 and abs(cost_history[-2] - cost) < tol:
            print(f"{method} converged after {j} epochs.")
            break
    return theta, cost_history


# --- Part g) ---


def ols_gh(x_train, y_train, x_eval, degree):
    """
    Regular ols from scikit
    """
    model = make_pipeline(
        PolynomialFeatures(degree=degree, include_bias=False), #ingen scaling fordi vi scaler i make_data
        StandardScaler(),
        LinearRegression(fit_intercept=True)
    )
    model.fit(x_train.reshape(-1, 1), y_train)
    y_pred = model.predict(x_eval.reshape(-1, 1)).ravel()
    return y_pred, model

def bootstrap(degrees, x_train, x_test, y_train, y_test, boots_reps, seed=seed):
    
    rng = np.random.default_rng(seed)

    boot_results = {
        "degree": [],
        "mse_boots": [],       # <- MSE på test, bootstrap-aggregert
        "var_boots": [],  # <- prediktiv varians på test
        "bias2_boots": [], # <- (mean_pred − y_test)^2  (observasjons-bias^2)
        "train_mse": [],
        "test_mse": [],
    }

    for d in degrees:
        # 1) én direkte fit (ikke bootstrap) for referanse
        yhat_train, model = ols_gh(x_train, y_train, x_train, degree=d)
        yhat_test = model.predict(x_test.reshape(-1,1)).ravel()
        boot_results["train_mse"].append(mse(y_train, yhat_train))
        boot_results["test_mse"].append(mse(y_test,  yhat_test))

        # 2) bootstrap-prediksjoner på TEST
        boots_pred = np.empty((boots_reps, x_test.size), dtype=float)

        for b in range(boots_reps):
            idx = rng.choice(x_train.size, size=x_train.size, replace=True)
            xb, yb = x_train[idx], y_train[idx]
            boots_pred[b], _ = ols_gh(xb, yb, x_test, degree=d)

        # 3) aggreger mot y_test (y holdes fast)
        mean_t = boots_pred.mean(axis=0)
        var_t  = boots_pred.var(axis=0, ddof=1)
        mse_t  = ((boots_pred - y_test[None, :])**2).mean(axis=0)
        bias2_obs = (mean_t - y_test)**2

        boot_results["degree"].append(d)
        boot_results["mse_boots"].append(mse_t.mean())
        boot_results["var_boots"].append(var_t.mean())
        boot_results["bias2_boots"].append(bias2_obs.mean())

    for k in list(boot_results.keys()):
        boot_results[k] = np.array(boot_results[k])
    return boot_results


# --- Part h) ---

# K-fold cross-validation for OLS


def kfold_cv_mse_ols(degree, k, x, y, seed=seed):
    kf = KFold(n_splits=k, shuffle=True, random_state=seed)
    kfold_mses = []
    for train_idx, val_idx in kf.split(x):
        x_train_cv, x_val = x[train_idx], x[val_idx]
        y_train_cv, y_val = y[train_idx], y[val_idx]

        predicted_val, _ = ols_gh(x_train_cv, y_train_cv, x_val, degree)
        kfold_mses.append(mse(y_val, predicted_val))

    return np.mean(kfold_mses)


# K-fold cross-validation for ridge, lasso


def cv_for_methods(method, degree, lambdas, k, x, y, seed=seed):
    x = np.asarray(x)
    y = np.asarray(y)

    kf = KFold(n_splits=k, shuffle=True, random_state=seed)
    cv_means = []

    for lam in lambdas:
        kfold_mses = []
        for train_idx, val_idx in kf.split(x):
            x_train_cv, x_val = x[train_idx], x[val_idx]
            y_train_cv, y_val = y[train_idx], y[val_idx]

            if method == "ridge":
                base = Ridge(alpha=lam, fit_intercept=False, max_iter=300_000, tol=1e-3)
            elif method == "lasso":
                base = Lasso(
                    alpha=lam,
                    fit_intercept=False,
                    max_iter=2_000_000,
                    tol=5e-3,
                    selection="cyclic",
                )
            else:
                raise ValueError("method must be 'ridge' or 'lasso'")

            model = make_pipeline(
                PolynomialFeatures(degree, include_bias=True), base
            )

            model.fit(x_train_cv.reshape(-1, 1), y_train_cv)
            pred_vals = model.predict(x_val.reshape(-1, 1)).ravel()
            kfold_mses.append(mse(y_val, pred_vals))

        cv_means.append(np.mean(kfold_mses))

    cv_means = np.array(cv_means)
    best_idx = int(np.argmin(cv_means))

    return {
        "lambdas": np.array(lambdas, dtype=float),
        "cv_mse": cv_means,
        "best_lambda": float(lambdas[best_idx]),
        "best_mse": float(cv_means[best_idx]),
    }
