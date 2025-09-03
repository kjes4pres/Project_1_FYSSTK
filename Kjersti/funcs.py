'''
Functions for project 1 FYS-STK4155
Author: Kjersti Stangeland, September 2025
'''

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def make_data(n):
    '''
    Make a dataset with a given number (n) datapoints
    of the Runge function.
    '''
    x = np.linspace(-1, 1, n)
    y = 1/(1 + 25*x**2) + np.random.normal(0, 0.1)

    return x, y


def polynomial_features(x, p, intercept=True):
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

def OLS_parameters(X, y):
    ''' 
    Find the OLS parameters
    '''
    X_T = np.transpose(X)
    X_T_X = X_T @ X

    return np.linalg.pinv(X_T_X) @ X_T @ y


def Ridge_parameters(X, y, lamb=0.01):
    # Assumes X is scaled and has no intercept column

    I = np.eye(np.shape(X.T @ X)[0])
    
    return np.linalg.inv(X.T @ X + lamb*I) @ X.T @ y

def MSE(y_data, y_pred):
    ''' 
    Mean square error
    '''
    return np.mean((y_data - y_pred)**2)

def R2(y_data, y_pred):
    return 1 - (np.sum((y_data - y_pred)**2))/(np.sum((y_data - np.mean(y_data))**2))

def standardize(X, y):
    # Standardize features (zero mean, unit variance for each feature)
    X_mean = X.mean(axis=0) # The mean of each column/feature
    X_std = X.std(axis=0)
    X_std[X_std == 0] = 1  # safeguard to avoid division by zero for constant features
    X_norm = (X - X_mean) / X_std

    # Center the target to zero mean (optional, to simplify intercept handling)
    y_mean = y.mean()
    y_centered = y - y_mean

    return X_norm, y_centered

def split_n_train(X, y, size):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=size)

    return X_train, X_test, y_train, y_test

# ----------- Functions for plotting -----------

def plot_mse_degree(df, n_vals, method):
    fig, ax = plt.subplots()

    colormap='viridis'
    num_colors = len(n_vals)
    cmap = plt.get_cmap(colormap, num_colors)

    for i, en in enumerate(n_vals):
        n_df = df[df['n'] == en]
        color = cmap(i) 
        ax.plot(n_df['p'], n_df['MSE'], marker='o', markersize='3', linewidth='2', color=color, label=f'N: {en}')
    
    ax.set_title('MSE as a function of polynomial degree')
    ax.legend(loc='upper right', fontsize=8)
    ax.set_xlabel('Polynomial degree')
    ax.set_ylabel('MSE')

    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    ax.text(0.05, 0.95, method, transform=ax.transAxes, fontsize=12,
        verticalalignment='top', bbox=props)

    plt.show()

def plot_R2_degree(df, n_vals, method):
    fig, ax = plt.subplots()

    colormap='viridis'
    num_colors = len(n_vals)
    cmap = plt.get_cmap(colormap, num_colors)

    for i, en in enumerate(n_vals):
        n_df = df[df['n'] == en]
        color = cmap(i) 
        ax.plot(n_df['p'], n_df['R2'], marker='o', markersize='3', linewidth='2', color=color, label=f'N: {en}')
    
    ax.set_title(r'$R^2$ as a function of polynomial degree')
    ax.legend(loc='upper right', fontsize=8)
    ax.set_xlabel('Polynomial degree')
    ax.set_ylabel(r'$R^2$')

    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    ax.text(0.05, 0.95, method, transform=ax.transAxes, fontsize=12,
        verticalalignment='top', bbox=props)

    plt.show()

def plot_mse_datapoints(df, p_vals, method):
    fig, ax = plt.subplots()

    colormap='plasma'
    num_colors = len(p_vals)
    cmap = plt.get_cmap(colormap, num_colors)

    for i, pe in enumerate(p_vals):
        p_df = df[df['p'] == pe]
        color = cmap(i) 
        ax.plot(p_df['n'], p_df['MSE'], marker='o', markersize='3', linewidth='2', color=color, label=f'p: {pe}')
    
    ax.set_title('MSE as a function of number of datapoints')
    ax.legend(loc='upper right', fontsize=8)
    ax.set_xlabel('Number of datapoints')
    ax.set_ylabel('MSE')

    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    ax.text(0.05, 0.95, method, transform=ax.transAxes, fontsize=12,
        verticalalignment='top', bbox=props)

    plt.show()

def plot_R2_datapoints(df, p_vals, method):
    fig, ax = plt.subplots()

    colormap='plasma'
    num_colors = len(p_vals)
    cmap = plt.get_cmap(colormap, num_colors)

    for i, pe in enumerate(p_vals):
        p_df = df[df['p'] == pe]
        color = cmap(i) 
        ax.plot(p_df['n'], p_df['R2'], marker='o', markersize='3', linewidth='2', color=color, label=f'p: {pe}')
    
    ax.set_title(r'$R^2$ as a function of number of datapoints')
    ax.legend(loc='upper right', fontsize=8)
    ax.set_xlabel('Number of datapoints')
    ax.set_ylabel(r'$R^2$')

    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    ax.text(0.05, 0.95, method, transform=ax.transAxes, fontsize=12,
        verticalalignment='top', bbox=props)

    plt.show()

def plot_theta_degree_intercept(df, n_vals, method):
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))

    colormap='viridis'
    num_colors = len(n_vals)
    cmap = plt.get_cmap(colormap, num_colors)

    for i, en in enumerate(n_vals):
        n_df = df[df['n'] == en]
        color = cmap(i) 
        ax[0].plot(n_df['p'], n_df['theta'].apply(lambda x: x[1]), marker='o', markersize='3', linewidth='2', color=color, label=f'N: {en}')
        ax[1].plot(n_df['p'], n_df['theta'].apply(lambda x: x[2]), marker='o', markersize='3', linewidth='2', color=color, label=f'N: {en}')
        
    ax[0].set_title(r'$\theta_1$')
    ax[1].set_title(r'$\theta_2$')

    fig.suptitle(f'Features as a function of polynomial degree \n Method: {method}', y=1.05)

    for axs in ax:
        axs.legend(loc='upper right', fontsize=8)
        axs.set_xlabel('Polynomial degree', fontsize=8)
        axs.set_ylabel(r'$\theta$', fontsize=8)    

    plt.show()

def plot_theta_degree_no_intercept(df, n_vals, method):
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))

    colormap='viridis'
    num_colors = len(n_vals)
    cmap = plt.get_cmap(colormap, num_colors)

    for i, en in enumerate(n_vals):
        n_df = df[df['n'] == en]
        color = cmap(i) 
        ax[0].plot(n_df['p'], n_df['theta'].apply(lambda x: x[0]), marker='o', markersize='3', linewidth='2', color=color, label=f'N: {en}')
        ax[1].plot(n_df['p'], n_df['theta'].apply(lambda x: x[1]), marker='o', markersize='3', linewidth='2', color=color, label=f'N: {en}')
        
    ax[0].set_title(r'$\theta_1$')
    ax[1].set_title(r'$\theta_2$')

    fig.suptitle(f'Features as a function of polynomial degree \n Method: {method}', y=1.05)

    for axs in ax:
        axs.legend(loc='upper right', fontsize=8)
        axs.set_xlabel('Polynomial degree', fontsize=8)
        axs.set_ylabel(r'$\theta$', fontsize=8)    

    plt.show()