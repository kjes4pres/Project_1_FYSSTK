"""
Functions for project 1, FYS-STK4155

Authors:
- Kjersti Stangeland
- Ingvild Olden Bjerklund
- Jenny Guldvog
- Sverre Johansen

September 2025
"""

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
from sklearn.metrics import mean_squared_error as mse
from sklearn.linear_model import Lasso
from sklearn.model_selection import KFold

import matplotlib.style as mplstyle

mplstyle.use(["ggplot", "fast"])

# plt.rcParams.update({
#     "text.usetex": True,
#     "font.family": "serif",
#     "font.size": 10,
# })


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
    for x in (-1, 1).

    Creates train and test data sets
    """
    x = np.linspace(-1, 1, n)

    y_clean = f_true(x)
    y = y_clean + np.random.normal(0, 0.1, n)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=seed, shuffle=True
    )

    train = (x_train, y_train)
    test = (x_test, y_test)
    full = (x, y, y_clean)
    return train, test, full


def MSE(y_data, y_pred):
    """
    Mean square error
    """
    mse = np.mean((y_data - y_pred) ** 2)

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


def OLS_parameters(X, y):
    """
    Find the OLS parameters
    """
    return np.linalg.pinv(X) @ y


def Ridge_parameters(X, y, lamb=0.01):
    """
    Doc string kommer...
    """
    # Assumes X is scaled and has no intercept column
    I = np.eye(np.shape(X.T @ X)[0])

    return np.linalg.inv(X.T @ X + lamb * I) @ X.T @ y


# --- Part a) ---
def OLS_results(n_vals, p_vals):
    """
    ...
    """
    results = []

    for n in n_vals:
        train, test, full = make_data(n)  # making a dataset with size n
        x_train, y_train = train  # training data
        x_test, y_test = test  # test data
        x_all, y_all, y_all_clean = full  # actual data

        x_train = x_train.reshape(-1, 1)
        x_test = x_test.reshape(-1, 1)
        x_all = x_all.reshape(-1, 1)

        # making an OLS model for a given polynomial degree, p
        for p in p_vals:
            model = make_pipeline(
                PolynomialFeatures(degree=p, include_bias=True),
                StandardScaler(with_mean=False),
                LinearRegression(fit_intercept=False),
            )

            # using the training data to train the model
            model.fit(x_train, y_train)

            # using the test data to make a prediction, unsee data for the model
            y_pred_test = model.predict(x_test)
            y_pred_train = model.predict(x_train)

            # assessing the model with scores
            mse_test = mean_squared_error(y_test, y_pred_test)
            r2_test = r2_score(y_test, y_pred_test)

            mse_train = mean_squared_error(y_train, y_pred_train)
            r2_train = r2_score(y_train, y_pred_train)

            # extracting the model features
            theta = model.named_steps["linearregression"].coef_

            # saving the results in a pandas dataframe
            results.append(
                {
                    "n": n,
                    "p": p,
                    "theta": theta,
                    "MSE_test": mse_test,
                    "R2_test": r2_test,
                    "MSE_train": mse_train,
                    "R2_train": r2_train,
                    "y_pred_test": y_pred_test,
                    "y_pred_train": y_pred_train,
                    "y_test": y_test,
                    "y_train": y_train,
                    "y_all": y_all,
                    "x_test": x_test,
                    "x_train": x_train,
                    "x_all": x_all,
                }
            )

    df_OLS = pd.DataFrame(results)

    return df_OLS


def plot_OLS_results(df_OLS, n, p):
    """
    Plot the OLS results for a specific number of datapoints 'n' and polynomial degree `p`.
    """
    row = df_OLS[(df_OLS["n"] == n) & (df_OLS["p"] == p)].iloc[0]

    x_train = row["x_train"]
    y_train = row["y_train"]
    x_test = row["x_test"]
    y_test = row["y_test"]
    x_all = row["x_all"]
    y_all = row["y_all"]
    y_pred_test = row["y_pred_test"]
    y_pred_train = row["y_pred_train"]

    plt.figure(figsize=(8, 5))

    # Plot actual data
    plt.scatter(x_all, y_all, s=6, label="Actual data")

    # Plot training data
    plt.scatter(x_train, y_train, s=6, label="Training data")

    # Plot test data
    plt.scatter(x_test, y_test, s=6, label="Test data")

    # Plot model prediction on test data
    plt.scatter(x_test, y_pred_test, s=6, label="Predicted (test)")

    # Plot model prediction on test data
    plt.scatter(x_train, y_pred_train, s=6, label="Predicted (train)")

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"OLS Polynomial Regression (n={n}, p={p})")
    plt.legend()
    plt.show()


# --- Part b) ---
def Ridge_results(n_vals, p_vals, lambdas):
    """
    ...
    """
    results = []

    for n in n_vals:
        train, test, full = make_data(n)  # making a dataset with size n
        x_train, y_train = train
        x_test, y_test = test
        x_all, y_all, y_all_clean = full

        x_train = x_train.reshape(-1, 1)
        x_test = x_test.reshape(-1, 1)
        x_all = x_all.reshape(-1, 1)

        for p in p_vals:
            for l in lambdas:
                model = make_pipeline(
                    PolynomialFeatures(degree=p, include_bias=True),
                    StandardScaler(with_mean=False),
                    Ridge(alpha=l, fit_intercept=False),
                )

                model.fit(x_train, y_train)
                y_pred = model.predict(x_test)

                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)

                theta = model.named_steps["ridge"].coef_

                results.append(
                    {
                        "n": n,
                        "p": p,
                        "lambda": l,
                        "theta": theta,
                        "MSE": mse,
                        "R2": r2,
                        "y_pred": y_pred,
                        "y_test": y_test,
                        "y_train": y_train,
                        "y_all": y_all,
                        "x_test": x_test,
                        "x_train": x_train,
                        "x_all": x_all,
                    }
                )

    df_Ridge = pd.DataFrame(results)
    return df_Ridge


def plot_Ridge_results(df_Ridge, n, p, l):
    """
    Plot Ridge regression results for specific number of data points 'n', polynomial degree 'p', and lambda 'l'.
    """
    row = df_Ridge[
        (df_Ridge["n"] == n) & (df_Ridge["p"] == p) & (df_Ridge["lambda"] == l)
    ].iloc[0]

    x_train = row["x_train"]
    y_train = row["y_train"]
    x_test = row["x_test"]
    y_test = row["y_test"]
    x_all = row["x_all"]
    y_all = row["y_all"]
    y_pred = row["y_pred"]

    plt.figure(figsize=(8, 5))

    # Plot actual data
    plt.scatter(x_all, y_all, s=6, label="Actual data")

    # Plot training data
    plt.scatter(x_train, y_train, s=6, label="Training data")

    # Plot test data
    plt.scatter(x_test, y_test, s=6, label="Test data")

    # Plot predicted test values
    plt.scatter(x_test, y_pred, s=6, label="Predicted (test)")

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(rf"Ridge Regression (n={n}, p={p}, $\lambda$={l:.2e})")
    plt.legend()
    plt.show()


# --- Part c) ---


def ols_gradient(X, y, beta):
    return (2 / len(y)) * (X.T @ (X @ beta - y))


def ridge_gradient(X, y, beta, lambda_):
    return (2 / len(y)) * (X.T @ (X @ beta - y)) + 2 * lambda_ * beta


def lasso_gradient(X, y, beta, lmbd):
    return (2 / len(y)) * X.T @ (X @ beta - y) + np.sign(beta) * lmbd


def gradient_descent_ols(
    X, y, learning_rate=0.01, n_iterations=1000, tol=1e-6, use_tol=False
):
    n_samples, n_features = X.shape
    theta = np.zeros(n_features)
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
    X, y, alpha, learning_rate=0.01, n_iterations=1000, tol=1e-6, use_tol=False
):
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
    X,
    y,
    method="gd",
    lr_method="ols",
    learning_rate=0.01,
    n_iterations=1000,
    tol=1e-6,
    use_tol=False,
    beta=0.9,
    beta1=0.8,
    beta2=0.6,
    epsilon=1e-8,
    lambda_=0.01,
):
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
def gradient_descent_lasso(
    X, y, lmbd, learning_rate=0.0001, n_iterations=1000, tol=1e-6, use_tol=False
):
    n_samples, n_features = X.shape
    theta = np.zeros(n_features)
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
    n_iterations=1000,
    tol=1e-6,
    use_tol=False,
    beta=0.9,
    epsilon=1e-8,
    lambda_=0.01,
):
    n_samples, n_features = X.shape
    theta = np.zeros(n_features)
    batch_size = 10
    cost_history = []
    m = np.zeros(n_features)  # For momentum and Adam
    v = np.zeros(n_features)  # For Adam
    for i in range(n_iterations):
        init_pos = np.random.choice(np.arange(0, n_samples - batch_size), 1)
        X_, y_ = (
            X[init_pos[0] : init_pos[0] + batch_size],
            y[init_pos[0] : init_pos[0] + batch_size],
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
            m = beta * m + (1 - beta) * gradient
            v = beta * v + (1 - beta) * (gradient**2)
            m_hat = m / (1 - beta ** (i + 1))
            v_hat = v / (1 - beta ** (i + 1))
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
        if use_tol and i > 0 and abs(cost_history[-2] - cost) < tol:
            print(f"{method} converged after {i} iterations.")
            break
    return theta, cost_history


# --- Part g) ---


def ols(x_train, y_train, x_eval, degree):
    model = make_pipeline(
        PolynomialFeatures(degree=degree, include_bias=True),
        LinearRegression(fit_intercept=False),
    )

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
        "train_mse": [],
    }

    for d in degrees:
        # Direct-fit (regular OLS without bootstrapping) on train/test MSE
        y_predicted_train, model = ols(x_train, y_train, x_train, degree=d)
        y_predicted_test = model.predict(x_test.reshape(-1, 1)).ravel()

        train_mse = mse(y_train, y_predicted_train)
        test_mse = mse(y_test, y_predicted_test)

        boots_predicted = np.empty((boots_reps, x_grid.size), dtype=np.float64)

        # Bootstrap, and OLS on the bootstrap values
        for b in range(boots_reps):
            idx_b = rng.choice(x_train.size, size=x_train.size, replace=True)
            xb, yb = x_train[idx_b], y_train[idx_b]
            y_pred_boot, _ = ols(xb, yb, x_grid, degree=d)
            boots_predicted[b] = y_pred_boot

        mean_boots = boots_predicted.mean(axis=0)
        var_boots = boots_predicted.var(axis=0, ddof=1)
        bias2_boots = (mean_boots - y_true) ** 2
        mse_boots = ((boots_predicted - y_true[None, :]) ** 2).mean(axis=0)

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

# K-fold cross-validation for OLS


def kfold_cv_mse_ols(degree, k, x, y, seed=seed):
    kf = KFold(n_splits=k, shuffle=True, random_state=seed)
    kfold_mses = []
    for train_idx, val_idx in kf.split(x):
        x_train_cv, x_val = x[train_idx], x[val_idx]
        y_train_cv, y_val = y[train_idx], y[val_idx]

        predicted_val, _ = ols(x_train_cv, y_train_cv, x_val, degree)
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
                base = Ridge(alpha=lam, fit_intercept=True, max_iter=300_000, tol=1e-3)
            elif method == "lasso":
                base = Lasso(
                    alpha=lam,
                    fit_intercept=True,
                    max_iter=2_000_000,
                    tol=5e-3,
                    selection="cyclic",
                )
            else:
                raise ValueError("method must be 'ridge' or 'lasso'")

            model = make_pipeline(
                PolynomialFeatures(degree, include_bias=False), StandardScaler(), base
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
