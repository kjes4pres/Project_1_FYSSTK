def stochastic_gradient_descent_advanced(X, y, method='gd', lr_method='ols', learning_rate=0.01, n_iterations=1000, tol=1e-6, use_tol=False, beta=0.9, epsilon=1e-8, lambda_=0.01):
    n_samples, n_features = X.shape
    theta = np.zeros(n_features)
    batch_size = 10
    cost_history = []   
    m = np.zeros(n_features)  # For momentum and Adam
    v = np.zeros(n_features)  # For Adam 
    for i in range(n_iterations):
        init_pos = np.random.choice(np.arange(0,n_samples-batch_size),1)
        X_, y_ = X[init_pos[0]:init_pos[0] + batch_size], y[init_pos[0]:init_pos[0] + batch_size]
        if lr_method == 'ols':
            gradient = ols_gradient(X_, y_, theta)
        elif lr_method == 'ridge':
            gradient = ridge_gradient(X_, y_, theta, lambda_=lambda_)
        elif lr_method == "lasso":
            gradient = lasso_gradient(X_,y_, theta, lambda_)
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
            cost = (1/n_samples) * np.sum((X_ @ theta - y_)**2)
        elif lr_method == 'ridge':
            cost = (1/n_samples) * np.sum((X_ @ theta - y_)**2) + lambda_ * np.sum(theta**2)
        elif lr_method == 'lasso':
            cost = (1/n_samples) * np.sum((X_ @ theta - y_)**2) + lambda_ * np.sum(np.abs(theta))
        cost_history.append(cost)
        if use_tol and i > 0 and abs(cost_history[-2] - cost) < tol:
            print(f"{method} converged after {i} iterations.")
            break
    return theta, cost_history



x = np.linspace(-1,1)
