# Project 1 FYS-STK4155, Fall 2025
*Authors: Kjersti Stangeland, Jenny Guldvog, Ingvild Olden Bjerkelund & Sverre Johansen*

This is a collaboration repository for project 1 in FYS-STK4155. The project invlovles the assessment of regression methods, gradient descent and resampling techniques. The aim is to gain an overview of the strengths and weaknesses of the different methods, learn about their implications and test their abilities to predict the Runge function. 

### How to install required packages
A `requirements.txt` file is located in the repository. To reproduce our results, use the packages listed here. To install the packages, download the `requirements.txt` file, open your terminal and locate your project repository where you placed the downloaded file, in the command line write "´pip install -r requirements.txt´" or if you're using a conda environment type `conda install --file requirements.txt`.

### Overview of contents
The repository is organizewd as followed:
* `Code/functions.py` contains functions and modules used across the project.
* `Code/a_and_b.ipynb` makes use of the functions in `/Code/functions.py` and produces results for task a and b of the project. OLS and Ridge regression are analyzed in terms of MSE and R^2 scores. The dependence on model complexity/polynomial degree, numberof samples, and regularization parameter are assessed.
* `Code/c_and_d.ipynb` makes use of the functions in `/Code/functions.py` and produces results for task c and d of the project. OLS and Ridge regression are analyzed comparing analytical and gradient descent methods. Gradient descent methods with adaptive learning rate is also introduced and used to investigate the preformance of each method. 
* `Code/c_and_d_and_e.ipynb` produces some results for task c, d, and e, including Lasso.
* `Code/f.ipynb` makes use of functions in `Code/functions.py` and produces results for task f in the project. Stochastic gradient descent with optimization methods.
* `Code/g_and_h.ipynb` makes use of the functions in `/Code/functions.py` and produces results for task g and h of the project. OLS, Ridge and Lasso regression are analyzed using Bootstrap and K-fold cross-validation resampling techiniques. The dependence on model complexity/polynomial degree, number of samples, and regularization parameter are assessed.
