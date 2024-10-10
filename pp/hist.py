import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import expon
import csv


filename = "C:\CSEXP\pp\poisson_lambda_1.000000.csv"
data = np.loadtxt(filename)

lambda_param = 1.0
x = np.linspace(0, np.max(data), 100)
pdf = lambda_param * np.exp(-lambda_param * x)
plt.plot(x, pdf, '-', color='orange', linewidth=2, label=r'True, $\lambda = 1.0$')
plt.hist(data, bins=30, density=True, alpha=0.6, color='gray', edgecolor="black", linestyle="-", label='Random values')
title = f"Histogram and Exponential PDF (lambda = {lambda_param})"
#plt.title(title)
plt.xlabel('x')
plt.ylabel('probability density')
plt.legend(loc='best')
plt.xlim(0, 5)
plt.ylim(0,lambda_param+1.0)
plt.show()