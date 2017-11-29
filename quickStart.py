# This is quickStart.py
# Declare extensions to the core language:
import numpy as np
import pandas as pd
import random
import statsmodels.formula.api as smf

# Create subject characteristics for simulated data 
N = 20  # N <= 26
# First create lists as "list comprehensions"
id = [chr(ord('A') + ii) for ii in range(N)]
ages = [random.choice(list(range(20, 35))) for ii in range(N)]
male = [0 if random.random() < 0.5 else 1 for ii in range(N)]
data = pd.DataFrame({"id": id, "ages": ages, "male": male},
                    index=['P' + str(ii) for ii in range(N)])
print(data)
print(type(data))
print(type(data["ages"]))  # columns are of class "Series"

# Generate simulated regression data
b = (3.0, 2.5, -1.4)
k = len(b) - 1
sig = 2.5
X = [[1] + [random.normalvariate(0, 1) for x in range(k)]
     for y in range(N)]
y = [random.normalvariate(np.dot(x, b), sig) for x in X]
reg = np.array([ivs[1:] + [dv] for ivs, dv in zip(X, y)])
colnames = ["x" + str(1 + ii) for ii in range(k)] + ["y"]

# Create a DataFrame from a numpy array
reg = pd.DataFrame(reg, columns=colnames)

# (inefficient) matrix approach to regression
Xa = np.array(X)
bhat = np.linalg.inv(Xa.transpose() @ X) @ Xa.transpose() @ y
print(bhat)

# Regression with module 'statsmodels'
m0 = smf.ols('y ~ x1 + x2', data=reg, hasconst=True).fit()
print(m0.summary())
