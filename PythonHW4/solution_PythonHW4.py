# File: PythonHW4.py
# Context: CMU MSP 36601
# Author: 
# Date: 

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

# Read "hw4.csv" into "hw4"
hw4 = pd.read_csv("hw4.csv")

# Non-graphical EDA
print(hw4.shape[0], "rows and", hw4.shape[1], "columns")
print("\nColumn names:")
print(hw4.columns)
print("\nData types:")
print(hw4.dtypes)
print("\nRandom sample:")
print(hw4.sample(8))
print("\nStats for quantitative variables:")
print(hw4.describe())
print("\ntx counts:")
print(hw4['tx'].value_counts())
print("\ntx percents:")
print(round(100 * hw4['tx'].value_counts() / hw4['tx'].count(), 1))

# Perform cleanup (tx to lower case)
hw4['tx'] = hw4['tx'].map(lambda x: x.lower())
print("\ntx counts:")
print(hw4['tx'].value_counts())


# Graphical EDA
fig1 = plt.figure()
ax1 = fig1.add_subplot(111)  # 1x1 grid, 1st subplot
PF = (hw4['tx'] == 'placebo') & (hw4['female'] == 1)
pf = ax1.scatter(hw4.loc[PF, 'age'], hw4.loc[PF, 'score'], c='b', marker='^')
PM = (hw4['tx'] == 'placebo') & (hw4['female'] == 0)
pm = ax1.scatter(hw4.loc[PM, 'age'], hw4.loc[PM, 'score'], c='b', marker='h')
TF = (hw4['tx'] == 'tx1') & (hw4['female'] == 1)
tf = ax1.scatter(hw4.loc[TF, 'age'], hw4.loc[TF, 'score'], c='r', marker='^')
TM = (hw4['tx'] == 'tx1') & (hw4['female'] == 0)
tm = ax1.scatter(hw4.loc[TM, 'age'], hw4.loc[TM, 'score'], c='r', marker='h')
UF = (hw4['tx'] == 'tx2') & (hw4['female'] == 1)
uf = ax1.scatter(hw4.loc[UF, 'age'], hw4.loc[UF, 'score'], c='g', marker='^')
UM = (hw4['tx'] == 'tx2') & (hw4['female'] == 0)
um = ax1.scatter(hw4.loc[UM, 'age'], hw4.loc[UM, 'score'], c='g', marker='h')
plt.xlabel('Age')
plt.ylabel('Score')
plt.title('Python HW 4')
plt.ylim(0, 110)
plt.legend([pf, pm, tf, tm, uf, um],
           ('placebo/F', 'placebo/M', 'tx1/F', 'tx1/M', 'tx2/F', 'tx2/M'),
           loc='lower left')
fig1.savefig(filename="hw4EDA.pdf")
# Use plt.close() to dismiss the plot window, if desired

# Fit regression model(s), finding the best one
m1 = smf.ols("score ~ age + I(age**2) + age + tx*female", data=hw4).fit()
print("age^2 plus tx*gender interaction:", round(m1.bic, 1))

m2 = smf.ols("score ~ age + I(age**2) + age*tx + tx*female", data=hw4).fit()
print("age^2 plus tx I/A with age and gender:", round(m2.bic, 1))

m3 = smf.ols("score ~ age + I(age**2) + age*female + tx*female",
             data=hw4).fit()
print("age^2 plus female I/A with age and tx:", round(m3.bic, 1))

m4 = smf.ols("score ~ age + I(age**2) + age + tx + female", data=hw4).fit()
print("age^2 with 3 main effects:", round(m4.bic, 1))
print(m1.summary())

# Make a residual vs. fit plot
fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
ax2.scatter(m1.fittedvalues, m1.resid)
plt.xlabel('Fitted Values')
plt.ylabel('Residuals')
plt.title('Python HW 4 Residual Analysis')
plt.axhline(y=0.0, color='gray', alpha=0.5)
fig2.savefig("hw4Resid.pdf")


# Use numpy to verify the coefficients of the best model
# Construct a full X matrix and a y vector (nx1 matrix)
# Compute beta hat
# Print out the regression vs. linear algebra versions of the coefficients
X = np.column_stack([
        [1 for i in range(len(hw4))],
        [int(t == 'tx1') for t in hw4['tx']],
        [int(t == 'tx2') for t in hw4['tx']],
        hw4['age'],
        hw4['age']**2,
        hw4['female'],
        hw4['female']*(hw4['tx'] == 'tx1'),
        hw4['female']*(hw4['tx'] == 'tx2')])

y = hw4['score'].values
bhat = np.linalg.inv(X.transpose() @ X) @ X.transpose() @ y
print(bhat)
print(np.column_stack((m1.params, bhat)))
