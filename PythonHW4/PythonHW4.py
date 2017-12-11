# File: PythonHW4.py
# Context: CMU MSP 36601
# Author: 
# Date: 

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

# Read "hw4.csv" into "hw4"
hw4 = 

# Non-graphical EDA
# >> Use print() to assure that I will see your output in batch mode <<


# Perform cleanup (tx to lower case)


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


# Fit regression model(s), finding the best one using BIC


# Make a residual vs. fit plot
fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
plt.axhline(y=0.0, color='gray', alpha=0.5)
fig2.savefig("hw4Resid.pdf")


# Use numpy to verify the coefficients of the best model
# Construct a full X matrix and a y vector (nx1 matrix)
# Compute beta hat
# Print out the regression vs. linear algebra versions of the coefficients
X = np.column_stack((
        
    ))

y = hw4['score'].values
bhat = 
print(np.column_stack((m1.params, bhat)))
