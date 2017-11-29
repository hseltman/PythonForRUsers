# Python (version =3.6) for R Users: Stat Modules I
# CMU MSP 36601, Fall 2017, Howard Seltman

import numpy as np
import pandas as pd
import random
import math
import matplotlib.pyplot as plt


# ** 1. Numpy **

# b. array() function

x = np.array([list(range(i, i+5)) for i in range(4)])
x
print(type(x), x.ndim, x.shape, sep=", ")

x.flatten()

x.transpose()
x[:, 0:4]
x[1]
x[:, 1]
x[:, 1].shape
x[:, 1].ndim

# c. Direct array creation
np.empty((2, 3))  # no guarantee on contents
np.zeros((2, 3))  # guaranteed zeros
np.identity(3)
np.ones((2, 3))
np.full((2, 3), 0.5)  # fill with your specified value


# e. Array from a function:
np.fromfunction(lambda x, y: 10*(x+1) + (y+1), (2, 3))


# f. Vector of evenly spaced values and reshaping
v = (np.arange(5, 7.6, 0.5))
v
m = v.reshape(2, 3)
m
m = v.reshape(2, 3, order="F")  # Fortran (and R) order
m

# g. Diagonals (like R, extracts a vector from a matrix or constructs a
#    matrix from a vector)
np.diag([1, 3, 4])
np.diag(np.arange(9).reshape(3, 3))


# h. The "@" operator performs matrix multiplication (also OK for an array
#    and a list):
m1 = np.array([[1, 3], [1, 1], [2, 4]])
m2 = np.array([[2, 2, 2, 2], [4, 4, 4, 4]])
m2 @ m1
m1 @ m2

# i. The transpose() method
m1.transpose()
m1.T @ m1
m1t = m1.transpose()
m1[0, 0] = 0
m1t[0, 0]

# j. Example: row and column means and sums
[sum(row) for row in m1]
m1.sum(axis=1)
[np.mean(col) for col in m1.T]
m1.mean(axis=0)

# k. Elementwise math on arrays
np.diag([1, 2, 3]) + 10
np.diag([1, 2, 3]) + np.ones((3, 3))
np.divide(np.diag([1, 2, 3]), np.ones((3, 3)))  # or use "/"
new_arr = np.empty((3, 3))
np.divide(np.diag([1, 2, 3]), np.ones((3, 3)), new_arr)
new_arr
new_arr != 0
np.greater(new_arr, np.identity(3))
a1 = np.array([[True, True], [True, False]])
a2 = np.array([[False, False], [True, False]])
a1 and a2  # fails
a1 & a2
np.array([[5, 5], [5, 0]]) & np.array([[2, 2], [2, 0]])

# l. Other linear algebra functions are available:
(m1 @ m1.T).trace()
np.dot(np.array([1, 2, 3]), np.array([4, 5, 6]))
mat = np.array([[1, 2], [3, 4]])
mat @ mat @ mat
np.linalg.matrix_power(mat, 3)


# ** 2. The pandas module **

prices = pd.Series([12.34, 5.10, 18.60, 2.50],
                   index=['A54', 'C17', 'B23', 'M17'])

# d. Create a DataFrame using the class constructor
names = ["Pooya", "Ralph", "Jihae", "Ling"]
ages = [28, 31, 24, 22]
MSP = [True, False, False, True]

pd.DataFrame([names, ages, MSP])

dtf = pd.DataFrame(list(zip(names, ages, MSP)),
                   columns=["name", "age", "MSP"])
dtf
pd.DataFrame({'name': names, 'age': ages, 'MSP': MSP})
pd.DataFrame((names, ages, MSP))  # fails
type(dtf)

# e. Save as csv
fileLoc = r"data\fakeData.csv"  # raw string
dtf.to_csv(fileLoc, index=False, header=True)

# f. Read from csv
dtf2 = pd.read_csv(fileLoc)

# g. Check Series and DataFrame info
type(prices)
len(prices)
prices.index
prices.shape
prices.get_values()
prices.values
prices.dtype

type(dtf)
dtf.dtypes

dtf.get_values()

dtf.axes
dtf.ndim
dtf.size
dtf.shape
dtf.head(2)
dtf.tail(2)
dtf.index  # R's rownames()
dtf.index = [chr(ord('A')+i) for i in range(len(dtf))]
dtf

# h. Subsetting
prices[3]
prices['M17']
prices['C17':]
prices[:'C17']  # includes C17!!!
prices[1:3]  # does not include prices[3]

dtf['age']
type(dtf['age'])
dtf['age']['C']  # or dtf['age'][2]
dtf[1]  # fails!!!
dtf[[1]]
dtf[[1, 0]]
dtf[['MSP', 'name']]

dtf[1:3]  # slices rows; excludes row 3
dtf['B':'C']  # includes row 'C'

dtf[[True, False, True, False]]
dtf['age'].max()
dtf['age'] < dtf['age'].max()
dtf[dtf['age'] < dtf['age'].max()]

# note: parentheses required and must use "&" instead of "and"
dtf[(dtf['age'] > 22) & (dtf['age'] < 30)]
dtf.age[dtf['age'] <= 28]
dtf[(dtf['age'] <= 28) & dtf.MSP][['age', 'MSP']]
# Overloaded use of "~" complement operator in pandas:
dtf[['age', 'MSP']][~((dtf['age'] <= 28) & dtf.MSP)]

dtf.name[0:2]  # not robust!
dtf.loc['B':'C']  # based on index label and inclusive!
dtf.loc['B':'C', 'age':'MSP']
dtf.loc['B':'C', ['age', 'MSP']]
dtf.loc[['B', 'D'], 'age':'MSP']
dtf.loc[:, 'age':'MSP']
dtf.loc[[True, False, False, True]]
dtf.loc[dtf.age > 28]
dtf[dtf.age > 28][['name', 'MSP']]  # two steps
dtf.loc[dtf.age > 28, ['name', 'MSP']]  # one step
dtf.iloc[[0, 3]]
dtf.iloc[dtf.age > 28]  # fails
dtf.iloc[[0, 3], 0:2]  # exclusive like [0, 2)
dtf.iloc[[i for (i, age) in zip(range(len(dtf)), dtf.age)]]
dtf.iloc[[i for (i, age) in enumerate(dtf['age']) if age < 28], 0:2]
dtf.iloc[[1, 3], ['MSP', 'age']]  # fails
dtf.iloc[[1, 3], dtf.axes[1].isin(['MSP', 'age'])]

# Deprecated method: .ix[]
dtf.ix[[0, 3], ['MSP', 'age']]
dtf.drop("MSP", axis=1)  # axis=0 drops columns by index id
dtf.filter(regex="^[a-z]")  # add axis=0 to filter rownames

temp = pd.DataFrame([[np.nan, 2, 3], [4, 5, 6]])
temp
temp.dropna(axis=0)  # return rows with no nan's
temp.dropna(axis=1)
temp = temp.fillna(999)  # perhaps useful before an export
temp

# i. Logic testing
dtf.isnull()
dtf > 28
dtf.age > 28
(dtf.age == 31) | dtf.MSP  # parentheses required; "or" fails

# j. Descriptive statistics
prices.sum()  # ignores nan's
prices.count()  # ignores nan's
dtf.count()
dtf.count(axis=1)  # 0=rows (for each column) vs. 1 columns
dtf.mean()  # runs DataFrame.mean(axis=0)

dtf['age'].mean(skipna=False)  # runs Series.mean()

#  k. Plotting of Series or DataFrame objects
dtf['age'].plot.hist()
dtf['age'].plot.hist(bins=20)
dtf.MSP.plot.pie()
dtf.age.plot.box(title='Ages')
dtf.age.plot.kde()
dtf.boxplot('age', by='MSP')
dtf.plot.scatter('age', 'MSP')


# l. Restructuring
wide = pd.DataFrame([["A", 3, 4, 5],
                     ["B", 6, 7, 8],
                     ["C", 9, 10, 11]],
                    columns=["id", "v1", "v2", "v3"])
wide
tall = pd.melt(wide, 'id')
tall
tall.pivot('id', 'variable', 'value')

# m. Adding new columns and rows
dtf['age2'] = dtf['age'] * 2
dtf['ratio'] = dtf.age2 / dtf.age
dtf
dtf.insert(2, 'score', [12, 15, 22, 11])
dtf

dtf.rename(columns={'age': 'Age'}, inplace=True)
dtf.rename(columns={'Age': 'age'}, inplace=True)

dtf['rx'] = [1, 2, 3, 1]
codes = {1: 'Placebo', 2: 'Drug A', 3: 'Drug B'}
dtf['rxc'] = dtf[['rx']].applymap(codes.get)
dtf

# equivalents to R's cbind() and rbind()
D1 = pd.DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                  columns=['c1', 'c2', 'c3'])
D2 = pd.DataFrame([[11, 12, 13], [14, 15, 16], [17, 18, 19]],
                  columns=['c1', 'c2', 'c3'])
pd.concat([D1, D2])

pd.concat([D1, D2], axis=1)

# n. The groupby() method
N = 10
dat = pd.DataFrame({'id': ["S" + str(ii) for ii in range(N)],
                    'age': [22 + random.choice(range(3)) for
                            ii in range(N)],
                    'male': [random.random() < 0.5 for ii in
                             range(N)],
                    'score': [round(random.random(), 2) for
                              ii in range(N)]})

dat
gm = dat.groupby('male')
gm
len(gm)
gm.groups
for (male, grp) in gm:
        print('-- male --' if male else '-- female --')
        print(grp)
        print("mean score is", grp.score.mean())
        print()

gm.get_group(False)

gma = dat.groupby(['male', 'age'])
gma.groups
gma.get_group((False, 24))

gm.mean()
gm['age'].mean()
gm.max()
gma.mean()

gma.agg([np.sum, np.mean, np.std])

gm.transform(lambda x: (x-x.mean())/x.std())
pd.concat([dat, gm.transform(lambda x: (x-x.mean())/x.std())],
          axis=2)

# Note the following confusing issue:
np.std(dat.age)  # numpy default is ddof=0 (pop. sd)
np.std(dat.age, ddof=1)  # sample sd
dat.age.std()  # pandas changes the ddof default to 1


def z(obj):
    if obj.dtype in (np.float64, np.int64):
        return (obj - np.mean(obj)) / np.std(obj)
    return(obj)


print(dtf.apply(z))

# o. map() for Series and apply() for DataFrame
dtf['age'].map(math.log)
dtf['age'].map(lambda x: x**2)

dtf['name'].map(str.lower)
dtf[['age', 'score', 'age2', 'ratio']].apply(np.median)

# p. Convert integer columns to float so that DataFrames will play nice
#     with some other modules
dtf.dtypes
list(dtf.dtypes == 'int64')
dtf.loc[:, dtf.dtypes == 'int64'] = \
        dtf.loc[:, dtf.dtypes == 'int64'].astype('float64')
dtf


# ** 3. matplotlib (http://matplotlib.org/) **
# Default settings of Spyder show plots immediately, which prevents adding
# legends, etc.  Try changing Tools / Preferences / IPython Console / Graphics
# to "Backend: Qt5.  (Requires IPython reset, e.g., with control-period.)
# Then plots show in a separate window, but all plt command work together.

# evenly sampled time at 200ms intervals
t = np.arange(0., 5., 0.2)  # like R's seq(0, 4.8, 0.2)

# Show red dashes, blue squares and green triangles.
# The 'format string' specifies color and line type using codes
# found in ?plt.plot.
plt.plot(t, t, 'r--', t, t**2, 'bs', t, t**3, 'g^')
plt.ylabel('score')
plt.xlabel('time')
plt.title("Powers")
plt.savefig("power.png")

plt.close()  # optional
