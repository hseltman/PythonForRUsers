# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 13:41:59 2017

@author: hseltman
"""


# Demonstrate vectorization in Python
x = [1, 5, 7, 2, 4, 5]
print(x)
print(type(x))
print(len(x))

y = [(z, z**2) for z in x if z < 7]
print(y)
print(set(y))
