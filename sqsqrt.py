# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 07:57:09 2017

@author: hseltman
"""


def sqsqrt(val):
    """ compute square root of even numbers or squares of odds """
    import math
    even = val % 2 == 0
    if even and val < 0:
        raise Exception("Cannot square root negative numbers")
    if even:
        return math.sqrt(val)
    return val**2


if __name__ == "__main__":
    print("sqqrt(3)=", sqsqrt(3), "\n")
    print("sqqrt(4)=", sqsqrt(4), "\n")
    print("sqqrt(-3)=", sqsqrt(-3), "\n")
    print("Attempting to compute sqsqrt(-4)\n")
    print("sqqrt(-4)=", sqsqrt(-4), "\n")
