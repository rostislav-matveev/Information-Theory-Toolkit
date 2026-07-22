#!/usr/bin/python3

import numpy as np
from incidence import *
from shape import Shape
from persistent import persistent_dict

import matplotlib
matplotlib.use("GTK3Agg",warn=False, force=True)
from matplotlib import pyplot as plt

def slice(f,start,direction,step=0.05,ranges= ((0,1),(0,1)) ):
    alpha = start[0]
    beta  = start[1]
    result=list()
    while ( ranges[0][0] <= alpha <= ranges[0][1] and
            ranges[1][0] <= beta  <= ranges[1][1]     ):
        result.append(f(alpha,beta))
        alpha += step*direction[0]
        beta  += step*direction[1]
    return result

def plot(*fcns):
    plt.clf()
    for f in fcns:
        plt.plot(f)
    plt.show()

def plotslice(s,start,direction,step=None):
    if step == None:
        step = s.step
    s_est = slice( s,    start,direction,step=step, ranges= ((0,1),(0,1)) )
    s_min = slice( s.min,start,direction,step=step, ranges= ((0,1),(0,1)) )
    s_max = slice( s.max,start,direction,step=step, ranges= ((0,1),(0,1)) )
    plot(s_est, s_max, s_min )
    
class tmpcls:
    def __init__(self):
        self.m = self._m
        
    def _m(self,*args,**kwargs):
        print(self)
        print(args)
        print(kwargs)
        
