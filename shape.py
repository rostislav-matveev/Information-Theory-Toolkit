#!/usr/bin/python3

import numpy as np
from operatornorm import operatornorm, vectornorm, matrix_data
# import random
# from scipy.optimize import differential_evolution, minimize
# from time import process_time,perf_counter

class Shape:
    def __init__(self,M,description="",step=0.05,paranoid=False,saveM=False,):
        self.description = description
        self.step = step
        if saveM: self.M = M
        h, w, d1, d2 = matrix_data(M)
        if paranoid and w*d1 != h*d2:
            raise ValueError("Matrix sizes do not match")
        self.X   = np.log(w)
        self.Y   = np.log(h)
        self.XlY = np.log(d2)
        self.YlX = np.log(d1)
        self.XY  = self.X + self.YlX
        self.I   = self.X + self.Y -self.XY
        self.spec= np.linalg.svd(M.astype(np.float), compute_uv=False)
        if ( paranoid and
             abs(log(self.spec[0])- 0.5*(self.XlY+self.YlX)) >= 1e-15 ):
            raise ValueError("log λ2 != L(X,Y)")
        self.shape_tbl = self._shape_tbl(M)
        self.entanglement = 2*self(1/2,1/2)-3*self(1/3,1/3)

    def min(self,alpha,beta):
        return max( alpha*self.X + beta*self.Y - self.I,
                    alpha*self.XlY,
                    beta*self.YlX )

    def max(self,alpha,beta):
        return max( alpha*self.X + beta*self.Y - self.I,
                    alpha*self.XlY+beta*self.YlX )
                    

