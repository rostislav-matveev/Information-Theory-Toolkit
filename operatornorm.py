#!/usr/bin/python3

import numpy as np
import random

def vectnorm(x,p):
    return sum(np.power(x,p))**(1/p)

def operatornorm(M,p,q,epsilon=1E-15,
                 tries=10, printfreq=100, iterdepth=1000):
    """Evaluate ||M||_{p->q}.
    epsilon -- the precision level
    tries   -- number of tries, each starts from a different
               random vector.

    returns tuple (length=tries) of results for all starting points.
    """

    ### Prepare ingredients

    k,l = M.shape
    Mstar = np.transpose(M)
    q1 = q - 1
    p1 = p - 1
    p1inverse = 1/p1

    
    results = dict(epsilon=epsilon,
                   tries=tries,
                   iterdepth=iterdepth,
                   norms=list(),
                   iterations=list(),
                   resonance_x=list(),
                   resonance_y=list())

    for tr in range(tries):
        print(f"Try {tr}")
        ### prepare start vector
        ### Odd tries start with random vector
        ### even tries start from a coordinate vector
        if tr%2 == 0:
            x = np.zeros(k)
            x[np.random.randint(k)] = 1
        else:
            x = np.random.rand(k)
        x = x/vectnorm(x,p)

        ### Iterate until increase in norm is less then epsilon
        ### First step
        norm_prev = -1 # make sure at least one iteration happens
        y = M @ x
        norm = vectnorm(y,q)
        
        y1 = np.power(y,q1)
        x1 = Mstar @ y1
        x2 = np.power(x1,p1inverse)
        x = x2 / vectnorm(x2,p)


        ### Iterate
        i=0
        while norm - norm_prev > epsilon:
            norm_prev = norm

            ### make new resonance pair x,y
            y1 = np.power(y,q1)
            x1 = Mstar @ y1
            x2 = np.power(x1,p1inverse)
            x = x2 / vectnorm(x2,p)
            y = M @ x
            norm = vectnorm(y,q)

            ### check iteration depth
            i += 1
            if i%printfreq == 0:
                print(f"{i}: {norm} - {norm_prev} = {norm-norm_prev} > {epsilon}")
            if i > iterdepth:
                print(f"Iterdepth ({iterdepth}) exceeded. Break...")
                break

        ### Store the results
        results["norms"].append(norm)
        results["iterations"].append(i)
        results["resonance_x"].append(x)
        results["resonance_y"].append(y)
        
    return results
