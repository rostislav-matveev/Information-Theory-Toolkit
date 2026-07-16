#!/usr/bin/python3

import numpy as np
from operatornorm  import operatornorm
from flagincidence import flagincidence
import matplotlib
matplotlib.use("GTK3Agg",warn=False, force=True)
from matplotlib import pyplot as plt

def Lmin(q,alpha,beta):
    pts = q*q + q + 1
    lns = pts
    flags = pts*(q+1)
    
    HX = np.log(pts)
    HY = np.log(lns)
    HXY = np.log(flags)
    IXY = HX + HY - HXY
    HXcY = HXY - HY
    HYcX = HXY - HX
    lmin = max(alpha*HXcY, beta*HYcX, alpha*HX + beta*HY - IXY)
    return lmin
    
def vertex(q):
    pts = q*q + q + 1
    lns = pts
    flags = pts*(q+1)
    
    HX = np.log(pts)
    HY = np.log(lns)
    HXY = np.log(flags)
    IXY = HX + HY - HXY
    HXcY = HXY - HY
    HYcX = HXY - HX
    
    alpha0 = (HYcX*IXY) / (HX*HY - IXY*IXY)    
    beta0  = (HXcY*IXY) / (HX*HY - IXY*IXY)
    value  = (HYcX*HXcY*IXY) / (HX*HY - IXY*IXY)

    return (alpha0,beta0,value)

def printest(alpha,beta,lmin,lest):
    print("α={:.2f}; β={:.2f}; Lmin(α,β)={:.5f}; Lest(α,β)={:.5f}; diff={:.1e}".
          format(alpha,beta,lmin,lest,lest-lmin))
    
def checkprofile(q,step=0.01,precision=1e-13):
    M = flagincidence(q)
    V = vertex(q)
    points = int(1/step)
    estimate = np.zeros(points*points).reshape(points,points)
    minimal  = np.zeros(points*points).reshape(points,points)
    normdiff = np.zeros(points*points).reshape(points,points)
    
    for a in range(0,points):
        alpha = a*step
        try:
            lebesgue_p = 1/(1-alpha)
        except:
            pass
        for b in range(0,points-a):
            beta = b*step
            try:
                lebesgue_q = 1/beta
            except:
                pass
            minimal[a,b]  = Lmin(q,alpha,beta)

            ### Skip the boundary cases. Set est=min
            if alpha >= 1 or alpha <= 0 or beta >= 1 or beta <= 0:
                 estimate[a,b] = minimal[a,b]
                 printest(alpha,beta,minimal[a,b],estimate[a,b])
                 continue
             
            norm = operatornorm(M,
                                lebesgue_p,
                                lebesgue_q,
                                precision,
                                start="r r r r r r r r r r r r r r r r r r r r ")
            # if max(norm)-min(norm) > 2*precision:
                # print("WARNING: Numeric estimation of norm gives inconsistent values:")
                # print("Difference={:.1e}".format(np.log(max(norm)/min(norm))))
            normdiff[a,b] = np.log(max(norm)/min(norm))
            estimate[a,b] = np.log(max(norm))
            printest(alpha,beta,minimal[a,b],estimate[a,b])
    return (minimal,estimate,normdiff)

    # if output:
        # outfile=open(output,"w")
        # outfile.write("q={}; step={}; precision={}; vertex=({},{},{})\n".
                      # format(q,step,precision,*V))
        # outfile.write("alpha, beta, Lmin, Lreal, Diff\n")
        # def sendout(alpha,beta,lmin,lreal):
            # outfile.write("{:.2f}, {:.2f}, {}, {}, {:.1e}\n".
                          # format(alpha,beta,lmin,lreal,lreal-lmin))
    # else:
        # def sendout(alpha,beta,lmin,lreal):
            # print("α={:.2f}; β={:.2f}; Lmin(α,β)={:.5f}; Lreal(α,β)={:.5f}; diff={:.1e}\n".
                  # format(alpha,beta,lmin,lreal,lreal-lmin))

def plotvslice(n,m,e,nd):
    plt.clf()
    plt.plot(m[n,:])
    plt.plot(e[n,:])
    plt.plot(nd[n,:])
    plt.show()

import json

def dumpdata(q,m,e,nd):
    with open(f"Lmin{q}","w") as fd:
        json.dump(m.tolist(),fd)
    with open(f"Lest{q}","w") as fd:
        json.dump(e.tolist(),fd)
    with open(f"Ndiff{q}","w") as fd:
        json.dump(nd.tolist(),fd)

def loaddata(q):
    with open(f"Lmin{q}","r") as fd:
        m = np.array(json.load(fd))
    with open(f"Lest{q}","r") as fd:
        e = np.array(json.load(fd))
    with open(f"Ndiff{q}","r") as fd:
        nd = np.array(json.load(fd))
    return (m,e,nd)

def exportgraphs():
    pass

### To calc, call (with q=prime)
### m,e,nd = checkprofile(q)
###
### m --- Lmin for the given q
### e --- estimates obtained by estimating the log-norm
### nd --- difference betwee highest and lowest estimates.

### To save in 3 files
### dumpdata(q,m,e,nd)

### To load
### m,e,nd = loaddata(q)

### To view vertical slices @ alpha = n*step (step=0.01):
### plotvslice(n,m,e,nd)

### Vertex:
### (α,β,Lmin(α,β)) = vertex(q)
