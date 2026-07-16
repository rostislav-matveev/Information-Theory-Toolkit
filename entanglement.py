#!/usr/bin/python3

import numpy as np
from incidence import projflag_incidence, cycle_incidence, ham_incidence
from operatornorm  import operatornorm

class container:
    pass

def entanglement(M,check=True):
    eXY = np.sum(M)
    eX, eY = M.shape
    
    if check:
        for i in range(eX):
            if eXY != eX * np.sum(M[i,:]):
                raise Exception("Incidence format")
        for i in range(eY):
             if eXY != eY * np.sum(M[:,i]):
                raise Exception("Incidence format")

    XY = np.log(eXY)
    X  = np.log(eX)
    Y  = np.log(eY)
    XiY = X + Y - XY
    XlY = XY - Y
    YlX = XY - X
    Emax = min(XiY,XlY,YlX)
    Norms = operatornorm(M,3/2,3)
    Norm  = max(Norms["norms"])
    Nmax  = np.log(Norm)
    Nmin  = np.log(min(Norms["norms"]))
    gap   = Nmax - Nmin
    E     = (XlY + YlX) - 3 * Nmax
    Ebar = E / Emax
    print(f"\n[X]\t=\t{X}")
    print(f"[Y]\t=\t{Y}")
    print(f"[XY]\t=\t{XY}")
    print(f"[X|Y]\t=\t{XlY}")
    print(f"[Y|X]\t=\t{YlX}")
    print(f"[X:Y]\t=\t{XiY}")
    print(f"[E_max]\t=\t{Emax}")
    print(f"Gap\t=\t{gap}")
    print(f"||M||_(3/2,3)=\t{Norm}")
    print(f"S(1/3,1/3)=\t{Nmax}")
    print(f"E\t=\t{E}")
    print(f"Ebar\t=\t{Ebar}")
    print(f"\nM & {E} & {Emax} & {Ebar}")
    result = container()
    result.norms = Norms
    result.X = X
    result.Y = Y
    result.XiY = XiY
    result.XlY = XlY
    result.YlX = YlX
    result.Emax = Emax
    result.Norm = Norm
    result.E = E
    result.Ebar = Ebar
    return result


def ham_ent_tbl(N):
    result = container()
    K = N//2
    result.X    = np.zeros((N-1,K))
    result.XlY  = np.zeros((N-1,K))
    result.XiY  = np.zeros((N-1,K))
    result.Emax = np.zeros((N-1,K))
    result.E    = np.zeros((N-1,K))
    result.Ebar = np.zeros((N-1,K))
    for n in range(2,N+1):
        for k in range(1,n//2+1):
            M = ham_incidence(n,k)
            ent = entanglement(M)
            result.X[n-2,k-1] = ent.X
            result.XlY[n-2,k-1] = ent.XlY
            result.XiY[n-2,k-1] = ent.XiY
            result.Emax[n-2,k-1] = ent.Emax
            result.E[n-2,k-1] = ent.E
            result.Ebar[n-2,k-1] = ent.Ebar
    return result

def ham_tbl(N):
    # result   = container()
    # result.M = dict()
    # result.E = dict()
    result = dict()
    for n in range(2,N+1):
        for k in range(1,n//2+1):
            print("================",(n,k),"==================")
            # result.M[(n,k)] = ham_incidence(n,k)
            # result.E[(n,k)] = entanglement(result.M[(n,k)])
            M = ham_incidence(n,k)
            result[(n,k)] = entanglement(result.M[(n,k)])
    return result
