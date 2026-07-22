#!/usr/bin/python3.11

import logging
import numpy as np
import random
from scipy.optimize import differential_evolution, minimize
from time import process_time,perf_counter

msg = logging.getLogger(__name__)
msg.setLevel(logging.INFO)


class _OperatorNormResult:
    def __init__( self, norm, x, y, p, q, method,success,
                  cpu_time=0, real_time=0, method_variant="",
                  iterations=1, message="", exception=None ):
        self.norm=norm
        self.x=x
        self.y=y
        self.p=p
        self.q=q
        self.method=method
        self.success=success
        self.cpu_time=cpu_time
        self.real_time=real_time
        self.method_variant=method_variant
        self.iterations=iterations
        self.message=message
        self.exception=exception

    def transpose(self):
        x = self.x
        self.x = self.y
        self.y = x
        p = self.p
        self.p = hoelder(self.q)
        self.q = hoelder(p)
        self.method_variant +=",adjoint"

    adjoint = transpose
    
    def __bool__(self):
        # return self.success and not np.isnan(self.norm)
        ### some methods return value even with a problem like
        ### maxiter exceeded. Don't check for success, just whether
        ### there is a numeric value.
        return not np.isnan(self.norm)
    
    def __lt__(self,other):
        s = -np.inf if not self  else self.norm
        o = -np.inf if not other else other.norm
        return s < o
    
    def __le__(self,other):
        s = -np.inf if not self  else self.norm
        o = -np.inf if not other else other.norm
        return s <= o
    
    def __gt__(self,other):
        s = -np.inf if not self  else self.norm
        o = -np.inf if not other else other.norm
        return s > o
    
    def __ge__(self,other):
        s = -np.inf if not self  else self.norm
        o = -np.inf if not other else other.norm
        return s >= o
    
    def __eq__(self,other):
        s = -np.inf if not self  else self.norm
        o = -np.inf if not other else other.norm
        return s == o
    
    def __ne__(self,other):
        s = -np.inf if not self  else self.norm
        o = -np.inf if not other else other.norm
        return s != o
    
    def __repr__(self):
        if self.success and not np.isnan(self.norm):
            status = "successful"
        else:
            status = "failed"
        return "\n".join([ "\n---------------------",
                           f"norm:\t\t{self.norm}",
                           f"(p,q):\t\t({self.p:3.2f},{self.q:3.2f})",
                           f"Method:\t\t{self.method}({self.method_variant}); {status}",
                           f"iterations:\t{self.iterations}",
                           f"time cpu/re:\t{self.cpu_time:1.4f}/{self.real_time:1.4f}",
                           f"message:\t{self.message}",
                           f"exception:\t{self.exception}",
                           "---------------------"])

    __str__ = __repr__
    
class _FailedOperatorNormResult(_OperatorNormResult):
    def __init__(self, method, p, q, cpu_time=0, real_time = 0,
                 variant="", message="", exception=None):
        self.norm = np.nan
        self.x = None
        self.y = None
        self.iterations = 0
        self.p = p
        self.q = q
        self.cpu_time = cpu_time
        self.real_time = real_time
        self.method = method
        self.method_variant = variant
        self.success = False
        self.message = message
        self.exception=exception

def vectornorm(x,p):
    return sum(np.power(x,p))**(1/p)

def hoelder(p):
    ### TODO:check range
    if p == 1:
        return np.inf
    if p == np.inf:
        return 1
    return p/(p-1)

def matrix_data(M):
    """find the dimensions and degrees of M"""
    message = "Matrix does not appear to be an incidence matrix of a biregular bipartite graph."

    h,w = M.shape
    entries = set(M.reshape( (h*w,) ))
    
    if not (entries == {0,1} or entries == {1,} ):
        msg.critical("Matrix contains non-zero/one entries")
        raise ValueError(message+"\n It contains non-zero/one entries")

    u = np.ones(h,dtype=np.uint32)
    degrees = set(u@M)
    if len(degrees) != 1:
        msg.critical("Matrix left degree is not constant")
        raise ValueError(message+"\n Left degree is not constant")
    d1 = int(degrees.pop())

    u = np.ones(w,dtype=np.uint32)
    degrees = set(M@u)
    if len(degrees) != 1:
        msg.critical("Matrix right degree is not constant")
        raise ValueError(message+"\n Right degree is not constant.")
    d2 = int(degrees.pop())
            
    return (h, w, d1, d2)



def operatornorm_minimal(M,p,q):

    cpu_start  = process_time()
    real_start = perf_counter()
    ### TODO: check params
    
    msg.debug("Trying method minimal")
    h, w, d1, d2 = matrix_data(M)
    
    norm = max( d1 * w**(1-1/p) / (h**(1-1/q)), 
                d2**(1-1/p),
                d1**(1/q)                    )
    
    return _OperatorNormResult(norm = norm,
                               x = None, y = None,
                               p = p, q = q,
                               method = "minimal",
                               success = True, 
                               iterations = 1,
                               message = "Lower bound from operatornorm_minimal",
                               cpu_time  = process_time() - cpu_start,
                               real_time = perf_counter() - real_start)

def operatornorm_exact(M,p,q):

    cpu_start  = process_time()
    real_start = perf_counter()
    ### TODO: check params
    
    msg.debug("Trying method exact")
    h, w, d1, d2 = matrix_data(M)
    p_prime = hoelder(p)
    q_prime = hoelder(q)
    
    if p == 1:
        norm = d1**(1/q)
        x = np.zeros(w,dtype=np.uint)
        x[0] = 1
        y = M @ x ### no need to find resonance, since all coords are 0,1
        msg.debug(f"Value: {norm}")
        return _OperatorNormResult(norm = norm,
                                   x = x, y = y,
                                   p = p, q = q,
                                   method = "exact",
                                   success = True, 
                                   iterations = 1,
                                   message = "Boundary case",
                                   cpu_time  = process_time() - cpu_start,
                                   real_time = perf_counter() - real_start)
    if p == np.inf:
        norm = d2 * (h**(1/q)),
        x = np.ones(w,dtype=np.uint)
        y = np.ones(h,dtype=np.uint)
        msg.debug(f"Value: {norm}")
        return _OperatorNormResult(norm = norm,
                                   x = x, y = y,
                                   p = p, q = q,
                                   method = "exact",
                                   success = True,
                                   iterations = 1,
                                   message = "Boundary case",
                                   cpu_time  = process_time() - cpu_start,
                                   real_time = perf_counter() - real_start)

    if q == 1:
        norm = d1 * (w**(1/p_prime))
        x = np.ones(w,dtype=np.uint)
        y = np.ones(h,dtype=np.uint)
        msg.debug(f"Value: {norm}")
        return _OperatorNormResult(norm = norm,
                                   x = x, y = y,
                                   p = p, q = q,
                                   method = "exact",
                                   success = True,
                                   iterations = 1,
                                   message = "Boundary case",
                                   cpu_time  = process_time() - cpu_start,
                                   real_time = perf_counter() - real_start)

    if q == np.inf:
        norm = d2**(1/p_prime)
        y = np.zeros(h,dtype=np.uint)
        y[0] = 1
        x =y @ M ### no need to find resonance, since all coords are 0,1
        msg.debug(f"Value: {norm}")
        return _OperatorNormResult(norm = norm,
                                   x = x, y = y,
                                   p = p, q = q,
                                   method = "exact",
                                   success = True,
                                   iterations = 1,
                                   message = "Boundary case",
                                   cpu_time  = process_time() - cpu_start,
                                   real_time = perf_counter() - real_start)
        
    if q <= p:
        norm = d2 * (h**(1/q)) / (w**(1/p))
        x = np.ones(w,dtype=np.uint)
        y = np.ones(h,dtype=np.uint)
        msg.debug(f"Value: {norm}")
        return _OperatorNormResult(norm = norm,
                                   x = x, y = y,
                                   p = p, q = q,
                                   method = "exact",
                                   success = True,
                                   iterations = 1,
                                   message = "non-hypercontractive case",
                                   cpu_time  = process_time() - cpu_start,
                                   real_time = perf_counter() - real_start)
    
    msg.error(f"Exact method failed")
    return _FailedOperatorNormResult(method="exact",p=p,q=q,
                                     message=f"Can not use exact method for p={p}, q={q}")
_rng = np.random.default_rng()

def _get_start_x(start_x,length):
    if not isinstance(start_x,str):
        return ( start_x, "custom" )
    
    start_x = start_x.lower().strip()
    
    if start_x in {"random","rand", "rnd","r"}:
        return ( np.abs(_rng.normal(size=length)), "random" )
    
    if start_x in {"coordrnd", "coordrand", "cr", "coordrandom", "crandom" }:
        x = np.zeros(length,dtype=np.float64)
        x[_rng.choice(length)] = 1
        return (x, "crandom" )
    
    if start_x in {"u", "unif", "uniform" }:
        return (np.ones(length,dtype=np.float64), "crandom" )

    message = f"Do not understand start_x={start_x}" 
    msg.critical(message)
    raise ValueError(message)

def operatornorm_power(M, p, q,
                       start_x,
                       epsilon=1e-15,
                       maxiter=10000):
    """Evaluates (p,q) norm of M using power method starting from x=start.
       Iterations stop when relative improvement is less then epsilon
       or after maxiter iterations.  

       Return a tuple (norm, iterations,resonance_x,resonance_y)
    """

    ### I will use ell^infty normalization on every step. It seems sensible. 
    cpu_start  = process_time()
    real_start = perf_counter()

    ### TODO: check params

    x, variant = _get_start_x( start_x, M.shape[1] )
                  
    msg.debug(f"Trying method power({variant})")
    if p in {1,np.inf} or q in {1,np.inf}:
        msg.error(f"Power method does not work for (p,q)=({p}.{q}")
        return _FailedOperatorNormResult( p = p, q = q,
                                          method = "power",
                                          method_variant = variant,
                                          message = "can not use this method for p or q == np.inf or 1")

    message = ""
    Mstar = np.transpose(M)
    p_prime = hoelder(p)
    q_prime = hoelder(q)

    ### Because there are big powers we keep 0 <= x <= 1, as close to
    ### 1 as possible
    x = x/np.max(x)

    norm_prev = -1 # make sure at least one iteration happens
    y1 = M @ x
    norm = vectornorm(y1,q)/vectornorm(x,p)

    i=0

    ### compare on log scale
    while (norm - norm_prev)/norm > epsilon:
        norm_prev = norm

        ### q could be very large. So we try to ell^infty-renormalize
        ### y1 so that thing don't go out of control
        y2 = y1/np.max(y1)
        y = np.power(y2,q-1)
        x1 = Mstar @ y
        ### ell^infty renormalize x1 in case p_prime is big
        x2 = x1/np.max(x1)
        x = np.power(x2,p_prime - 1)
        y1 = M @ x
        norm = vectornorm(y1,q)/vectornorm(x,p)
        i += 1
        msg.debug(f"{i:>4}. Norm:{norm}")
        # breakpoint()
        if i > maxiter:
            ### keep the message to use it in the result.
            message = f"Number of iterations exceeded maxiter={maxiter}"
            msg.warning(message)
            msg.debug(f"Value: {norm}")
            ### prepare y for the result
            y2 = y1/np.max(y1)
            y  = np.power(y2,q-1)
            return _OperatorNormResult( norm = norm,
                                        x = x, y = y,
                                        p = p, q = q,
                                        method = "power",
                                        method_variant = variant,
                                        success = False,
                                        iterations = i,
                                        message = message,
                                        cpu_time  = process_time() - cpu_start,
                                        real_time = perf_counter() - real_start)

    if np.isnan(norm):
        message = "Calculation failed, got nan. Probably because exponents are too big"
    else:
        message = "Optimization terminated successfully."
    msg.debug(f"Value: {norm}")
    ### prepare y for the result
    y2 = y1/np.max(y1)
    y  = np.power(y2,q-1)
        
    return _OperatorNormResult( norm = norm,
                                x = x, y = y,
                                p = p, q = q,
                                method = "power",
                                method_variant = variant,
                                success = not np.isnan(norm),
                                iterations = i,
                                message = message,
                                cpu_time  = process_time() - cpu_start,
                                real_time = perf_counter() - real_start)
    
def operatornorm_slsqp(M, p, q,
                        start_x,
                        epsilon=1e-15,
                        maxiter=3000):
    
    cpu_start  = process_time()
    real_start = perf_counter()
    
    x, variant = _get_start_x( start_x, M.shape[1] )
    x = x/vectornorm(x,p)
    
    msg.debug(f"Trying method power({variant})")

    if p == np.inf or q == np.inf:
        msg.error(f"Slspq method does not work for (p,q)=({p},{q}")
        return _FailedOperatorNormResult( p = p, q = q,
                                          method = "slsqp",
                                          method_variant = variant,
                                          message = "can not use this method for p or q = np.inf")
    
    def objective(x):
        return -vectornorm(M @ x, q)

    def constraint(x):
        return np.sum(np.power(x,p)) - 1.0

    constraints = {
        "type": "eq",
        "fun": constraint,
    }

    result = minimize(objective, x, 
                      method="SLSQP",
                      constraints=constraints,
                      options={
                          "maxiter": maxiter,
                          "ftol": epsilon,
                          "disp": False})

    x = result.x/vectornorm(result.x, p)
    y1 = M @ x
    norm = vectornorm(y1, q)
    y = np.power(y1, q-1)

    msg.debug(f"Value: {norm}")
    return _OperatorNormResult(norm = norm,
                               x = x, y = y,
                               p = p, q = q,
                               method = "slsqp",
                               method_variant = variant,
                               success = result.success,
                               iterations = result.nit,
                               message = result.message,
                               cpu_time  = process_time() - cpu_start,
                               real_time = perf_counter() - real_start)

    
def operatornorm_differential_evolution(M, p, q,
                                         popsize=10,
                                         epsilon=1e-15,
                                         workers=1,
                                         maxiter=5000):
    cpu_start  = process_time()
    real_start = perf_counter()
    msg.debug(f"Trying method diff_evo(popsize={popsize})")
    h, w = M.shape
    
    def objective(x):
        denominator = vectornorm(x, p)

        if denominator < 10*epsilon:
            return 0.0

        return -vectornorm(M @ x, q) / denominator
    
    result = differential_evolution(objective,
                                    bounds=[(0.0, 1.0)] * w,
                                    maxiter=maxiter,
                                    popsize=popsize,
                                    tol=epsilon,
                                    polish=True,
                                    workers=workers,
                                    updating="immediate" if workers == 1 else "deferred"
                                    )
    x = result.x/vectornorm(result.x, p)
    y1 = M @ x
    norm = vectornorm(y1, q)
    y = np.power(y1,q-1)
    msg.debug(f"Value: {norm}")
    return _OperatorNormResult(norm = norm,
                               x = x, y = y,
                               p = p, q = q,
                               method = "differential evolution",
                               method_variant = f"popsize={popsize}",
                               success = result.success,
                               iterations = result.nit,
                               message = result.message,
                               cpu_time  = process_time() - cpu_start,
                               real_time = perf_counter() - real_start )

