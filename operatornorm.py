#!/usr/bin/python3

import numpy as np
import random
from scipy.optimize import differential_evolution, minimize
from time import process_time,perf_counter


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

    def __bool__(self):
        return self.success and not np.isnan(self.norm)

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

def sortresultskey(result):
    if  not result.success or np.isnan(result.norm):
        return -np.inf
    return result.norm
        

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
        raise ValueError(message+"\n It contains non-zero/one entries.")

    u = np.ones(h,dtype=np.uint8)
    degrees = set(u@M)
    if len(degrees) != 1:
        raise ValueError(message+"\n Left degree is not constant.")
    d1 = degrees.pop()

    u = np.ones(w,dtype=np.uint8)
    degrees = set(M@u)
    if len(degrees) != 1:
        raise ValueError(message+"\n Right degree is not constant.")
    d2 = degrees.pop()
            
    return (h, w, d1, d2)

def _operatornorm_exact(M,p,q):

    cpu_start  = process_time()
    real_start = perf_counter()
    
    h, w, d1, d2 = matrix_data(M)
    p_prime = hoelder(p)
    q_prime = hoelder(q)
    
    if p == 1:
        norm = d1**(1/q)
        x = np.zeros(w,dtype=np.uint)
        x[0] = 1
        y = M @ x ### no need to find resonance, since all coords are 0,1
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
        return _OperatorNormResult(norm = norm,
                                   x = x, y = y,
                                   p = p, q = q,
                                   method = "exact",
                                   success = True,
                                   iterations = 1,
                                   message = "non-hypercontractive case",
                                   cpu_time  = process_time() - cpu_start,
                                   real_time = perf_counter() - real_start)
    
    return _FailedOperatorNormResult(method="exact",p=p,q=q,
                                     message=f"Can not use exact method for p={p}, q={q}")
       
def _operatornorm_power(M, p, q,
                        start_x,
                        epsilon=1e-15,
                        maxiter=10000):
    """Evaluates (p,q) norm of M using power method starting from x=start.
       Iterations stop when relative improvement is less then epsilon
       or after maxiter iterations.  

       Return a tuple (norm, iterations,resonance_x,resonance_y)
    """

    cpu_start  = process_time()
    real_start = perf_counter()

    ### Don't do checks for privite functions
    if p in {1,np.inf} or q in {1,np.inf}:
        return _FailedOperatorNormResult( p = p, q = q,
                                          method = "power",
                                          message = "can not use this method for p or q == np.inf or 1")

    message = ""
    Mstar = np.transpose(M)
    p_prime = hoelder(p)
    x = start_x/vectornorm(start_x,p)

    norm_prev = -1 # make sure at least one iteration happens
    y1 = M @ x
    norm = vectornorm(y1,q)
    y = np.power(y1,q-1) ### resonating to y
    x1 = Mstar @ y     ### push back
    x2 = np.power(x1, p_prime-1) ### resonating to x2
    x = x2 / vectornorm(x2,p) ### normalize

    i=0

    while (norm - norm_prev)/norm > epsilon:
        norm_prev = norm
        y = np.power(y1,q-1)
        x1 = Mstar @ y
        x2 = np.power(x1,p_prime - 1)
        x = x2 / vectornorm(x2,p)
        y1 = M @ x
        norm = vectornorm(y1,q)
        # print(i,norm,norm_prev)
        i += 1
        if i > maxiter:
            message = f"Number of iterations exceeded maxiter={maxiter}"
            print(message)
            break
        
    y = np.power(y1,q-1)
    message = "Optimization terminated successfully."
    return _OperatorNormResult( norm = norm,
                                x = x, y = y,
                                p = p, q = q,
                                method = "power",
                                success = True,
                                iterations = i,
                                message = message,
                                cpu_time  = process_time() - cpu_start,
                                real_time = perf_counter() - real_start)
    
def _operatornorm_slsqp(M, p, q,
                        start_x,
                        epsilon=1e-15,
                        maxiter=3000):
    
    cpu_start  = process_time()
    real_start = perf_counter()
    if p == np.inf or q == np.inf:
        return _FailedOperatorNormResult( p = p, q = q,
                                          method = "slsqp",
                                          message = "can not use this method for p or q = np.inf")
    def objective(x):
        return -vectornorm(M @ x, q)

    def constraint(x):
        return np.sum(np.power(x,p)) - 1.0

    constraints = {
        "type": "eq",
        "fun": constraint,
    }

    x = start_x/vectornorm(start_x,p)

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
    return _OperatorNormResult(norm = norm,
                               x = x, y = y,
                               p = p, q = q,
                               method = "slsqp",
                               success = result.success,
                               iterations = result.nit,
                               message = result.message,
                               cpu_time  = process_time() - cpu_start,
                               real_time = perf_counter() - real_start)

    
def _operatornorm_differential_evolution(M, p, q,
                                         popsize=5,
                                         epsilon=1e-15,
                                         workers=1,
                                         maxiter=1000):
    cpu_start  = process_time()
    real_start = perf_counter()

    h, w = M.shape
    
    def objective(x):
        denominator = vectornorm(x, p)

        if denominator < 10*epsilon:
            return 0.0

        return -vectornorm(M @ x, q) / denominator
    
    result = differential_evolution(objective,
                                    bounds=[(0.1, 1.0)] * w,
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
    return _OperatorNormResult(norm = norm,
                               x = x, y = y,
                               p = p, q = q,
                               method = "differential evolution",
                               method_variant = f"popsize={popsize}",
                               success = result.success,
                               iterations = result.nit,
                               message = result.message,
                               cpu_time  = process_time() - cpu_start,
                               real_time = perf_counter() - real_start)

def _parse_method(method):
    method = method.replace("("," ").replace(")"," ").replace(","," ").replace("|"," ").replace("."," ")
    mv = method.split()
    if len(mv) >= 2:
        if mv[0] == "differential_evolution":
            return [ "differential_evolution", int(mv[1]) ]
        return mv[:2]
    if len(mv) == 1:
        m = mv.pop()
        if m in {"power", "slsqp"}:
            return [m, "random"]
        if m == "exact":
            return ["exact",""]
        if m == "differential_evolution":
            return ["differential_evolution", 10]
        return [m, ""]
    if len(mv) == 0:
        return ["power", "random"]
    
def operatornorm(M,p,q,epsilon=1E-15,
                 methods = ["exact"] +
                           ["power uniform", "power coordinate"] +
                           ["power random",] * 10 +
                           ["slsqp uniform", "slsqp coordinate"] +
                           ["slsqp random",] * 10 + ["differential_evolution 10"]
                ):

    """Evaluate ||M||_{p->q}.
    epsilon -- the precision level
    

    returns list of results sorted by the decreasing value of the norm
    """
    
    if isinstance(methods,str):
        methods = [methods]
        
    h, w = M.shape
    results = list()
    
    for method in methods:
        method, variant = _parse_method(method)

        ### prepare start vector
        if method in {"power","slsqp"} and variant == "uniform":
            start_x = np.ones(w,dtype=np.uint) ### not uint8 to avoid overflows
        elif method in {"power","slsqp"} and variant == "coordinate":
            start_x = np.zeros(w,dtype=np.uint)
            start_x[random.randrange(w)] = 1
        elif method in {"power","slsqp"} and variant == "random":
            rng = np.random.default_rng()
            start_x = np.abs(rng.normal(size=w))
        elif method in {"exact","differential_evolution"}:
            pass
        else:
            raise ValueError(f"I don't understand method {method}({variant})")

        try:
            print(f"Trying method {method}({variant})") 
            if method == "exact":    
                results.append(_operatornorm_exact(M,p,q))
            elif method == "power":
                result = _operatornorm_power(M,p,q,start_x)
                result.method_variant=variant
                results.append(result)
            elif method == "slsqp":
                result = _operatornorm_slsqp(M,p,q,start_x)
                result.method_variant=variant
                results.append(result)
            elif method == "differential_evolution":
                result = _operatornorm_differential_evolution(M,p,q,popsize=variant)
                result.method_variant=f"{variant}"
                results.append(result)
            else:
                print(f"Unknown method {method}")
                results.append(_FailedOperatorNormResult(method=method,
                                                         p=p,q=q,
                                                         message=f"Unknown method {method}"))
        except Exception as e:
            results.append(_FailedOperatorNormResult(method=method,variant=variant,
                                                     p=p,q=q,
                                                     exception=e))
    results.sort(key=sortresultskey,reverse=True)
    return results
