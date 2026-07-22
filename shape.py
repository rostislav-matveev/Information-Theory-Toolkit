#!/usr/bin/python3.11

import logging
import numpy as np
import operatornorm
from operatornorm import * 

logFormatter = logging.Formatter("[%(levelname)-1.1s]:%(funcName)-10.10s:%(relativeCreated)-5d: %(message)s")
consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(logFormatter)
rootlogger = logging.getLogger()
rootlogger.addHandler(consoleHandler)
msg = logging.getLogger(__name__)
msg.setLevel(logging.INFO)

class _Data:
    def __init__(self,**kwargs):
        self.__dict__=kwargs


class Shape:
    def __init__(self,M,steps,
                 description="",
                 maxiter = 100,
                 miniter = 20,
                 assume_symmetric=False,
                 assume_homogeneous=False,
                 use_minimal=True,
                 save_xy = True,
                 save_stats = False,
                 paranoid=True):

        self.description = description
        self.steps = steps
        self.step = 1/steps
        self.M = M

        h, w, d1, d2 = matrix_data(M)

        ### Cancel assume_symmetric if matrix is not square.
        if assume_symmetric and h!=w:
            msg.warning(f"Matrix is not square ({h}x{w}). I set assume_symmetric = False")
            assume_symmetric = False
            
        ### Prepare storage for xy and stats.
        if save_xy:
            self.xy = dict()
            self._save_xy = self._save_xy_fcn
        else:
            self._save_xy = self._relax

        if save_stats:
            self.stats = dict()
            self._save_stats = self._save_stats_fcn
        else:
            self._save_stats = self._relax
            
        ### store these params for access during init
        self._options = _Data( maxiter=maxiter,
                               miniter=miniter,
                               assume_symmetric=assume_symmetric,
                               assume_homogeneous=assume_homogeneous,
                               use_minimal = use_minimal,
                               save_xy=save_xy,
                               save_stats=save_stats,
                               paranoid=paranoid )
        
        self.X   = np.log(w)
        self.Y   = np.log(h)
        self.XlY = np.log(d2)
        self.YlX = np.log(d1)
        self.XY  = self.X + self.YlX
        self.I   = self.X + self.Y -self.XY
        self.spec= np.linalg.svd(M.astype(np.float64), compute_uv=False)

        if ( paranoid and
             abs(np.log(self.spec[0])- 0.5*(self.XlY+self.YlX)) >= 1e-3 ):
            raise ValueError("log(λ1) != L(X,Y). Check calc and matrix.")

        msg.debug(f"Evaluating shape for {description}...")
        self._compute_shape_tbl(M)
        msg.debug(f"Done evaluating shape for {description}")
        ### we calculate value at (1/3,1/3) separately, because it is
        ### importnat for entanglement.
        self.E = self.XlY + self.YlX - 3*self._compute_shape(1/3,1/3)
        self.Ebar = self.entanglement/min(self.XlY,self.YlX,self.I)
        denominator = self.X*self.Y - self.I*self.I
        if denominator > 1e-14:
            self.alpha0 = self.YlX*self.I / denominator
            self.beta0  = self.XlY*self.I / denominator
            self.S0     = self._compute_shape(self.alpha0,self.beta0)
        else:
            self.alpha0 = nan
            self.beta0  = nan
            self.S0     = nan
        self.Smax0 = self.max(self.alpha0,self.beta0)
        self.Smin0 = self.min(self.alpha0,self.beta0)
        
        self.R   = (self.Smax0-self.S0) / (self.Smax0-self.Smin0)
        maxdeg     = max(self.XlY,self.YlX)
        loglambda1 = (self.XlY+self.YlX)/2
        loglambda2 = np.log(self.spec[1])
        self.Expander = (loglambda1 - loglambda2) / (loglambda1 - maxdeg/2)
        self.Expander_sym = (loglambda1 - loglambda2) / (loglambda1 - loglambda1/2)
        
    def _relax(*args,**kwargs):
        pass

    def _save_xy_fcn(self,alpha,beta,val):
        self._xy[alpha,beta]=val
    
    def _save_stats_fcn(self,alpha,beta,val):
        if not (alpha,beta) in self._stats.keys(): self._stats[alpha,beta]=list()
        self._stats[alpha,beta].append(val)

    def min(self,alpha,beta):
        return max( alpha*self.X + beta*self.Y - self.I,
                    alpha*self.XlY,
                    beta*self.YlX )

    def max(self,alpha,beta):
        return max( alpha*self.X + beta*self.Y - self.I,
                    alpha*self.XlY+beta*self.YlX )

    def _compute_shape(self,alpha,beta):
        p = 1/(1-alpha)
        p_prime = hoelder(p)

        q = 1/beta
        q_prime = hoelder(q)

        M_star = np.transpose(self.M)
        
        ### if homogeneous, it doesn't make sense to start with
        ### coordinate vector more then once.
        start_xs = ["coordrnd"]*2 + ["uniform"]*2
        if self._options.assume_homogeneous:
            start_xs +=  ["random"]*self._options.maxiter
        else:
            start_xs += ["coordrnd", "coordrnd", "random", "random"]*(self._options.maxiter//4 + 3)
        n_try = 0
        n_notbetter = 0
        success = 0
        logNmax = -np.inf

        ### Add lower bound.
        if self._options.use_minimal:
            ### Don't include this in trials count
            r = operatornorm_minimal(self.M,p,q)
            logN = np.log(r.norm)
            msg.debug( f"({alpha:>4.2f},{beta:>4.2f})(m ); {n_try:>2}/{n_notbetter:<2}; "
                       f"Better: {logN:>8.6f} >  {logNmax:<8.6f}; Gap: {logN - logNmax:<7.1e}")
            self._save_stats(alpha,beta,r)
            self._save_xy(alpha,beta,r)
            logNmax = logN
                
        while ( n_try       < self._options.maxiter and
                n_notbetter < self._options.miniter ):
            
            ### Direct
            n_try += 1
            r = operatornorm_power(self.M,p,q, start_x=start_xs[n_try-1])
            ### label for printing [m|c|u|r][*| ]
            s = start_xs[n_try-1][0] + " "
            
            self._save_stats(alpha,beta,r)

            if r:
                success += 1
                logN = np.log(r.norm)

                if logN > logNmax:
                    n_notbetter = 0
                    msg.debug( f"({alpha:>4.2f},{beta:>4.2f})({s}); {n_try:>2}/{n_notbetter:<2}; "
                               f"Better: {logN:>8.6f} >  {logNmax:<8.6f}; Gap: {logN - logNmax:<7.1e}")
                    logNmax = logN
                    self._save_xy(alpha,beta,r)
                else:
                    n_notbetter += 1
                    msg.debug( f"({alpha:>4.2f},{beta:>4.2f})({s}); {n_try:>2}/{n_notbetter:<2}; "
                               f"Worse:  {logN:>8.6f} <= {logNmax:<8.6f};")
            else:
                msg.debug(f"({alpha:>4.2f},{beta:>4.2f})({s}); {n_try:>2}/{n_notbetter:<2}; Fail.")
                
            ### Adjoint
            n_try += 1
            r = operatornorm_power(M_star,q_prime,p_prime, start_x=start_xs[n_try-1])
            r.transpose()
            s = start_xs[n_try-1][0] + "*"
            
            self._save_stats(alpha,beta,r)

            if r:
                success += 1
                logN = np.log(r.norm)

                if logN > logNmax:
                    n_notbetter = 0
                    msg.debug( f"({alpha:>4.2f},{beta:>4.2f})({s}); {n_try:>2}/{n_notbetter:<2}; "
                               f"Better: {logN:>8.6f} >  {logNmax:<8.6f}; Gap: {logN - logNmax:<7.1e}")
                    logNmax = logN
                    self._save_xy(alpha,beta,r)
                else:
                    n_notbetter += 1
                    msg.debug( f"({alpha:>4.2f},{beta:>4.2f})({s}); {n_try:>2}/{n_notbetter:<2}; "
                               f"Worse:  {logN:>8.6f} <= {logNmax:<8.6f};")
            else:
                msg.debug(f"({alpha:>4.2f},{beta:>4.2f})({s}); {n_try:>2}/{n_notbetter:<2}; Fail.")
                
        if success:
            return logNmax
        else:
            msg.error(f"({alpha:>4.2f},{beta:>4.2f}): Could not evaluate norm. All attempts fail.")
            msg.error(f"({alpha:>4.2f},{beta:>4.2f}): Check self._stats[{alpha},{beta}].")
            return np.nan
        
    def _compute_shape_tbl(self,M):

        ### initialize with nans
        self.shape_tbl = np.ndarray((self.steps+1,self.steps+1),dtype=np.float64)
        self.shape_tbl[:] = np.nan

        msg.debug("Fill the non-hypercontractive triangle and boundary.")
        ### Fill the bounfary and affine part in the upper triangle
        ### Shall I do that?
        self.shape_tbl[0,0]=0
        for i in range(1,self.steps+1):    
            self.shape_tbl[i,0] = i*self.step*self.XlY
            self.shape_tbl[0,i] = i*self.step*self.YlX
            for j in range(self.steps-i,self.steps+1):
                self.shape_tbl[i,j] = i*self.step*self.X + j*self.step*self.Y - self.I

        msg.debug("Fill the hypercontractive triangle.")
        ### Fill the hypercontractive cases.
        ### Do not skip the last column (i=self.steps-1) for clarity
        for i in range(1,self.steps):
            for j in range(1,self.steps-i):
                if self._options.assume_symmetric and not np.isnan(self.shape_tbl[j,i]):
                    self.shape_tbl[i,j]=self.shape_tbl[j,i]
                    msg.info( f"[{i:>2},{j:>2}] "
                              f"shape_tbl[{i:>2},{j:>2}] = shape_tbl[{j:>2},{i:>2}] = "
                              f"{self.shape_tbl[i,j]:<8.6f}" )
                    continue
                self.shape_tbl[i,j] = self._compute_shape(i*self.step,j*self.step)
                msg.info(f"[{i:>2},{j:>2}] shape_tbl[{i:>2},{j:>2}] = {self.shape_tbl[i,j]:<8.6f}")
        

    def __call__(self,alpha,beta):
        ### TODO: check params
        if ( not 0 <= alpha <= 1 or not 0 <= beta <= 1 ):
            raise ValueError(f" ({alpha},{beta}) not in [0,1]^2")

        ### Do quadratic fit? Worth it?
        if alpha + beta >= 1:
            return alpha*self.X + beta*self.Y - self.I
        a = alpha/self.step
        i = int(a)
        a_frac = a - i
        b = beta/self.step
        j = int(b)
        b_frac = b - j
        if a_frac + b_frac <= 1:
            return ( a_frac*self.shape_tbl[i+1,j] +
                     b_frac*self.shape_tbl[i,j+1] +
                     (1-a_frac-b_frac)*self.shape_tbl[i,j] )
        else:
            return ( (1-b_frac)*self.shape_tbl[i+1,j] +
                     (1-a_frac)*self.shape_tbl[i,j+1] +
                     (a_frac+b_frac-1)*self.shape_tbl[i+1,j+1] )
    
    
    def __repr__(self):
        results = [self.description]
        results.append(f"{'H(X):':<14} {self.X: <9.6f}")
        results.append(f"{'H(Y):':<14} {self.Y: <9.6f}")
        results.append(f"{'H(X|Y):':<14} {self.XlY: <9.6f}")
        results.append(f"{'H(Y|X):':<14} {self.YlX: <9.6f}")
        results.append(f"{'I(X:Y):':<14} {self.I: <9.6f}")
        results.append(f"{'H(XY):':<14} {self.XY: <9.6f}")
        results.append(f"{'L(X,Y):':<14} {(self.XlY+self.YlX)/2: <9.6f}")
        results.append(f"{'E(X,Y):':<14} {self.E: <9.6f}")
        results.append(f"{'Ebar(X,Y):':<14} {self.Ebar: <9.6f}")
        results.append(f"{'log(λ1):':<14} {np.log(self.spec[0]): <9.6f}")
        results.append(f"{'log(λ2):':<14} {np.log(self.spec[1]): <9.6f}")
        results.append(f"{'(α_0,β_0):':<14} ({self.alpha0:>5.3f},{self.beta0:>5.3f})")
        results.append(f"{'Smin(α_0,β_0):':<14} {self.Smin0:<9.6f}")
        results.append(f"{'S(α_0,β_0):':<14} {self.S0:<9.6f}")
        results.append(f"{'Smax(α_0,β_0):':<14} {self.Smax0:<9.6f}")
        results.append(f"{'Rigidity:':<14} {self.R:<9.6f}")
        results.append(f"{'Expander coefficient:':<14}")
        results.append(f"{'':<14}{self.Expander: <9.6f}")
        results.append(f"{'Symmetric expander coefficient:':<14}")
        results.append(f"{'':<14}{self.Expander_sym: <9.6f}")
        results.append(f"{'shape_tbl: step:':<18} {self.step:<4.2f}")
        results.append(f"{'shape_tbl: points:':<18} {self.steps+1} x {self.steps+1}")
        nans = np.count_nonzero(np.isnan(self.shape_tbl))
        results.append(f"{'shape_tbl: nans: ':<18} {nans}")
        return "\n".join(results)

        
