#!/usr/bin/python3

import numpy as np
from finite_fields import PrimeField
from itertools import combinations, product

################################################################################
### Projective Flags

### Points in FP^2 will be indexed by
### 1. [0:0:1] -- number 0               ( n = 0 )
### 2. [0:1:k] -- numbers 1 to q         ( n = 1 + k )
### 3. [1:j:k] -- numbers q+1 to q^2+q+1 ( n = 1 + q + q*j + k )

### in the dual space use the same ordering. For simplicity do only for
### primes, not to realize arithmetic in non-prime fields.

def relax(*args,**kwargs):
    pass

# maybe_print = print
maybe_print = relax

def _get_point_indexFP2(i,j,k,F):
    """Return the index of the point with coordinates [i:j:k], in
    the projective plane over field F.

    Points in FP^2 are indexed by
    1. [0:0:1] -- number 0               ( n = 0 )
    2. [0:1:k] -- numbers 1 to q         ( n = k+1  )
    3. [1:j:k] -- numbers q+1 to q^2+q+1 ( n = q*(j+1) + k+1 )
    """
    
    i = F(i)
    j = F(j)
    k = F(k)
    
    if (i,j) == (0,0): return 0
    
    if i == 0:
        k = k * (j**(-1))     
        return int(k)+1
    
    i_inv = i**(-1) 
    j = j * i_inv 
    k = k * i_inv
    
    return F.order * (int(j)+1) + (int(k) + 1) 

def _incidence_proj_flags_v1(F):

    """Construct the incidence matrix for the graph of point-in-line
    incidence in the projective plane over the field of prime order q
    """

    size = F.order*F.order + F.order + 1
    M = np.zeros( (size,size) , dtype=np.uint8)
    
    ### FIRST BATCH: [i:j:k]=[0:0:1]
    (i,j,k)=(0,0,1)
    maybe_print("Point:",(i,j,k))
    n = _get_point_indexFP2(i,j,k,F)
    maybe_print("\tn:",n)
    line0 = [0,1,0]
    line1 = [1,0,0]
    maybe_print("\tL:",line1)
    m = _get_point_indexFP2(*line1,F)
    maybe_print("\tm:",m)
    M[n,m] = 1
    for t in F.range():
        line = [ c0 + c1*t for (c0,c1) in zip(line0,line1) ]
        maybe_print("\tL:",line)
        m = _get_point_indexFP2(*line,F)
        maybe_print("\tm:",m)
        M[n,m] = 1

    ### SECOND BATCH: [i:j:k]=[0:1:*]
    (i,j)=(0,1)
    for k in F.range():
        maybe_print("P:",(i,j,k))
        n = _get_point_indexFP2(i,j,k,F)
        maybe_print("\tn:",n)
        line0 = [0, F.order - k, 1]
        line1 = [1,0,0]
        maybe_print("\tL:",line1)
        m = _get_point_indexFP2(*line1,F)
        maybe_print("\tm:",m)
        M[n,m] = 1
        for t in F.range():
            line = [ c0 + c1*t for (c0,c1) in zip(line0,line1) ]
            maybe_print("\tL:",line)
            m = _get_point_indexFP2(*line,F)
            maybe_print("\tm:",m)
            M[n,m] = 1

    ### THIRD BATCH: [i:j:k] = [1:*:*]
    i = 1
    ### case 1 [1:0:0], 1 count
    (j,k) = (0,0)
    maybe_print("P:",(i,j,k))
    n = _get_point_indexFP2(i,j,k,F)
    maybe_print("\tn:",n)
    line0 = (0,0,1)
    line1 = (0,1,0)
    maybe_print("\tL:",line1)
    m = _get_point_indexFP2(*line1,F)
    maybe_print("\tm:",m)
    M[n,m] = 1
    for t in F.range():
        line = [ c0 + c1*t for (c0,c1) in zip(line0,line1) ]
        maybe_print("\tL:",line)
        m = _get_point_indexFP2(*line,F)
        maybe_print("\tm:",m)
        M[n,m] = 1
        
    ### case 2 [1:0:k], F.order-1 count
    j = 0
    for k in F.range(nonzero=True):
        maybe_print("P:",(i,j,k))
        n = _get_point_indexFP2(i,j,k,F)
        maybe_print("\tn:",n)
        line0 = [0,1,0]
        line1 = [F.order-k,0,1]
        maybe_print("\tL:",line1)
        m = _get_point_indexFP2(*line1,F)
        maybe_print("\tm:",m)
        M[n,m] = 1
        for t in F.range():
            line = [ c0 + c1*t for (c0,c1) in zip(line0,line1) ]
            maybe_print("\tL:",line)
            m = _get_point_indexFP2(*line,F)
            maybe_print("\tm:",m)
            M[n,m] = 1
            
    ### case 3: [1:j:k], (q-1)*q count
    for j in F.range(nonzero=True):
        for k in F.range():
            maybe_print("P:",(i,j,k))
            n = _get_point_indexFP2(i,j,k,F)
            maybe_print("\tn:",n)
            line0 = [0,F.order-k,j]
            line1 = [j,F.order-k-1,j]
            maybe_print("\tL:",line1)
            m = _get_point_indexFP2(*line1,F)
            maybe_print("\tm:",m)
            M[n,m] = 1
            for t in F.range():
                line = [ c0 + c1*t for (c0,c1) in zip(line0,line1) ]
                m = _get_point_indexFP2(*line,F)
                M[n,m] = 1
    return M

def _list_pointsFP2(F):
    yield [0,0,1]
    for k in F.range():
        yield [0,1,k]
    for j in F.range():
        for k in F.range():
            yield [1,j,k]

def _incidence_proj_flags_v2(F):

    """Construct the incidence matrix for the graph of point-in-line
    incidence in the projective plane over the field of prime order q.
    F --- finite field
    returns incidence matrix as ndarray(dttype=np.uint8)
    """

    size = F.order*F.order + F.order +1 
    M = np.zeros( (size,size) , dtype=np.uint8)

    for m, pt in enumerate(_list_pointsFP2(F)):
        for n, li in enumerate(_list_pointsFP2(F)):
            coupling = F(sum([ cpt*cli for (cpt,cli) in zip(pt,li) ]))
            if coupling == 0:
                M[n,m] = 1
    return M

### v2 is twice faster!
incidence_proj_flags = _incidence_proj_flags_v2

### Projective Flags
################################################################################

################################################################################
### Cycle graph

def incidence_cycle(n):
    M = np.zeros((n,n))
    for i in range(n):
        j = (i+1)%n
        M[i,i] = 1
        M[i,j] =1
    return M

### Cycle graph
################################################################################

################################################################################
### Linear flags

def _subspaces(d, n, F):
    """
    Generate all d-dimensional subspaces of F^n.

    Each subspace is represented by its unique d x n reduced
    row-echelon basis matrix.
    (d x n)-matrix 
    """

    # Choose the pivot columns of the RREF matrix.
    for pivots in combinations(range(n), d):
        pivot_set = set(pivots)

        # In row i, arbitrary entries can occur only in nonpivot
        # columns strictly to the right of the pivot pivots[i].
        free_positions = [ (i, j)
                           for i, pivot in enumerate(pivots)
                           for j in range(pivot + 1, n)
                           if j not in pivot_set
                          ]

        for values in product(F.range(), repeat=len(free_positions)):
            basis = np.zeros((d, n))

            # Insert the pivot columns.
            for i, pivot in enumerate(pivots):
                basis[i, pivot] = 1

            # Insert the free entries.
            for (i, j), value in zip(free_positions, values):
                basis[i, j] = value

            yield basis


def _is_subspace(K,L,F):
    """ checks whether K is a subspace of L, where K,L is given by bases 
    in row echelon form"""

    Lheight, Lwidth = L.shape
    Kheight, Kwidth = K.shape

    if Kwidth != Lwidth:
        raise ValueError("K and L must have the same number of columns")

    ### TODO checks
    
    Lpivots = [ (i, np.flatnonzero(L[i])[0] ) for i in range(Lheight) ]
    ### assume all pivots are 1's
    ### TODO: checks
    
    for Krow in K:
        remainder = Krow.copy()
        for i,j in Lpivots:
            remainder -= remainder[j]*L[i]
        if np.any(remainder % F.order) != 0:
            return False
        
    return True
        
def gauss_choose(q, n: int, k: int) -> int:
    """Return the Gaussian binomial coefficient [n choose k]_q.

    For q a prime power, this is the number of k-dimensional subspaces
    of the vector space F_q^n.
    """
    ### TODO: add checks
    
    # The limit at q = 1 is the ordinary binomial coefficient.
    if q == 1:
        return comb(n, k)

    # Use symmetry [n choose k]_q = [n choose n-k]_q.
    k = min(k, n - k)
    result = 1
    for i in range(k):
        result *= q ** (n - i) - 1
        result //= q ** (i + 1) - 1

    return result

def incidence_lin_flags(k, l, n, F):
    """
    Return the incidence matrix of kD in lD flags in nD space over F.

    Rows are indexed by k-dimensional subspaces K of F_q^n.
    Columns are indexed by l-dimensional subspaces L of F_q^n.

    The entry M[i, j] equals 1 precisely when K_i is contained in L_j.

    Parameters
    ----------
    F: PrimeField(q)
        Prime field;
    k, l, n : int
        Dimensions satisfying 0 < k < l < n.

    Returns
    -------
    numpy.ndarray
        A {0,1}-matrix of shape

            ([n choose k]_q, [n choose l]_q).
    """

    ### TODO: Add tests
    dimGrk = gauss_choose(F.order, n, k)
    dimGrl = gauss_choose(F.order, n, l)
    
    M = np.zeros((dimGrl, dimGrk),
                 dtype=np.uint8,  
                )
    for i, K in enumerate(_subspaces(k,n,F)):
        for j, L in enumerate(_subspaces(l,n,F)):
            if _is_subspace(K, L, F):
                M[j,i] = 1

    return M

### Linear flags
################################################################################

################################################################################
### Hamming graph
### TODO: optimize
def incidence_hamming(n, k):
    """
    Return the incidence matrix of Ham(n, k).

    Both rows and columns are indexed by binary strings in {0,1}^n,
    identified with the integers 0, ..., 2**n - 1.

    The entry M[x, y] equals 1 exactly when the Hamming distance
    between x and y is k.
    """
    size = 1 << n
    M = np.zeros((size, size), dtype=np.uint8)

    for x in range(size):
        for y in range(size):
            # XOR has a 1 precisely in the coordinates where x and y differ.
            M[x, y] = bin(x ^ y).count("1") == k

    return M

### Hamming graph
################################################################################
