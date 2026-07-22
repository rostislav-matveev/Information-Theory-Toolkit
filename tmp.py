from shape import Shape
from incidence import *
import pickle
from persistent import persistent_dict

l=persistent_dict("Shape")

for pr in [2,3,5,7,11,13,17,19]:
    print(pr)
    F = PrimeField(pr)
    M = incidence_proj_flags(F)
    print("shape")
    l["pf",F]=Shape(M,100,description=f"Projective flags over {F}",assume_homogeneous=True, assume_symmetric = True, save_xy=True, save_stats=False)
    print(l["pf",F])

for n in range(2,9):
    for k in range(1,n//2):
        M = incidence_hamming(n,k)
        print("shape")
        l["ham",n,k]=Shape(M,100,description=f"Hamming graph on {n}-strings at distance {k}",assume_homogeneous=True, assume_symmetric = True, save_xy=True, save_stats=False)
        print(l["ham",n,k])

for q in [2]:
    F=PrimeField(q)
    for n in range(4,8):
        for l in range(2,n):
            for k in range(1,min(l,n-l)):
                print(k,l,n,F)
            
        print("shape")
        l["ham",n,k]=Shape(M,100,description=f"Hamming graph on {n}-strings at distance {k}",assume_homogeneous=True, assume_symmetric = True, save_xy=True, save_stats=False)
        print(l["ham",n,k])


M=incidence_lin_flags(1,2,4,PrimeField(2))
s=Shape(M,100,description=f"Linear flags ({1} in {2}) in  {PrimeField(2)}^{4}",assume_homogeneous=True, assume_symmetric = False, save_xy=True, save_stats=False)
    

for i in range(21):
    for j in range(i): 
        if i+j < 20:
            gap = abs(s7.shape_tbl[i,j]-s7.shape_tbl[j,i])
            print(f"[{i:>2},{j:>2}]: {gap:2e}")

M5 = incidence_proj_flags(PrimeField(5))
s5 = Shape(M5,20,description="Projective flags over GF(5)",assume_homogeneous=True, assume_symmetric = True, save_xy=True,save_stats=True)

M7 = incidence_proj_flags(PrimeField(7))
s7 = Shape(M7,20,description="Projective flags over GF(7)",assume_homogeneous=True, assume_symmetric = True, save_xy=True,save_stats=True)

M11 = incidence_proj_flags(PrimeField(11))
s11 = Shape(M11,20,description="Projective flags over GF(11)",assume_homogeneous=True, assume_symmetric = True, save_xy=True,save_stats=True)

M13 = incidence_proj_flags(PrimeField(13))
s13 = Shape(M13,20,description="Projective flags over GF(13)",assume_homogeneous=True, assume_symmetric = True, save_xy=True,save_stats=True)

M19 = incidence_proj_flags(PrimeField(19))
s19 = Shape(M19,20,description="Projective flags over GF(19)",assume_homogeneous=True, assume_symmetric = True, save_xy=True,save_stats=True)

M23 = incidence_proj_flags(PrimeField(23))
s23 = Shape(M23,20,description="Projective flags over GF(23)",assume_homogeneous=True, assume_symmetric = True, save_xy=True,save_stats=True)


M29 = incidence_proj_flags(PrimeField(29))
s29 = Shape(M29,20,description="Projective flags over GF(29)",assume_homogeneous=True, assume_symmetric = True, save_xy=True,save_stats=True)

M31 = incidence_proj_flags(PrimeField(31))
s31 = Shape(M31,20,description="Projective flags over GF(31)",assume_homogeneous=True, assume_symmetric = True, save_xy=True,save_stats=True)

M37 = incidence_proj_flags(PrimeField(37))
s37 = Shape(M37,20,description="Projective flags over GF(37)",assume_homogeneous=True, assume_symmetric = True, save_xy=True,save_stats=True)

M41 = incidence_proj_flags(PrimeField(41))
s41 = Shape(M41,20,description="Projective flags over GF(41)",assume_homogeneous=True, assume_symmetric = True, save_xy=True,save_stats=True)

M5 = incidence_proj_flags(PrimeField(5))
s5 = Shape(M5,20,description="Projective flags over GF(5)",assume_homogeneous=True, assume_symmetric = True, save_xy=True,save_stats=True)

M5 = incidence_proj_flags(PrimeField(5))
s5 = Shape(M5,20,description="Projective flags over GF(5)",assume_homogeneous=True, assume_symmetric = True, save_xy=True,save_stats=True)

M5 = incidence_proj_flags(PrimeField(5))
s5 = Shape(M5,20,description="Projective flags over GF(5)",assume_homogeneous=True, assume_symmetric = True, save_xy=True,save_stats=True)

M5 = incidence_lin_flags(1,2,4,PrimeField(3))
s5 = Shape(M5,20,description="lin  flags 1,3,5 over GF(5)",assume_homogeneous=True, assume_symmetric = True, save_xy=True,save_stats=True)


