#!/bin/bash
# usage: xtb_point.sh phi psi -> "E(Eh) phi_final psi_final" after a constrained GFN2/ALPB(water) optimisation
E=/media/aldo/Aldo/envs/chem/bin
read p1 p2 p3 p4 < <(sed -n 1p torsions_a.txt); read s1 s2 s3 s4 < <(sed -n 2p torsions_a.txt)
wd=$(mktemp -d scan_XXXX); cd $wd
cat > c.inp <<X
\$constrain
  force constant=1.0
  dihedral: $p1,$p2,$p3,$p4,$1
  dihedral: $s1,$s2,$s3,$s4,$2
\$end
X
OMP_NUM_THREADS=${NT:-1} $E/xtb ../maltose_a.xyz --gfn 2 --alpb water --opt normal --input c.inp > out.log 2>&1
en=$(grep "TOTAL ENERGY" out.log | tail -1 | awk '{print $4}')
tors=$($E/python - <<PY
import numpy as np
L=open("xtbopt.xyz").read().split("\n"); n=int(L[0]); X=np.array([[float(v) for v in l.split()[1:4]] for l in L[2:2+n]])
def dih(p0,p1,p2,p3):
    b0,b1,b2=p0-p1,p2-p1,p3-p2; b1n=b1/np.linalg.norm(b1); v=b0-np.dot(b0,b1n)*b1n; w=b2-np.dot(b2,b1n)*b1n
    return np.degrees(np.arctan2(np.dot(np.cross(b1n,v),w),np.dot(v,w)))
i=[int(v)-1 for v in "$p1 $p2 $p3 $p4 $s1 $s2 $s3 $s4".split()]
print(f"{dih(*X[i[:4]]):.1f} {dih(*X[i[4:]]):.1f}")
PY
)
echo "$en $tors"; cd ..; rm -rf $wd
