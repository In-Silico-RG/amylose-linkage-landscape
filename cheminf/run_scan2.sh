#!/bin/bash
# Sequential relaxed (phi,psi) scan, xtb native $scan: 36 psi rows (psi -120 -> 230, 36 pts) at fixed phi.
# Each row is seeded by the lower-energy VALID frame of the forward (scan2/col) or backward (scan2/colb)
# phi column scan at psi = -120.
cd "$(dirname "$0")"
E=/media/aldo/Aldo/envs/chem/bin
read p1 p2 p3 p4 < <(sed -n 1p torsions_a.txt); read s1 s2 s3 s4 < <(sed -n 2p torsions_a.txt)
$E/python validate_frames.py scan2/col/xtbscan.log --column 2>/dev/null | grep -v -i warn > scan2/col.val
$E/python validate_frames.py scan2/colb/xtbscan.log --column-back 2>/dev/null | grep -v -i warn > scan2/colb.val
for k in $(seq 0 35); do
  phi=$(( (100 + 10*k + 180) % 360 - 180 )); d=scan2/row_$phi; mkdir -p $d
  kb=$(( (36 - k) % 36 ))                                  # backward frame index for the same phi
  ef=$(awk -v p=$phi '$1==p && $6==1 {print $3}' scan2/col.val); eb=$(awk -v p=$phi '$1==p && $6==1 {print $3}' scan2/colb.val)
  src=col; idx=$k
  if [ -n "$eb" ] && { [ -z "$ef" ] || awk -v a=$eb -v b=$ef 'BEGIN{exit !(a<b)}'; }; then src=colb; idx=$kb; fi
  if [ -z "$ef" ] && [ -z "$eb" ]; then src=col; idx=$k; fi                   # no valid seed: use forward anyway
  echo "$phi seed=$src frame=$idx Ef=$ef Eb=$eb" >> scan2/seeds.log
  $E/python validate_frames.py scan2/$src/xtbscan.log --extract $idx $d/start.xyz
  printf '$constrain\n  force constant=0.5\n  dihedral: %s,%s,%s,%s,%s\n  dihedral: %s,%s,%s,%s,-120\n$end\n$scan\n  2: -120,230,36\n$end\n' $p1 $p2 $p3 $p4 $phi $s1 $s2 $s3 $s4 > $d/c.inp
  echo $phi
done | xargs -P 8 -I{} bash -c 'cd scan2/row_{} && OMP_NUM_THREADS=1 '$E'/xtb start.xyz --gfn 2 --alpb water --opt normal --input c.inp > out.log 2>&1'
: > scan2_10deg.dat
for k in $(seq 0 35); do phi=$(( (100 + 10*k + 180) % 360 - 180 )); $E/python validate_frames.py scan2/row_$phi/xtbscan.log $phi 2>/dev/null | grep -v -i warn >> scan2_10deg.dat; done
echo DONE >> scan2_10deg.dat
