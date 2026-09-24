#!/bin/bash
# Sensitivity maps (2026-09-22): same sequential scan design as run_scan2.sh with another solvent model.
# usage: run_scan_solv.sh <tag> <xtb solvent flags>   e.g.  run_scan_solv.sh gbsa "--gbsa water"   |   run_scan_solv.sh vac ""
# Steps: free optimisation of maltose_a.xyz -> phi column scans forward/backward at psi=-120 -> 36 psi rows -> scan_<tag>_10deg.dat
cd "$(dirname "$0")"
tag=$1; SOLV=$2; D=scan_$tag; E=/media/aldo/Aldo/envs/chem/bin; export OMP_NUM_THREADS=1
mkdir -p $D/opt $D/col $D/colb
( cd $D/opt && $E/xtb ../../maltose_a.xyz --gfn 2 $SOLV --opt normal > out.log 2>&1 )
cp $D/opt/xtbopt.xyz $D/col/start.xyz; cp $D/opt/xtbopt.xyz $D/colb/start.xyz
cp scan2/col/c.inp $D/col/c.inp; cp scan2/colb/c.inp $D/colb/c.inp
( cd $D/col  && $E/xtb start.xyz --gfn 2 $SOLV --opt normal --input c.inp > out.log 2>&1 ) &
( cd $D/colb && $E/xtb start.xyz --gfn 2 $SOLV --opt normal --input c.inp > out.log 2>&1 ) &
wait
read p1 p2 p3 p4 < <(sed -n 1p torsions_a.txt); read s1 s2 s3 s4 < <(sed -n 2p torsions_a.txt)
$E/python validate_frames.py $D/col/xtbscan.log --column 2>/dev/null | grep -v -i warn > $D/col.val
$E/python validate_frames.py $D/colb/xtbscan.log --column-back 2>/dev/null | grep -v -i warn > $D/colb.val
: > $D/seeds.log
for k in $(seq 0 35); do
  phi=$(( (100 + 10*k + 180) % 360 - 180 )); d=$D/row_$phi; mkdir -p $d
  kb=$(( (36 - k) % 36 ))
  ef=$(awk -v p=$phi '$1==p && $6==1 {print $3}' $D/col.val); eb=$(awk -v p=$phi '$1==p && $6==1 {print $3}' $D/colb.val)
  src=col; idx=$k
  if [ -n "$eb" ] && { [ -z "$ef" ] || awk -v a=$eb -v b=$ef 'BEGIN{exit !(a<b)}'; }; then src=colb; idx=$kb; fi
  if [ -z "$ef" ] && [ -z "$eb" ]; then src=col; idx=$k; fi
  echo "$phi seed=$src frame=$idx Ef=$ef Eb=$eb" >> $D/seeds.log
  $E/python validate_frames.py $D/$src/xtbscan.log --extract $idx $d/start.xyz
  printf '$constrain\n  force constant=0.5\n  dihedral: %s,%s,%s,%s,%s\n  dihedral: %s,%s,%s,%s,-120\n$end\n$scan\n  2: -120,230,36\n$end\n' $p1 $p2 $p3 $p4 $phi $s1 $s2 $s3 $s4 > $d/c.inp
  echo $phi
done | xargs -P 8 -I{} bash -c 'cd '$D'/row_{} && OMP_NUM_THREADS=1 '$E'/xtb start.xyz --gfn 2 '"$SOLV"' --opt normal --input c.inp > out.log 2>&1'
: > scan_${tag}_10deg.dat
for k in $(seq 0 35); do phi=$(( (100 + 10*k + 180) % 360 - 180 )); $E/python validate_frames.py $D/row_$phi/xtbscan.log $phi 2>/dev/null | grep -v -i warn >> scan_${tag}_10deg.dat; done
echo DONE >> scan_${tag}_10deg.dat
