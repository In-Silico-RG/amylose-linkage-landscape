#!/bin/bash
#SBATCH -J amy_ohx
#SBATCH -p legacy
#SBATCH -N 1
#SBATCH --ntasks=24
#SBATCH --mem=60G
#SBATCH -o slurm-%j.out
# xtb 6.7.1 (static, ~/soft/xtb-dist) GFN2/ALPB water constrained optimisations of every start
# (phi, psi fixed as in the map), 24 single-thread jobs at a time (1 thread is fastest, see memory).
# Runs in node /tmp; copies back <start>.opt.xyz (comment line holds the energy) and a log tail.
X=~/soft/xtb-dist/bin/xtb; SRC=$PWD; W=/tmp/amy_ohx_$SLURM_JOB_ID; mkdir -p $W
export XTBPATH=~/soft/xtb-dist/share/xtb OMP_NUM_THREADS=1 OMP_STACKSIZE=1G
opt1() {
  c=$(dirname $1); s=$(basename $1 .xyz); d=$W/$c/$s; mkdir -p $d; cp $SRC/$1 $d/in.xyz; cp $SRC/$c/c.inp $d/
  (cd $d && timeout 1200 $X in.xyz --gfn 2 --alpb water --opt normal --input c.inp > out.log 2>&1)
  [ -f $d/xtbopt.xyz ] && cp $d/xtbopt.xyz $SRC/$c/$s.opt.xyz || echo "$1 no result" >> $SRC/failed.txt
  rm -rf $d
}
export -f opt1; export X SRC W
ls c_*/s_*.xyz | grep -v opt | xargs -P 24 -I{} bash -c 'opt1 {}'
rm -rf $W; echo done > $SRC/XTB_DONE
