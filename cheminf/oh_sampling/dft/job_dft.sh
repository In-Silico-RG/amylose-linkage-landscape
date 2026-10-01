#!/bin/bash
#SBATCH -J amy_ohd
#SBATCH -p legacy
#SBATCH -N 1
#SBATCH --ntasks=24
#SBATCH --mem=90G
#SBATCH -o slurm-%j.out
# usage: sbatch job_dft.sh <dir> ...; r2SCAN-3c constrained opt then TZVP-D3 SP, 4 structures x 6 cores, node /tmp
module load orca/6.1.1
ORCA=$(which orca); W=/tmp/amy_ohd_$SLURM_JOB_ID; mkdir -p $W; export ORCA W
one() {
  src=$1; d=$W/$(basename $1); mkdir -p $d; cp $src/*.inp $src/frame.xyz $d/; cd $d
  for s in 3_opt_3c 4_sp_opt; do
    $ORCA $s.inp > $s.out 2>&1; cp $s.out $src/; cp $s.xyz $src/ 2>/dev/null
    grep -q "ORCA TERMINATED NORMALLY" $s.out || { echo "$src $s failed"; return 1; }
  done; echo done > $src/DONE
}
export -f one
printf '%s\n' "$@" | xargs -P 4 -I{} bash -c 'one {}'
rm -rf $W
