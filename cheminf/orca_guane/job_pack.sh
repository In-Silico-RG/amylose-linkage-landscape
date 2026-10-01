#!/bin/bash
#SBATCH -J amy_pack
#SBATCH -p amd
#SBATCH -N 1
#SBATCH --ntasks=96
#SBATCH --mem=300G
#SBATCH -o slurm-%j.out
# usage: sbatch job_pack.sh <cell dir> ...  (QOS: 4 jobs/user, so cells are packed, 12 cores each)
# ORCA runs in node-local /tmp (HOME quota 20 GB); only .inp/.out/.xyz/.property.txt come back.
module load orca/6.1.1
ORCA=$(which orca)
W=/tmp/amy_$SLURM_JOB_ID; mkdir -p $W
run_cell() {
  local src=$1 d=$W/$(basename $1)
  mkdir -p $d && cp $src/*.inp $src/frame.xyz $d/ && cp $src/*.out $src/*.xyz $d/ 2>/dev/null
  cd $d || return
  for s in 1_sp_tzvp 2_sp_svp 3_opt_3c 4_sp_opt; do
    grep -q "ORCA TERMINATED NORMALLY" $s.out 2>/dev/null && continue
    $ORCA $s.inp > $s.out 2>&1
    cp $s.out $src/; cp $s.xyz $s.property.txt $src/ 2>/dev/null
    grep -q "ORCA TERMINATED NORMALLY" $s.out || { echo "$src $s failed"; return 1; }
  done
  cp *.xyz $src/ 2>/dev/null; echo done > $src/DONE
}
for c in "$@"; do run_cell "$c" & done
wait
rm -rf $W
