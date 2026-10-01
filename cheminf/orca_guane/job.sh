#!/bin/bash
#SBATCH -J amy_dft
#SBATCH -p legacy
#SBATCH -N 1
#SBATCH --ntasks=12
#SBATCH --mem=45G
#SBATCH -o slurm-%j.out
# usage: sbatch job.sh <cell directory>; runs the four ORCA steps in order, skipping finished ones
module load orca/6.1.1
cd "$1"
ORCA=$(which orca)
for s in 1_sp_tzvp 2_sp_svp 3_opt_3c 4_sp_opt; do
  grep -q "ORCA TERMINATED NORMALLY" $s.out 2>/dev/null && continue
  $ORCA $s.inp > $s.out 2>&1
  grep -q "ORCA TERMINATED NORMALLY" $s.out || { echo "$s failed"; exit 1; }
done
echo done > DONE
