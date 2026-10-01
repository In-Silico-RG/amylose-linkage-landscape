#!/bin/bash
#SBATCH -J amy_md
#SBATCH -p gpu_titan
#SBATCH -w felix
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH -o slurm-%j.out
# usage: sbatch job_long.sh <SYS> <ns>   e.g. 6SC 200
# Extends the original 10 ns production tpr (CHARMM36/TIP3P, GROMACS 2023, Yurimar Ruiz) to <ns> ns
# from the same start (no topology on disk to regenerate velocities). Runs in felix /tmp;
# the trr stream (x,v,f every 10 ps) goes to /dev/null via symlink null.trr; xtc, edr, log, cpt copied back every hour.
module load gromacs/titan
SYS=$1; NS=$2; SRC=$PWD; W=/tmp/amy_md_${SYS}_$SLURM_JOB_ID; mkdir -p $W; cd $W
GMX=$(command -v gmx_mpi || command -v gmx); export GMX_MAXBACKUP=-1; ln -sf /dev/null null.trr
$GMX convert-tpr -s $SRC/${SYS}-prod.tpr -nsteps $((NS*500000)) -o long.tpr
( while sleep 3600; do cp long.xtc long.edr long.log long.cpt $SRC/${SYS}_long/ 2>/dev/null; done ) &
mkdir -p $SRC/${SYS}_long
$GMX mdrun -s long.tpr -deffnm long -o null.trr -ntomp $SLURM_CPUS_PER_TASK -nb gpu -pme gpu -bonded cpu -cpi long.cpt -cpt 30
cp long.xtc long.edr long.log long.cpt long.gro $SRC/${SYS}_long/
kill %1; rm -rf $W
