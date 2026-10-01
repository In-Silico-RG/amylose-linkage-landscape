#!/bin/bash
#SBATCH -J amy_gly
#SBATCH -p amd
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=120
#SBATCH --mem=8G
#SBATCH -o slurm-%j.out
# Resume of the GLYCAM06j-1/TIP3P SC6 replicas r1-r4 after job 30315 ended OUT_OF_MEMORY (MaxRSS 63 GB under
# --mem=60G: the HIP GPU path of gromacs/amd 2026.1 grows without bound; the CPU-only CHARMM jobs 30313/30314
# stayed at 1.4-2.1 GB). Same md.tpr, continued from md.cpt on CPU (-nb cpu -pme cpu), 30 threads each.
# r5 finished its 100 ns inside 30315. A killed mdrun is restarted from its last checkpoint (up to 3 tries).
module load gromacs/amd
GMX=$(command -v gmx_mpi)
i=0
for k in 1 2 3 4; do
  ( cd r$k
    for try in 1 2 3; do
      [ -f md.gro ] && break
      OMP_NUM_THREADS=30 mpirun -np 1 --bind-to none $GMX mdrun -s md.tpr -deffnm md -ntomp 30 -nb cpu -pme cpu \
          -pin on -pinoffset $((i*30)) -pinstride 1 -cpi md.cpt -cpt 30 > md_resume$try.out 2>&1
    done ) &
  i=$((i+1))
done
wait
