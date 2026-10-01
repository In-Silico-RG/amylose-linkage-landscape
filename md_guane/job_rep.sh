#!/bin/bash
#SBATCH -J amy_rep
#SBATCH -p amd
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=120
#SBATCH --mem=60G
#SBATCH -o slurm-%j.out
# usage (from rep_<SYS>/): sbatch ../job_rep.sh
# Five independent CHARMM36/TIP3P replicas, 100 ns each: starts = original trajectory at 2,4,6,8,10 ns,
# new Maxwell-Boltzmann velocities (gen-seed 1001..1005). Topology rebuilt from the 2025 production
# tpr by tpr2top.py and validated term by term (md_guane/<SYS>_val). xtc holds the sugar only.
module load gromacs/amd
GMX=$(command -v gmx_mpi)
i=0
for t in 2 4 6 8 10; do
  d=r$t; mkdir -p $d; cd $d
  sed "s/SEED/$((1000 + t/2))/" ../../rep.mdp > rep.mdp
  [ -f md.tpr ] || $GMX grompp -f rep.mdp -c ../start_${t}ns.gro -p ../topol.top -n ../idx.ndx -o md.tpr -maxwarn 2 > grompp.log 2>&1
  OMP_NUM_THREADS=24 mpirun -np 1 --bind-to none $GMX mdrun -s md.tpr -deffnm md -ntomp 24 -nb cpu -pme cpu \
      -pin on -pinoffset $((i*24)) -pinstride 1 -cpi md.cpt -cpt 30 > mdrun.out 2>&1 &
  cd ..; i=$((i+1))
done
wait
