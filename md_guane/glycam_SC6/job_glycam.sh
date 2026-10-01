#!/bin/bash
#SBATCH -J amy_gly
#SBATCH -p amd
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=120
#SBATCH --mem=60G
#SBATCH -o slurm-%j.out
# GLYCAM06j-1/TIP3P SC6 (tleap from sequence ROH 4GA5 0GA, every linkage set to phi 92, psi -142 as the
# CHARMM36 start; ParmEd conversion validated against OpenMM). EM once, then 5 replicas x
# (NVT 100 ps, gen-seed 2001..2005 -> NPT 1 ns -> 100 ns production), 24 cores each. xtc = sugar only.
module load gromacs/amd
GMX=$(command -v gmx_mpi)
run() { OMP_NUM_THREADS=24 mpirun -np 1 --bind-to none $GMX "$@"; }
[ -f em.gro ] || { $GMX grompp -f em.mdp -c sc6.gro -p sc6.top -n idx.ndx -o em.tpr -maxwarn 2 > em_gpp.log 2>&1 && run mdrun -deffnm em -ntomp 24 > em.out 2>&1; }
i=0
for k in 1 2 3 4 5; do
  ( d=r$k; mkdir -p $d; cd $d
    sed "s/SEED/$((2000 + k))/" ../nvt.mdp > nvt.mdp
    off=$((i*24))
    [ -f nvt.gro ] || { $GMX grompp -f nvt.mdp -c ../em.gro -p ../sc6.top -n ../idx.ndx -o nvt.tpr -maxwarn 2 > nvt_gpp.log 2>&1 && run mdrun -deffnm nvt -ntomp 24 -pin on -pinoffset $off > nvt.out 2>&1; }
    [ -f npt.gro ] || { $GMX grompp -f ../npt.mdp -c nvt.gro -t nvt.cpt -p ../sc6.top -n ../idx.ndx -o npt.tpr -maxwarn 2 > npt_gpp.log 2>&1 && run mdrun -deffnm npt -ntomp 24 -pin on -pinoffset $off > npt.out 2>&1; }
    [ -f md.tpr ] || $GMX grompp -f ../prod.mdp -c npt.gro -t npt.cpt -p ../sc6.top -n ../idx.ndx -o md.tpr -maxwarn 2 > md_gpp.log 2>&1
    run mdrun -deffnm md -ntomp 24 -pin on -pinoffset $off -cpi md.cpt -cpt 30 > md.out 2>&1 ) &
  i=$((i+1))
done
wait
