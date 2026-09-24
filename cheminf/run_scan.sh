#!/bin/bash
# relaxed (phi,psi) scan of methyl alpha-maltoside (4C1,4C1; maltose_a.xyz), GFN2-xTB/ALPB(water), 10 deg grid, 8 single-thread jobs
cd "$(dirname "$0")"
: > scan_10deg.dat
for phi in $(seq -180 10 170); do for psi in $(seq -180 10 170); do echo "$phi $psi"; done; done \
 | xargs -P 8 -n 2 bash -c 'e=$(./xtb_point.sh $0 $1); echo "$0 $1 $e"' >> scan_10deg.dat
echo DONE >> scan_10deg.dat
