#! /bin/bash

#SBATCH -p rome
#SBATCH -t 01:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=arne.eichholtz@student.uva.nl

srun find -O3 /gpfs/scratch1/nodespecific/ -maxdepth 2 -type d -user aeichholtz -exec rm -rf {} \;