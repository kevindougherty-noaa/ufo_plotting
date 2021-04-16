#!/bin/bash
#SBATCH -J test
#SBATCH -A da-cpu
#SBATCH -q batch 
#SBATCH --partition=bigmem
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=40
#SBATCH -t 0:45:00
#SBATCH --mail-user=$LOGNAME@noaa.gov

module use -a /scratch2/NCEPDEV/marineda/Jong.Kim/save/modulefiles/
module load anaconda/3.15.1

filepath=/scratch2/NCEPDEV/stmp1/Emily.Liu/data_aop20_global/scripts/OutData/AIRS/airs_aqua/
OUTDIR=./

for file in $(ls ${filepath}*output.nc4) ; do
  python plotting_iasi_cris_airs.py -d $file -o $OUTDIR > out.$(basename -- $file)  2>&1 &
  echo "python plotting_iasi_cris_airs.py $file "
done
wait
