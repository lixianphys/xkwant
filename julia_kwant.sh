#!/bin/bash

#SBATCH -J xkwant
#SBATCH -c 16
#SBATCH -t 2-00:00:00
#SBATCH --output=kwant_output
#SBATCH --mail-type=ALL
#SBATCH --mail-user=lixian.philips.wang@gmail.com

source ~/anaconda3/etc/profile.d/conda.sh
echo "Initialized Conda"

conda activate MyKwant
echo "Activated MyKwant conda env"

python setup.py install
echo "Installed xkwant by setup.py"

pip3 install tqdm
echo "pip installed tqdm"

python -W ignore scripts/hpc.py