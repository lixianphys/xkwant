#!/bin/bash
source ~/anaconda3/etc/profile.d/conda.sh
echo "Initialized Conda"

conda activate MyKwant
echo "Activated MyKwant conda env"

pip3 install -e .
echo "Installed xkwant by setup.py"

pip3 install tqdm
echo "pip installed tqdm"

python test_package.py
echo "Tested xkwant by test_package.py"
