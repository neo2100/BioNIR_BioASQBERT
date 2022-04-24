#!/bin/bash
#SBATCH -J Bionir_BioASQBERT_training
#SBATCH -n 1
#SBATCH -o training-%j.out
#SBATCH -e training-%j.err
#SBATCH -t 90
#SBATCH --mem=4G
#SBATCH --mail-type=END 
#SBATCH --gres=gpu:V100:1 # select a host with a Volta GPU

echo "Executing on $HOSTNAME"
RUNPATH=/home/shadmehr/Desktop/BioASQBERT/
cd $RUNPATH
source dataenv/bin/activate
python3  BioNIR_BioASQBERT/runTripleFinetune.py
#rhrk-singularity pytorch_22.03-py3.simg ./singularity_script.script
