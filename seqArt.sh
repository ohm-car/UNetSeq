#!/bin/bash

#SBATCH --job-name=seqArt
#SBATCH --mail-user=omkark1@umbc.edu
#SBATCH --mem=144G
#SBATCH --gres=gpu:4
#SBATCH --time=10-12:00:00
#SBATCH --constraint=rtx_6000
#SBATCH --output=/nfs/ada/oates/users/omkark1/ArteryProj/UNetSeq/outfiles/seqArt_run_10.out
#SBATCH --error=/nfs/ada/oates/users/omkark1/ArteryProj/UNetSeq/outfiles/seqArt_run_10.err


export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/

source activate mrcnn9

python /nfs/ada/oates/users/omkark1/ArteryProj/UNetSeq/trainFrames.py
