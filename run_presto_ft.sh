#!/bin/bash 

#SBATCH --partition=batch                   # Name of Partition
#SBATCH --job-name=PrestoFT                 # Name of job
#SBATCH --ntasks=1                          # Number of CPU processes
#SBATCH --cpus-per-task=8                   # Number of CPU threads
#SBATCH --time=72:00:00                     # Wall time (format: d-hh:mm:ss)
#SBATCH --mem=28gb                          # Amount of memory (units: gb, mg, kb)
#SBATCH --gpus=1                            # Number of GPU
#SBATCH --mail-type=BEGIN,END,FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=giorgia.milli@vito.be         
#SBATCH --nodelist=sasdsnode01

# Load the most recent version of CUDA
module load CUDA

# Activate pre-installed python environment
source activate sadenv

output_dir="/home/vito/millig/gio/data/presto_ft/"
data_dir="/home/vito/millig/gio/data/presto_ft/"
train_file="rawts-10d_catboost_train.parquet"
val_file="rawts-10d_catboost_val.parquet"
dekadal=true
n_epochs=0

# Run your python script here (don't forget to use srun)
srun python train.py \
    --output_dir "$output_dir" \
    --data_dir "$data_dir" \
    --train_file "$train_file" \
    --val_file "$val_file" \
    --dekadal "$dekadal" \
    --n_epochs "$n_epochs" \
    --warm_start \