#!/bin/bash

#SBATCH -n 1
#SBATCH -N 1
#SBATCH --mem=500G
#SBATCH -c 10
#SBATCH --job-name="scWorkflow"
#SBATCH --output=output_bonn_berlin.txt
#SBATCH -p general

# source activate severityPred_env
srun snakemake --cores --unlock

# srun snakemake --cores all all --rerun-incomplete -s snakefile


srun snakemake all --snakefile $(pwd)/snakefile --configfile $(pwd)/config.yml --directory $(pwd) --use-singularity --singularity-prefix $(pwd)/.snakemake/singularity --printshellcmds --singularity-args "-B /prj" --cores all 

# srun snakemake "../../output_berlin_stan/merged_training/pred_j.csv" --snakefile snakefile --configfile config.yaml --cores all