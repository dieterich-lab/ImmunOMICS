#!/bin/bash

#SBATCH -n 1
#SBATCH -N 1
#SBATCH --mem=240G
#SBATCH -c 40
#SBATCH --job-name="prediction"
#SBATCH --output=out.txt
#SBATCH -p general

source activate immun2sev
srun snakemake --cores --unlock

# get prediction output with all genes in the interesction of DESeq2 and edgeR
srun snakemake --cores all predict --snakefile src/snakefile   --configfile config-All.yml --rerun-incomplete
srun snakemake --cores all predict_val --snakefile src/snakefile   --configfile config-All.yml --rerun-incomplete
# get prediction output with the interesction of TOP15 genes from DESeq2 and edgeR
srun snakemake --cores all predict --snakefile src/snakefile   --configfile config-Top15.yml --rerun-incomplete
srun snakemake --cores all predict_val --snakefile src/snakefile   --configfile config-Top15.yml --rerun-incomplete
# get prediction output with the interesction of TOP10 genes from DESeq2 and edgeR
srun snakemake --cores all predict --snakefile src/snakefile   --configfile config-Top10.yml --rerun-incomplete
srun snakemake --cores all predict_val --snakefile src/snakefile   --configfile config-Top10.yml --rerun-incomplete
# get prediction output with the interesction of TOP5 genes from DESeq2 and edgeR
srun snakemake --cores all predict --snakefile src/snakefile   --configfile config-Top5.yml --rerun-incomplete
srun snakemake --cores all predict_val --snakefile src/snakefile   --configfile config-Top5.yml --rerun-incomplete
# get prediction output with the interesction of TOP5 genes from DESeq2 and edgeR and GenderAge integration
srun snakemake --cores all predict --snakefile src/snakefile-GA   --configfile config-GA.yml --rerun-incomplete
srun snakemake --cores all predict_val --snakefile src/snakefile-GA   --configfile config-GA.yml --rerun-incomplete
# get celltype-specific prediction output
srun snakemake --cores all all --snakefile src_celltype/sub_snakefile   --configfile config_celltype.yml --rerun-incomplete 
srun snakemake --cores all all --snakefile src_celltype/snakefile   --configfile config_celltype.yml --rerun-incomplete 
# get Figures as presented in the paper
srun snakemake --cores all all --snakefile src/snakefile-downstream   --configfile config-Top5.yml --rerun-incomplete


#run rule predict for prediction on the test set and rule predict_val for the prediction on the validation set