#!/bin/bash

#SBATCH -n 1
#SBATCH -N 1
#SBATCH --mem=240G
#SBATCH -c 10
#SBATCH --job-name="prediction"
#SBATCH --output=out.txt
#SBATCH -p general

######Conda option
# source activate immun2sev
# srun snakemake --cores --unlock

# # # get prediction output with all genes in the interesction of DESeq2 and edgeR
# srun snakemake --cores all all --snakefile src/snakefile   --configfile config-Top5.yml --rerun-incomplete
# srun snakemake --cores all predict_val --snakefile src/snakefile   --configfile config-Top5.yml --rerun-incomplete
# # get prediction output with the interesction of TOP5 genes from DESeq2 and edgeR and GenderAge integration
# srun snakemake --cores all all --snakefile src/snakefile-GA   --configfile config-GA.yml --rerun-incomplete
# srun snakemake --cores all predict_val --snakefile src/snakefile-GA   --configfile config-GA.yml --rerun-incomplete
# #get celltype-specific prediction output
# srun snakemake --cores all all --snakefile src_celltype/sub_snakefile   --configfile config_celltype.yml --rerun-incomplete 
# srun snakemake --cores all all --snakefile src_celltype/snakefile   --configfile config_celltype.yml --rerun-incomplete 
# # get Figures as presented in the paper, please respect names of output folders in config files to get the result
# srun snakemake --cores all all --snakefile src/snakefile   --configfile config-All.yml --rerun-incomplete
# srun snakemake --cores all all --snakefile src/snakefile   --configfile config-Top10.yml --rerun-incomplete
# srun snakemake --cores all all --snakefile src/snakefile   --configfile config-Top15.yml --rerun-incomplete
# srun snakemake --cores all all --snakefile src/snakefile-downstream   --configfile config-Top5.yml --rerun-incomplete


##### With singularity

var1=/prj/NUM_CODEX_PLUS/Amina/CellSubmission
var2=/home/alemsara
var3=ImmunOMICS/aminale_immun2sev_latest-2023-05-11-be16ed9eed81.sif
singularity run -B $var1 -B $var2 \
$var1/$var3 \
                  "snakemake --cores all all --snakefile src/snakefile  \
                  --configfile $var1/ImmunOMICS/config-Top5.yml --directory $var2"

singularity run -B $var1 -B $var2  \
$var1/$var3 \
                  "snakemake --cores all all --snakefile src/snakefile-GA  \
                  --configfile $var1/ImmunOMICS/config-GA.yml --directory $var2"

singularity run -B $var1 -B $var2 \
$var1/$var3\
                  "snakemake --cores all all --snakefile src_celltype/sub_snakefile  \
                  --configfile $var1/ImmunOMICS/config_celltype.yml --directory $var2"

singularity run -B $var1 -B $var2 \
$var1/$var3\
                  "snakemake --cores all all --snakefile src_celltype/snakefile  \
                  --configfile $var1/ImmunOMICS/config_celltype.yml --directory $var2"


# # # get Figures as presented in the paper, please respect names of output folders in config files to get the result
singularity run -B $var1 -B $var2  \
$var1/$var3 \
                  "snakemake --cores all all --snakefile src/snakefile  \
                  --configfile $var1/ImmunOMICS/config-All.yml --directory $var2"

singularity run -B $var1 -B $var2  \
$var1/$var3 \
                  "snakemake --cores all all --snakefile src/snakefile  \
                  --configfile $var1/ImmunOMICS/config-Top10.yml --directory $var2"

singularity run -B $var1 -B $var2  \
$var1/$var3 \
                  "snakemake --cores all all --snakefile src/snakefile  \
                  --configfile $var1/ImmunOMICS/config-Top15.yml --directory $var2"

singularity run -B $var1 -B $var2  \
$var1/$var3 \
                  "snakemake --cores all all --snakefile src/snakefile-downstream  \
                  --configfile $var1/ImmunOMICS/config-Top5.yml --directory $var2"
