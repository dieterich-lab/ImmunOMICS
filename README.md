COVID-19 severity prediction tool
================================

End-to-end Snakemake pipeline for predicting severity in COVID-19.


Overview
--------

The workflow shown below allows predicting COVID-19 severity from scRNA-seq data. One can run an end-to-end snakemake command that will basically perform cell annotation, features selection, training and prediction by setting specific parameters.

The workflow parameters should be set in the config file provided as a parameter for the snakemake command. 

The inputs are: 

- The training sets: list of datasets to be used for the training
- The testing sets: list of datasets to be tested by the trained model
- The output directory
- The number of Top genes differentially expressed between conditions (Mild or Severe) to be selected. (you can refer to our manuscript where we tested different numbers)

We provide in the following steps how to install/use the tool, set the config file and run the end-to-end snakemake command.

You can reproduce our results by using data in [zenodo](https://doi.org/10.5281/zenodo.7928637) and `job.sh` as we will describe below.
After reproducing our results you can use the trained model to predict severity on your data.

Quick Setup
-----------

```bash
# clone the repo
git clone https://github.com/AminaLEM/ImmunOMICS

# set code base path
SNK_REPO="`pwd`/ImmunOMICS"

```

Config file
-----------
Please provide input datasets in the Seurat format h5Seurat. 

* To convert h5ad file to h5Seurat, please use:

```bash
library(seuratDisk)
Convert("filename.h5ad", dest = "h5seurat", overwrite = TRUE)
```
* To save  an rds file as h5Seurat, please use (Check out [Conversions](https://mojaveazure.github.io/seurat-disk/articles/convert-anndata.html) vignette for further details):

```bash
library(seuratDisk)
SaveH5Seurat($varibale_name, filename = "filename.h5Seurat")
```

Please put all your input datasets in one directory. 

The input datasets should contain the following metadata columns: 
* "sampleID": sample IDs
* "condition": Mild or Severe
* "batch": the batch name of your data
* "who_score": if availbale else =condition (it serves as factor for the traning/validation split)

Here is the config file for the testing example. Data can be found in [zenodo](https://doi.org/10.5281/zenodo.7928637).


```
# INPUT & OUTPUT PATHS
path_out: "../../output"          # path to the output directory, if it doesn't exist it will be created 
path_inp: "../data"           # path to the directory containing input datasets in h5Seurat format


# INPUTS PARAMS
training_data:               # Training datasets
    set1:  'bonn.h5Seurat'        
    set2:  'berlin.h5Seurat'        
    
test_data:               # Testing datasets
    set3:  'korean.h5Seurat'        
    set4:  'stanford_pbmc.h5Seurat'   
    
nbTopGenes: 40  # number of top genes selected from pseudo bulk analysis to set GE modality. For a comparative analysis, we tested different numbers: 40, 15, 10, 5 each one in a separate config files.

```
Note that you can set as many training and testing datasets as you want. Datasets under `training_data` will be merged, and 80% will be used for the training, and 20 % for the validation split randomly 30 times. 

Reproducibility: Singularity
----------------------------

We recommend using the all-in-one image [aminale_immun2sev_latest-2023-05-11-be16ed9eed81.sif](https://doi.org/10.5281/zenodo.7928637).

To run the end-to-end pipeline on your data please use the following command. Make sure to set paths in the config file before you run the commands.

```bash 
singularity run -B /Host_directory aminale_immun2sev_latest-2023-05-11-be16ed9eed81.sif \
                  "snakemake --cores all all --snakefile src/snakefile  \
                  --configfile /path_to_config/config.yml --directory /writable_directory"
```

All commands needed to reproduce our results (main and comparative analysis that need in total 10 successive commands) are in `job.sh`; parameters are in the listed config files.

You can add your data under `test_data` in the config file to predict severity from our trained model.


Reproducibility: Conda   
----------------------

An alternative option to enhance reproducibility is to install software used via Conda.

You can find Miniconda installation instructions for Linux [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html).
Make sure you install the [Miniconda Python3 distribution](https://docs.conda.io/en/latest/miniconda.html#linux-installers).
```
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```
For performance and compatibility reasons, you should install `Mamba` via conda to install Snakemake. See [Snakemake documentation](https://snakemake.readthedocs.io/en/stable/getting_started/installation.html) for more details.
```
conda install -c conda-forge mamba
```
Once you have installed Conda and Mamba, you can install the software dependencies via the following commands:

```bash
# install environment using Conda
mamba env create --file ${SNK_REPO}/environment.yml

# activate the new Conda environment
conda activate severityPred_env

# install seurat-disk package
R -e 'remotes::install_github("mojaveazure/seurat-disk")'
```

Output files
-----------------------

Once the pipeline has run successfully, you should expect the following files in the output directory:
*   **`merge_training/`:**
    *   `QC.rds` - merged data
    *   `pseudo_bulk.h5Seurat` - expression average of all genes
    *   `fold_change.csv` - output of the DE analysis between conditions (findMarkers). 
    *   `selected_ge.csv` - expression average of the top genes
    *   `annotation.csv` - matrix representing the number of each cell per sample & type
    *   `MLP_CC.pkl` - the learned model based on the Cell Composition (CC)
    *   `MLP_GE.pkl` - the learned model based on the Gene Expression (GE)
    *   `MLP_CC_GE.pkl` - the learned joint model based on the Cell Composition (CC) and the Gene Expression (GE)
    *   `val_set.pkl` - list of the training sets from the 30 samplings
    *   `fig_metrics.pdf` - figures representing the different evaluation metrics (AUROC, AUPRC, Accuracy, ...) between the three models "CC, GE, and CC&GE"
    *   `pred_GE.csv` - prediction output scores per column of the validation set using the GE model (you will get as many columns as the number of samplings)
    *   `pred_CC.csv` - prediction output scores per column of the validation set using the CC model (you will get as many columns as the number of samplings)
    *   `pred_CC_GE.csv` - prediction output scores per column of the validation set using the joint model (you will get as many columns as the number of samplings)
    *   `pred_GE.txt` - evaluation metrics represented by the mean and the confidence interval of 95% of the validation set using the GE model
    *   `pred_CC.txt` - evaluation metrics represented by the mean and the confidence interval of 95% of the validation set using the CC model
    *   `pred_CC_GE.txt` - evaluation metrics represented by the mean and the confidence interval of 95% of the validation set using the joint model
*   **`pred/`:** - contains the prediction result fir the testing sets. This include the following files: `MLP_CC_GE.pdf`,`fig_shap.pdf`, `MLP_GE.csv`, `MLP_CC.csv`, and results of baselines models.

In Top5 model results, you should expect an output folder `figures` containing all figures in the manuscript.


Notes & Tips
------------
- We recommand using HPC and run job.sh to reproduce results.
- We used Singularity version 3.8.7.
- Please make sure to mount/bind all host repositories you use (for inputs and outputs) into your container and set a writable directory for the --directory option in snakemake when using singularity option.   
- Due to a relatively high number of outputs from each step of the workflow, it might generate a false errors of not finding a file. In this case, you just need to run the workflow again and it will continue from where it stopped automatically. Please only post issues if the errors persist.
- Please accept warnings coming with seurat-disk installation
