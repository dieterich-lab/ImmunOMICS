COVID-19 severity prediction tool
================================

Snakemake pipeline for predicting severity in COVID-19.


Overview
--------

The workflow is.  All code is housed within scripts. In addition, this directory houses the data subdirectory which contains data for minimal examples.

Quick Setup
-----------

```bash
# clone the repo
git clone https://github.com/AminaLEM/Prediction_scOmics

# set code base path
SNK_REPO="`pwd`/Prediction_scOmics"

```

If you are running jobs on the cluster, it is best to first start a [tmux](https://github.com/tmux/tmux) session so that the session may be re-attached at a later time point. 

```bash
# start session
tmux new -s snkmk

# log back into a session
tmux -a snkmk
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

Here is the config file for the testing example.

```
# INPUT & OUTPUT PATHS
path_out: "../../output"          # path to the output directory, if it doesn't exist it will be created 
path_inp: "../data"           # path to the directory containing input datasets in h5Seurat format
path_ref: "../data/pbmc_multimodal.h5seurat"         # reference dataset for Seurat mapping  


# INPUTS PARAMS
training_data:               # Training datasets
    set1:  'bonn.h5Seurat'        
    
test_data:               # Testing datasets
    set1:  'berlin.h5Seurat'        
    set3:  'asan_.h5Seurat'        

```
Note that you can set as many training and testing datasets as you want. Datasets under `training_data` will be merged and 80% will be used for the training and 20 % for the validation split randomly 30 times. 

In case you want to test more datasets after generating your prediction model, just add the name of the datasets to the `testing_data` dictionnary and snakemake will generate only the missing outputs.

Output files
-----------------------

Once the pipeline has run successfully you should expect the following files in the output directory:
*   **`merge_training/`:**
    *   `QC.rds` - merged data
    *   `pseudo_bulk.h5Seurat` - expression average of all genes
    *   `fold_change.csv` - output of the DE analysis between conditions (findMarkers). 
    *   `selected_ge.csv` - expression average of the top genes
    *   `annotation.csv` - matrix representing number of each cells per sample & type
    *   `model_CC.pkl` - the learned model based on the Cell Composition (CC)
    *   `model_GE.pkl` - the learned model based on the Gene Expression (GE)
    *   `model_CC_GE.pkl` - the learned joint model based on the Cell Composition (CC) and the Gene Expression (GE)
    *   `train_set.pkl` - list of the training sets from the 30 samplings
    *   `val_set.pkl` - list of the training sets from the 30 samplings
    *   `val_set.pkl` - list of the training sets from the 30 samplings
    *   `fig_metrics.pdf` - figures representing the diffrent evaluation meterics (AUROC, AUPRC, Accuracy, ...) between the three models "CC, GE and CC&GE"
    *   `fig_shap.pdf` - figures representing barplots and violin plots of SHAP values from the joint model "CC&GE" on the validation set
    *   `pred_GE.csv` - prediction output scores per column of the validation set using the GE model (you will get as many columns as the number of samplings)
    *   `pred_CC.csv` - prediction output scores per column of the validation set using the CC model (you will get as many columns as the number of samplings)
    *   `pred_CC_GE.csv` - prediction output scores per columnof the validation set using the joint model (you will get as many columns as the number of samplings)
    *   `pred_GE.txt` - evaluation meterics represented by the mean and the confidence interval of 95% of the validation set using the GE model
    *   `pred_CC.txt` - evaluation meterics represented by the mean and the confidence interval of 95% of the validation set using the CC model
    *   `pred_CC_GE.txt` - evaluation meterics represented by the mean and the confidence interval of 95% of the validation set using the joint model
*   **`{test_data_filename}/`:** - contains the prediction result per testing set. This include the following files: `fig_metrics.pdf`,`fig_shap.pdf`, `pred_GE.csv`, `pred_CC.csv`, `pred_CC_GE.csv`, `pred_GE.txt`, `pred_CC.txt`, `pred_CC_GE.txt`

Reproducibility: Conda   
----------------------

One option to enhance reproducibility is to install software used via Conda.

```bash
# install environment using Conda
conda env create --file ${SNK_REPO}/environment.yml

# activate the new Conda environment
source activate severityPred_env
```


Reproducibility: Docker
-----------------------

An alternative option is to use a Docker image. One can easily generate a Docker image of all of the software used in this repository by using the Dockerfile. 

```bash
# build the Docker image using the Dockerfile
cd ${SNK_REPO}
docker build -t severityPred_env .
```

A pre-compiled Docker image is housed on the Docker Cloud. Download and use this Docker image:

```bash
# download the Docker image 
docker pull letaylor/severityPred_env:latest

# run the Snakemake pipeline through the container
# the Snakemake command used below is described in the src/README.md file
docker run -t -v ${SNK_REPO}:/SNK_REPO -v $(pwd):/CUR_DIR -e USERID=$UID sevpred_env:latest "snakemake --snakefile /SNK_REPO/src/snakefile --directory /CUR_DIR --cores all --configfile /SNK_REPO/src/config.yaml --printshellcmds"
```


Reproducibility: Singularity
----------------------------

A final option is to load the above Docker image using Singularity, which is designed for high-performance compute systems. To do so, simply add the --use-singularity flag when calling snakemake as descibed in the other README.md docs (within the different modules).

As an example, see below. Note that the Docker image is specified in the DOCKER variable of the config file (config.json).

```bash
# the Snakemake command used below is described in the qtl/README.md file
snakemake --snakefile ${SNK_REPO}/src/Snakefile --configfile config.json --predict --use-singularity --singularity-prefix $(pwd)/.snakemake/singularity
```
