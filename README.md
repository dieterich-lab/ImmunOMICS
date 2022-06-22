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
Please put all your input datasets in one directory. 
The input datasets should have the followinf metadata columns: 
* "sampleID": sample IDs
* "condition": Mild or Severe
* "batch": the batch name of your data
* "who_score": if availbale else =condition (it serves as factor for the traning/validation split)

Here is the config file for the testing example.

```
# INPUT & OUTPUT PATHS
path_out: "../../output_berlin_stan"          # path to the output directory, if it doesn't exist it will be created 
path_inp: "../data"           # path to the directory containing the input BAM files
path_ref: "../data/pbmc_multimodal.h5seurat"         # reference transcriptome 


# INPUTS PARAMS
training_data:               # bam files for 2 (pairewise comparison) or 3 conditions (3 way comparison)
    set14:  'stanford_pbmc_29.h5Seurat'        # testing set 2
#     set41:  'stanford_pbmc_29.h5Seurat'        # testing set 2
#     set4:  'mgh.h5Seurat'        # testing set 2
    set1: 'berlin.h5Seurat'         

test_data:               # bam files for 2 (pairewise comparison) or 3 conditions (3 way comparison)
#     set1:  'berlin.h5Seurat'        # testing set 1
    set3:  'asan_.h5Seurat'        # testing set 2
#     set4:  'stanford_pbmc_29.h5Seurat'        # testing set 2
#     set5:  'stanford_29.h5Seurat'        # testing set 2
    set7:  'cam.h5Seurat'        # testing set 2
#     set6:  'ncl.h5Seurat'        # testing set 2
    set8:  'ucl.h5Seurat'        # testing set 2
    set9:  'mgh.h5Seurat'        # testing set 2
    set10: 'bonn.h5Seurat'         

```
Note that you can set as many training and testing datasets as you want. Datasets under `training_data` will be merged and 80% will be used for the training and 20 % for the validation split randomly 30 times. 

In case you want to test more datasets after generating your prediction model, just add the name of the datasets to the `testing_data` dictionnary and snakemake will generate only the missing outputs.

Output files
-----------------------

Once the pipeline has run successfully you should expect the following files in the output directory:
*   **`merge_training/`:**
    *   `QC.rds` - merged data
    *   `pseudo_bulk.h5Seurat` - expression average of all genes
    *   `fold_change.csv` - findMarker output of the DE between conditions. 
    *   `selected_ge.csv` - expression average of the top genes
    *   `annotation.csv` - matrix representing number of each cells per sample & type
    *   `model.pkl` - the learned model
    *   `train_set.pkl` - list of the training sets from the 30 samplings
    *   `val_set.pkl` - list of the training sets from the 30 samplings
    

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
docker run -t -v ${SNK_REPO}:/SNK_REPO -v $(pwd):/CUR_DIR -e USERID=$UID letaylor/severityPred_env:latest "snakemake --snakefile "
```


Reproducibility: Singularity
----------------------------

A final option is to load the above Docker image using Singularity, which is designed for high-performance compute systems. To do so, simply add the --use-singularity flag when calling snakemake as descibed in the other README.md docs (within the different modules).

As an example, see below. Note that the Docker image is specified in the DOCKER variable of the config file (config.json).

```bash
# the Snakemake command used below is described in the qtl/README.md file
snakemake --snakefile ${SNK_REPO}/src/Snakefile --configfile config.json --predict --use-singularity --singularity-prefix $(pwd)/.snakemake/singularity
```
