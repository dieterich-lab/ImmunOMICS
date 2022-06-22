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
