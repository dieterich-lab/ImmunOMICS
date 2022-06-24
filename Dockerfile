FROM continuumio/miniconda3:latest

LABEL maintainer="Amina Lemsara <lemsaraamina@gmail.com>"

# Set up locales properly
RUN apt-get update && \
    apt-get install --yes --no-install-recommends locales && \
    apt-get purge && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN echo "en_US.UTF-8 UTF-8" > /etc/locale.gen && \
    locale-gen

ENV LC_ALL en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US.UTF-8

# Use bash as default shell, rather than sh
ENV SHELL /bin/bash

# run critical system updates ... one could also use the artful-curl pack
RUN apt-get update && \
    apt-get install --yes --no-install-recommends \
        wget bzip2 ca-certificates curl git && \
    apt-get purge && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*


# set up a user, to avoid running containers as root
ENV NB_USER container_user
ENV HOME /home/${NB_USER}
RUN adduser --disabled-password \
    --gecos "Default user" \
    ${NB_USER}

# download things to tmp
WORKDIR /tmp

# download the conda environment OR copy from local directory
# RUN wget --quiet https://raw.githubusercontent.com/letaylor/snakemake-qtl/master/environment.yml -O /tmp/environment.yml
COPY environment.yml /tmp/environment.yml

# install conda environment
# NOTE: conda clean -tipy
#   removes everything w/o confirmation (leaves all environments functional).
RUN conda update conda --yes && \
    #conda env create --name container_env --file environment.yml && \
    #conda list --name container_env && \
    #echo "source activate container_env" >> ${HOME}/.bashrc && \
    conda env update -v -n root --file /tmp/environment.yml && \
    conda list --name root && \
    #conda clean --all --yes && \
    conda clean -tipy && \
    rm /tmp/environment.yml

# export the conda path to force container_env since .bashrc necessarily 
# sourced when when the image is used
# ENV PATH /opt/conda/envs/container_env/bin:$PATH
# ENV CONDA_DEFAULT_ENV container_env
# ENV CONDA_PREFIX /opt/conda/envs/container_env
# set wd to user home
VOLUME ${HOME}
WORKDIR ${HOME}
COPY src ./
RUN chmod +x ./src

RUN snakemake --cores all all --snakefile scr/snakefile --configfile data/config.yml --printshellcmds


# clear tmp if there is anything in there...
RUN rm -rf /tmp/*

# set bash as the default entry point
ENTRYPOINT [ "/bin/bash", "-c" ]

# set the user
USER ${NB_USER}