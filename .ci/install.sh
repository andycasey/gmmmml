#!/bin/bash -x

# SSL workaround
mkdir -p ~/.config/Tectonic
cp .ci/tectonic.config.toml ~/.config/Tectonic/config.toml

# Conda install
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh;
bash miniconda.sh -b -p $HOME/miniconda
export PATH="$HOME/miniconda/bin:$PATH"
hash -r
conda config --set always_yes yes --set changeps1 no
conda update -q conda
conda info -a
conda install -c conda-forge -c pkgw-forge tectonic
tectonic --help
