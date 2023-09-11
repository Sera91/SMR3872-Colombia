#!/usr/bin/env bash

# This script activates the conda environment
# If the envrionment is missing it is installed.

if [ ! -z "$BASH" ] && [ ! -z "$BASH_SOURCE" ];
then
    scriptname="${BASH_SOURCE}";
elif  [ ! -z "${ZSH_NAME}" ] && [ ! -z "${(%):-%N}" ];
then
    scriptname="${(%):-%N}";
elif [ ! -z "$KSH_VERSION" ];
then
    scriptname="${.sh.file}"
else
    echo "Unsupported shell detected. Try: bash, zsh or ksh."
    return 1;
fi

SCRIPT_PATH=$(cd -- $(dirname "${scriptname}") && pwd)

CONDABIN=${SCRIPT_PATH}/env/miniconda/bin
CONDAACTIVATE=${CONDABIN}/activate
CONDA=${CONDABIN}/mamba

# If conda is not installed, download miniconda
if ! command -v ${CONDA} &> /dev/null
then
    echo "conda not found at ${CONDA}"
    echo "script path is ${SCRIPT_PATH}"
    EXAMPLEARROWPYBINDCMAKEMINICONDA=${SCRIPT_PATH}/env/miniconda
    if [ ! -d "$EXAMPLEARROWPYBINDCMAKEMINICONDA" ]
    then
        mkdir -p ${SCRIPT_PATH}/env
        cd ${SCRIPT_PATH}
        echo "Downloading miniconda..."
        case "$(uname -s)" in
            Darwin*)    OSKIND=MacOSX;;
            *)          OSKIND=Linux;;
        esac
        echo "Detected OS = ${OSKIND}"
        wget https://repo.continuum.io/miniconda/Miniconda3-latest-${OSKIND}-x86_64.sh -nv -O ${SCRIPT_PATH}/miniconda.sh \
        && bash ${SCRIPT_PATH}/miniconda.sh -b -p $EXAMPLEARROWPYBINDCMAKEMINICONDA \
        && . $EXAMPLEARROWPYBINDCMAKEMINICONDA/bin/activate \
        && conda config --set always_yes yes \
        && conda config --set env_prompt '({name}) ' \
        && conda update -q conda \
        && conda install mamba -c conda-forge \
        && rm miniconda.sh
    fi
    . $EXAMPLEARROWPYBINDCMAKEMINICONDA/bin/activate
    if ! command -v ${CONDA} &> /dev/null
    then
        echo "ERROR: cannot find conda command!"
        return 1
    fi
fi

# If environment does not exist, create it
EXAMPLEARROWPYBINDCMAKECONDAENV=${SCRIPT_PATH}/env/example-apache-arrow-pybind11-cmake
if [ ! -d "$EXAMPLEARROWPYBINDCMAKECONDAENV" ]
then 
    ${CONDA} env create -f environment.yml --prefix $EXAMPLEARROWPYBINDCMAKECONDAENV
fi

# Activate the environment
. ${CONDAACTIVATE}
. activate $EXAMPLEARROWPYBINDCMAKECONDAENV

# end setup-environment.

# Install pre-commit hooks if they have not already been installed
if test -d "$SCRIPT_PATH/.git" && ! test -f "$SCRIPT_PATH/.git/hooks/pre-commit";
then
    (cd ${SCRIPT_PATH}; pre-commit install)
fi

export PATH=${SCRIPT_PATH}/bin:${PATH}
