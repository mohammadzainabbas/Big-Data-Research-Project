#!/bin/bash
#====================================================================================
# Author: Mohammad Zain Abbas
# Date: 25th Nov, 2022
#====================================================================================
# This script is used to set up the enviorment & installations
#====================================================================================

# Enable exit on error
set -e -u -o pipefail

log () {
    echo "[[ log ]] $1"
}

error () {
    echo "[[ error ]] $1"
}

fatal_error () {
    error "$1"
    exit 1
}

#Function that shows usage for this script
function usage()
{
cat << HEREDOC

Setup env for Big Data Research Project

Usage: 
    
    $progname [OPTION] [Value]

Options:

    -h, --help              Show usage

Examples:

    $ $progname
    ⚐ → Installs all dependencies for your BDRP project.

HEREDOC
}

progname=$(basename "$0")
env_name='bdrp'

#Get all the arguments and update accordingly
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -h|--help)
        usage
        exit 1
        ;;
        *) printf "\n$progname: invalid option → '$1'\n\n⚐ Try '$progname -h' for more information\n\n"; exit 1 ;;
    esac
    shift
done

check_conda() {
    if [ ! $(type -p conda) ]; then
        fatal_error "'conda' not found. Please install it first and re-try ..."
    else
        log "'conda' found ..."
    fi
}

create_conda_env() {
    conda create -n $env_name python=3 -y || error "Unable to create new env '$env_name' ..."
    conda activate $env_name &> /dev/null || echo "" > /dev/null
}

setup_project() {
    pip install -e . || error "Unable to install project dependencies ..."
}

log "Starting Setup for BDRP !!!"

log "Checking for 'conda' ..."
check_conda
log "Creating a new 'conda' env ..."
create_conda_env
log "Setup the project ..."
setup_project

log "All done !!"
