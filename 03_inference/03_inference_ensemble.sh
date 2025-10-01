#!/usr/bin/env bash

set -Eeuo pipefail

#######################################################################
# conda environment
#######################################################################

__conda_setup="$('conda' 'shell.bash' 'hook' 2> /dev/null)"
eval "$__conda_setup"
unset __conda_setup

conda activate Prot_PTDE

nohup bash inference_ensemble.sh 2 > log/mut_counts_2.log 2>&1 &
nohup bash inference_ensemble.sh 3 > log/mut_counts_3.log 2>&1 &
nohup bash inference_ensemble.sh 4 > log/mut_counts_4.log 2>&1 &