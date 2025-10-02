#!/usr/bin/env bash

set -Eeuo pipefail

#######################################################################
# conda environment
#######################################################################

__conda_setup="$('conda' 'shell.bash' 'hook' 2> /dev/null)"
eval "$__conda_setup"
unset __conda_setup

conda activate Prot_PTDE

nohup python inference_muts.py --mut_counts 2 --input_path "../02_final_model" > log/mut_counts_2.log 2>&1 &
nohup python inference_muts.py --mut_counts 3 --input_path "../02_final_model" > log/mut_counts_3.log 2>&1 &
nohup python inference_muts.py --mut_counts 4 --input_path "../02_final_model" > log/mut_counts_4.log 2>&1 &
