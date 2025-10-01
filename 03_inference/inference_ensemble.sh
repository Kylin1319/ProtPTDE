#!/usr/bin/env bash

set -Eeuo pipefail

#######################################################################
# conda environment
#######################################################################

__conda_setup="$('conda' 'shell.bash' 'hook' 2> /dev/null)"
eval "$__conda_setup"
unset __conda_setup

conda activate Prot_PTDE

end=$(python3 -c "import json; print(json.load(open('../config/config.json'))['ensemble_size'])")
mut_counts=$1

for i in $(seq 1 $end); do
    python inference_muts.py --mut_counts "$mut_counts" --input_path "../02_final_model/ensemble_results/ensemble_${i}"
done
