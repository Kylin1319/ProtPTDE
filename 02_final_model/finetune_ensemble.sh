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

for i in $(seq 1 $end); do
    python finetune.py --input_folder "ensemble_results/ensemble_${i}"
done