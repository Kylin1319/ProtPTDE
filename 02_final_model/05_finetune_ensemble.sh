#!/usr/bin/env bash

set -Eeuo pipefail

mkdir -p log

nohup bash finetune_ensemble.sh > log/finetune_ensemble.log 2>&1 &