#!/usr/bin/env bash

set -Eeuo pipefail

mkdir -p log

nohup bash train_ensemble.sh > log/train_ensemble.log 2>&1 &