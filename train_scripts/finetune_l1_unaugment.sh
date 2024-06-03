#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd $SCRIPT_DIR/../

conda activate medficientsam

python src/train.py \
  experiment=finetune_l1_unaugmented \
  callbacks.model_checkpoint.dirpath=weights/finetuned-l1-unaugmented \
  logger.wandb.name=finetuned-l1-unaugmented
