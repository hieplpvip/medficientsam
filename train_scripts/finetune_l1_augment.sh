#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd $SCRIPT_DIR/../

conda activate medficientsam

python src/train.py \
  experiment=finetune_l1 \
  callbacks.model_checkpoint.dirpath=weights/finetuned-l1-augmented \
  logger.wandb.name=finetuned-l1-augmented
