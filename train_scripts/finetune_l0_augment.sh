#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd $SCRIPT_DIR/../

conda activate medficientsam

python src/train.py \
  experiment=finetune_l0 \
  callbacks.model_checkpoint.dirpath=weights/finetuned-l0-augmented \
  logger.wandb.name=finetuned-l0-augmented
