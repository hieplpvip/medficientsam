#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd $SCRIPT_DIR/../

conda activate medficientsam

python src/train.py \
  experiment=finetune_l2 \
  callbacks.model_checkpoint.dirpath=weights/finetuned-l2-augmented \
  logger.wandb.name=finetuned-l2-augmented
