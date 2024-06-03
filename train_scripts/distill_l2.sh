#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd $SCRIPT_DIR/../

conda activate medficientsam

python src/train.py \
  experiment=distill_l2_no_extracted \
  callbacks.model_checkpoint.dirpath=weights/distilled-l2 \
  logger.wandb.name=distill_l2_no_extracted_limit400k_lr75e-3_wd5e-4_bs8_epoch8
