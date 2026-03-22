#!/usr/bin/env bash
if [ -f $1 ]; then
  config=$1
else
  echo "need a config file"
  exit
fi

now=$(date +"%Y%m%d_%H%M%S")

export CUDA_VISIBLE_DEVICES=0,1
python -m torch.distributed.launch --master_port 1238 --nproc_per_node=2 --use_env \
         train_videomae.py --config ${config} --log_time $now
