#!/usr/bin/env bash
if [ -f $1 ]; then
  config=$1
else
  echo "need a config file"
  exit
fi

weight=$2

export CUDA_VISIBLE_DEVICES=0,1
python -m torch.distributed.launch --master_port 1235 --nproc_per_node=2 --use_env \
    test_zeroshot.py --config ${config} --weights ${weight} ${@:3}
