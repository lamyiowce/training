#!/usr/bin/env bash

source ~/specaugment/venv/bin/activate

export PYTHONPATH=/home/julia/ml_input_processing/
LD_LIBRARY_PATH=/usr/local/cuda-11.1/targets/x86_64-linux/lib/
NUM_EPOCHS=3
SNAPSHOT_PATH=/training-data-speech/snapshot
COMMON_FLAGS="--num_epochs $NUM_EPOCHS --params cloud --caching_period 0 --tf_data --num_samples 500 --wav --checkpoint --save_folder /training-data-speech/models --acc 23 --model_path /training-data-speech/models/deepspeech.pth.tar"

# Measure maximal possible model data ingestion.
cd ~/specaugment/mlcommons/training/speech_recognition/pytorch
rm train.log
rm train.err

echo python3 train.py $COMMON_FLAGS --tag test --repeat_single_batch --bs 16
python3 train.py $COMMON_FLAGS --tag test --repeat_single_batch --bs 16 #>> train.log 2>> train.err
#TF_DUMP_GRAPH_PREFIX=/tmp/generated \
#  TF_XLA_FLAGS="--tf_xla_clustering_debug --tf_xla_auto_jit=2" \
#  XLA_FLAGS="--xla_dump_hlo_as_text --xla_dump_to=/tmp/generated" \
#echo python3 train.py $COMMON_FLAGS --wav --config config-cloud-wav.yml --tag tpu --repeat_single_batch --bs 256 --steps_per_ex 1
#python3 train.py $COMMON_FLAGS --wav --config config-cloud-wav.yml --tag tpu --repeat_single_batch --bs 256 --steps_per_ex 1 >> train.log 2>> train.err
#echo python3 train.py $COMMON_FLAGS --wav --config config-cloud-wav.yml --tag tpu --repeat_single_batch --bs 512 --steps_per_ex 1
#python3 train.py $COMMON_FLAGS --wav --config config-cloud-wav.yml --tag tpu --repeat_single_batch --bs 512 --steps_per_ex 1 >> train.log 2>> train.err


#sleep 10m
#$DRYRUN sudo shutdown -h now
