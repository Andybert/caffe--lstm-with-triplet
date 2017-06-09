#!/bin/bash

TOOLS=/mnt/68FC8564543F417E/caffe/caffe-master/build/tools

export HDF5_DISABLE_VERSION_CHECK=1
export PYTHONPATH="/mnt/68FC8564543F417E/caffe/caffe-master/python:$PYTHONPATH"

GLOG_logtostderr=1  $TOOLS/caffe train -solver lstm_solver_RGB_3_9.prototxt -weights /mnt/68FC8564543F417E/caffe/caffe-master/examples/LRCN_TEST/8/RGB_lstm_model_iter_30000.caffemodel -gpu 1
echo "Done."
