#!/bin/bash

GPU_ID=$1
if [ "x$GPU_ID" = "x" ]; then
    GPU_ID=0
    fi
    GLOG_logtostderr=1 ../caffe/build/tools/finetune_net_x $GPU_ID solver.prototxt \
                     ./VGG_ILSVRC_16_layers.caffemodel

