#!/bin/bash

DST_DIR=../voc2007_train_cache/CaffeModel
mkdir -p $DST_DIR
cp -i deploy.prototxt $DST_DIR
cp -i input_mean.mat $DST_DIR
cp -i vgg16_voc2007_train_iter_60000 $DST_DIR/caffe_model


