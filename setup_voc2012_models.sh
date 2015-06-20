#!/bin/bash

TRAIN_CACHE_DIR=voc2012_train_cache

# set up svm linear model
mkdir -p models_svm_linear
cd models_svm_linear
../force_link_s.sh ../$TRAIN_CACHE_DIR/CaffeModel cnn
../force_link_s.sh ../$TRAIN_CACHE_DIR/BBoxRegTrain bboxreg
../force_link_s.sh ../$TRAIN_CACHE_DIR/Train/svm_linear classifier
../force_link_s.sh ../$TRAIN_CACHE_DIR/GPTrain/svm_linear gp
../force_link_s.sh ../$TRAIN_CACHE_DIR/PrepDataset/trainval-categlist.mat categ_list.mat

cd ..

# set up svm struct model

mkdir -p models_svm_struct
cd models_svm_struct
../force_link_s.sh ../$TRAIN_CACHE_DIR/CaffeModel cnn
../force_link_s.sh ../$TRAIN_CACHE_DIR/BBoxRegTrain bboxreg
../force_link_s.sh ../$TRAIN_CACHE_DIR/Train/svm_struct classifier
../force_link_s.sh ../$TRAIN_CACHE_DIR/GPTrain/svm_struct gp
../force_link_s.sh ../$TRAIN_CACHE_DIR/PrepDataset/trainval-categlist.mat categ_list.mat
cd ..

