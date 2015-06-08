#!/bin/bash

TRAIN_CACHE_DIR=voc2007_train_cache

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

# set up region proposal cache 
./force_link_s.sh $TRAIN_CACHE_DIR/RegionProposal/test/boxes.mat voc2007_test_bbox_cache.mat

# set up feature cache
./force_link_s.sh $TRAIN_CACHE_DIR/Features4Proposed voc2007_feature_cache

