#!/bin/bash

echo Get Trained Model
wget -c http://www.ytzhang.net/files/fgs-obj/cache/voc2007/voc2007_train_cache.cnn.tar.gz
echo ">>>" Decompressing
tar -xzf voc2007_train_cache.cnn.tar.gz
echo ">>>" Done

wget -c http://www.ytzhang.net/files/fgs-obj/cache/voc2007/voc2007_train_cache.rest.20150620.tar.gz
tar -xzf voc2007_train_cache.rest.20150620.tar.gz
echo ">>>" Decompressing
echo ">>>" Done

echo Get Orignal VGG Model

wget -c http://www.ytzhang.net/files/fgs-obj/cache/voc2007/cnn_finetuning.tar.gz
echo ">>>" Decompressing
tar -xzf cnn_finetuning.tar.gz
echo ">>>" Done


echo "******************************************************************"
echo "* Please Get VOC 2007 data by yourself and put it into ./voc2007 *"
echo "******************************************************************"


