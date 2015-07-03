#!/bin/bash

echo Get Trained Model

wget -c http://www.ytzhang.net/files/fgs-obj/cache/voc2012/voc2012_train_cache.cnn.tar.gz
echo ">>> Decompressing"
tar -xzf voc2012_train_cache.cnn.tar.gz
echo "Done"

wget -c http://www.ytzhang.net/files/fgs-obj/cache/voc2012/voc2012_train_cache.rest.tar.gz
echo ">>> Decompressing"
tar -xzf voc2012_train_cache.rest.tar.gz
echo "Done"


