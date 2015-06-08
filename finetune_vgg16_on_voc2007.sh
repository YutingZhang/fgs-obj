#!/bin/bash

# call the script for finetuning
cd cnn_finetuning
echo ================ Finetuning =================
./finetune_net.sh $1
echo ============== End of Finetuning ============

echo Install to voc2007_train_cache:
./install2cache_dir.sh


