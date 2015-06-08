#!/bin/bash

SHOW_HELP=0;
if [ "$1" == "--help" ]; then
    SHOW_HELP=1;
fi
if [ "$1" == "-h" ]; then
    SHOW_HELP=1;
fi
if [ -z "$1" ]; then
    SHOW_HELP=1;
fi

if [ $SHOW_HELP -eq "1" ]; then
    echo Usage: test_snapshots.sh DEVICE_ID VALIDATION_PROTOTXT ITER_NUM SNAPSHOT_PREFIX  >&2
    echo Example: test_snapshots.sh 0 imagenet_val.prototxt 200 e/x/a/m/p/l/e/caffe_imagenet_train_iter_ >&2
    exit;
fi

SCRIPT_DIR=`dirname "$0"`
SCRIPT_DIR=`readlink -f $SCRIPT_DIR`
TEST_NET_DIR=`readlink -f "$SCRIPT_DIR/../../build/tools"`

DEVICE_ID=$1
VALIDATION_PROTOTXT=$2
ITER_NUM=$3
SNAPSHOT_PREFIX=$4

ls $SNAPSHOT_PREFIX* | grep -v '\.solverstate$' | while read line; do
    echo ======= $line
    fnonly=`basename $line`
    if [ -f $fnonly.accuracy ]; then
        echo Existed ... SKIP
        continue
    fi
    rm -f tmp_fifo
    mkfifo tmp_fifo
    cat tmp_fifo &
    GLOG_logtostderr=1 "$TEST_NET_DIR/test_net_x.bin" $DEVICE_ID \
        $VALIDATION_PROTOTXT $line $ITER_NUM GPU 2>&1 | tee tmp_fifo | grep 'accuracy:' | grep 'Test accuracy:' | sed -e 's/^.*\] Test accuracy://' | tee  $fnonly.accuracy.doing
    mv $fnonly.accuracy.doing $fnonly.accuracy
done

rm -f tmp_fifo

