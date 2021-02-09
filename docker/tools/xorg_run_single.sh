#!/bin/bash

if (( $# < 1 ))
then
    echo "Usage: xorg_run_single.sh <display_i>, displai_i is a number from 0 to number of GPU"
    exit 0
fi

BASEDIR=$(dirname $0)

nohup Xorg :$1 vt5 -config ./single_card_conf/xorg$1.conf -noreset +extension GLX +extension RANDR +extension RENDER &> $BASEDIR/logs/$1_$(hostname).log &
echo $!
