#!/bin/sh
set -x

SCRIPTPATH="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
cd $SCRIPTPATH
mkdir -p logs
. ./constants.sh

GPU_NUMBER=$1
shift 1

CMD=$@

X_GPU=$(echo ${GPU_NUMBER} | tr "," "\n" | head -1)
CONTAINER=${NAME}_${X_GPU}

X_PID=$(./xorg_run_single.sh ${X_GPU})

nvidia-docker container run \
    -it \
	--rm \
	-v $SRC_DIR:/home/docker/src \
	-v $DATA_DIR:$DATA_DIR \
    -e DISPLAY=:$X_GPU \
	-e CUDA_VISIBLE_DEVICES=${GPU_NUMBER} \
	-v /tmp/.X11-unix:/tmp/.X11-unix \
	-w $WORK_DIR \
	-u ${UID}:${UID} \
	-h $NAME \
	--shm-size 50G \
	--name $CONTAINER \
	$TAG \
	$CMD

CONTAINER_RUNNING=`docker ps | grep $CONTAINER`
if [[ ! -z ${CONTAINER_RUNNING} ]];
then
	docker container stop -t 0 $CONTAINER;
fi

kill -9 $X_PID
