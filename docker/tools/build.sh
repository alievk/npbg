#!/bin/sh

set -x

. ./constants.sh

docker build -t ${NAME} -f ../Dockerfile ..

docker tag $NAME $TAG
