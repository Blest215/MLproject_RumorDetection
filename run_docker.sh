#!/bin/bash

DOCKERFILE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPOSITORY="shygiants/rumor"
CONTAINER_NAME="rumor"

# Build docker image
nvidia-docker build -t ${REPOSITORY} ${DOCKERFILE_DIR}

# TODO: check if corresponding container is running
# Remove current running container
nvidia-docker stop ${CONTAINER_NAME}
nvidia-docker rm ${CONTAINER_NAME}

# Run docker container
nvidia-docker run --name ${CONTAINER_NAME} -v ${DATASET_DIR}:/dataset -v ${SUMMARY_DIR}:/summary -p 8888:8888 -p 6006:6006 -d ${REPOSITORY} $1