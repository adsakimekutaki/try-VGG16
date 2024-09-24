NAME="work-finetuning"
IMAGE="ubuntu"

all: build run

build:
	docker build -t ${IMAGE} ./

run:
	docker run --name ${NAME} \
    --rm \
    -it \
    --volume ./src:/work/finetuning \
    --gpus all \
    ${IMAGE}

test:
	docker run --rm hello-world
	docker run --rm --gpus=all ${IMAGE} nvidia-smi
