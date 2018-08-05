nvidia-docker  run \
    -ti \
    -v "${PWD}:/app:rw" \
    -v "/nvme/simpsonsData/rawImageData/training:/data/train:rw" \
    -v "/nvme/simpsonsData/rawImageData/testing:/data/test:rw" \
    -p 666:8888 \
    fgiuste/neuroml:V3

### was the simpsons

#    gutmanlab/simpsonskeras:v1 /app/simpsonsmodel.py


