nvidia-docker  run \
    -ti \
    -v "${PWD}:/app:rw" \
    -v "${HOME}/devel/KerasSimpsons_Tensorflow/rawImageData/testing:/data/test:rw" \
    -v "${HOME}/devel/KerasSimpsons_Tensorflow/rawImageData/training:/data/train:rw" \
    -v "${PWD}/output:/output:rw" \
    -p 666:8888 \
    fgiuste/neuroml:V3


#    -v "/nvme/simpsonsData/rawImageData/training:/data/train:rw" \
#    -v "/nvme/simpsonsData/rawImageData/testing:/data/test:rw" \

### was the simpsons

#    gutmanlab/simpsonskeras:v1 /app/simpsonsmodel.py


