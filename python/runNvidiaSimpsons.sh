#docker run \
#    --runtime=nvidia \
nvidia-docker run \
    -ti \
    -v "${PWD}:/app:rw" \
    -v "/home/dagutman/devel/KerasSimpsons_Tensorflow/rawImageData/training:/data/train:rw" \
    -v "/home/dagutman/devel/KerasSimpsons_Tensorflow/rawImageData/testing:/data/test:rw" \
    -p 665:8888 --entrypoint python \
    fgiuste/neuroml:V3 /app/simpsonsmodel.py

#    gutmanlab/simpsonskeras:v1 /app/simpsonsmodel.py
