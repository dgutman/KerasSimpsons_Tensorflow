docker run \
    --runtime=nvidia \
    -ti \
    -v "${PWD}:/app:rw" \
    -v "/home/dagutman/devel/KerasSimpsons_Tensorflow/rawImageData/training:/data/train:rw" \
    -v "/home/dagutman/devel/KerasSimpsons_Tensorflow/rawImageData/testing:/data/test:rw" \
    -p 666:8888 \
    fgiuste/neuroml:simpsons

#    gutmanlab/simpsonskeras:v1 /app/simpsonsmodel.py
