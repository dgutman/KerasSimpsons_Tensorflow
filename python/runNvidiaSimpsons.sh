docker run \
    --runtime=nvidia \
    -ti \
    -v "${PWD}:/app:rw" \
    -v "/home/dagutman/devel/KerasSimpsons_Tensorflow/rawImageData/training:/data/trainingdata:rw" \
    -v "/home/dagutman/devel/KerasSimpsons_Tensorflow/rawImageData/validation:/data/validationdata:rw" \
    -p 666:8888 --entrypoint python \
    fgiuste/neuroml:simpsons /app/simpsonsmodel.py
