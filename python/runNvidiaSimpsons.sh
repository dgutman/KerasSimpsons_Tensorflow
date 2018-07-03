docker run \
    --runtime=nvidia \
    -ti \
    -v "${PWD}:/app" \
    -v "/home/dagutman/devel/KerasSimpsons_Tensorflow/rawImageData/:/data/trainingdata" \
    -p 666:8888 --entrypoint python \
    fgiuste/neuroml:simpsons /app/runSimpsonsModel_DGEdits_withprediction1.py