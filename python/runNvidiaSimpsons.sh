docker run \
    --runtime=nvidia \
    -ti \
    -v "${PWD}:/app" \
    -v "/home/dagutman/devel/KerasSimpsons_Tensorflow/rawImageData/:/imageData" \
    -p 666:8888 \
    fgiuste/neuroml:simpsons python runSimpsonsModel_DGEdits_withprediction1.py 