docker run \
    --runtime=nvidia \
    -ti \
    -v "${PWD}:/app" \
    -v "/home/dagutman/devel/KerasSimpsons_Tensorflow/rawImageData/:/imageData" \
    -p 666:8888 \
    tensorflow/tensorflow:latest-gpu 
    # python /app/runSimpsonsModel_NvidiaDocker.py 

  	# jupyter/tensorflow-notebook \
    # --rm \
#tensorflow/tensorflow:latest-gpu \


