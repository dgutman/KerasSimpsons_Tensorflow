docker run \
    --runtime=nvidia \
    --rm \
    -ti \
    -v "${PWD}:/app" \
    -p 666:8888 \
    tensorflow/tensorflow:latest-gpu \
    python /app/benchmark.py gpu 10000 

