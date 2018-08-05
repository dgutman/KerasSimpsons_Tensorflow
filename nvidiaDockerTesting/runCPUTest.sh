docker run \
    --runtime=nvidia \
    --rm \
    -ti \
    -v "${PWD}:/app" \
    gcr.io/tensorflow/tensorflow:latest-gpu \
    python /app/benchmark.py cpu 10000docker run \
    --runtime=nvidia \
    --rm \
    -ti \
    -v "${PWD}:/app" \
    gcr.io/tensorflow/tensorflow:latest-gpu \
    python /app/benchmark.py cpu 10000
