# KerasSimpsons_Tensorflow
Tutorial and stats to build a simple machine learning classifier using Tensorflow + Keras

## I am HEAVILY copying most if not all of the content from https://medium.com/@seixaslipe/building-a-simpsons-classifier-with-deep-learning-in-keras-36a47fe17f79

This is just my attempt to try this locally on my own system, and also do some benchmarks

This is based on these posts:
https://medium.com/alex-attia-blog/the-simpsons-character-recognition-using-keras-d8e1796eae36

and has code from here
https://github.com/alexattia/SimpsonRecognition.git


## Setting up my virtualenvironment in python
https://www.pyimagesearch.com/2017/09/25/configuring-ubuntu-for-deep-learning-with-python/


sudo pip install virtualenv virtualenvwrapper

mkvirtualenv simpsonsKeras 

## To start off
Download the data set from here:
https://www.kaggle.com/alexattia/the-simpsons-characters-dataset


mkdir rawImageData
cd rawImageData
unzip ~/Downloads/the-simpsons-character-dataset.zip





### This contains the following files
annotation.txt
characters_illustration.png
kaggle_simpson_testset
number_pic_char.csv
simpsons_dataset.tar.gz
weights.best.hdf5


## The file we are interested in for now is simpsons_data.tar.gz
tar -zxvf simpsons_data.tar.gz

### the rawImageData should only consists of the JPEG data, the files above should be moved to
simpsonsZipData or similar folder
## also moves the kaggle_simpson_testset into the root directory for the project as these contain
## images we will use for validation, at least for now



## setting up web app
cd web
ln -s ../RawImageData/
bower install

python -m SimpleHTTPServer


### virtual environemnt

mkvirtualenv KerasSimpsons -r requirements.txt


We want these plots:
https://medium.com/alex-attia-blog/the-simpsons-character-recognition-using-keras-d8e1796eae36


## Some tutorials for reference
https://alliseesolutions.wordpress.com/2016/09/08/install-gpu-tensorflow-from-sources-w-ubuntu-16-04-and-cuda-8-0/

## Used this for help setting up env
https://www.pyimagesearch.com/2017/09/25/configuring-ubuntu-for-deep-learning-with-python/

https://www.pyimagesearch.com/2017/09/27/setting-up-ubuntu-16-04-cuda-gpu-for-deep-learning-with-python/
