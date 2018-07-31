# A multi classes image classifier, based on convolutional neural network using Keras and Tensorflow. 
# A multi-label classifier (having one fully-connected layer at the end), with multi-classification (18 classes, in this instance).
# Largely copied from the code https://gist.github.com/seixaslipe
# This is based on these posts: https://medium.com/alex-attia-blog/the-simpsons-character-recognition-using-keras-d8e1796eae36
# Data downloaded from Kaggle 
# Will emulate the image classification functionlities for Neuro Pathology images/slides (WSI-Whole Slide images)
# Will implement/include data manipulating functionalities based on Girder (https://girder.readthedocs.io/en/latest/)
# Has 6 convulsions, filtering start with 64, 128, 256 with flattening to 1024
# Used Keras.ImageDataGenerator for Training/Validation data augmentation and the augmented images are flown from respective directory
# Environment: A docker container having Keras, TensorFlow, Python-2 with GPU based execution

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras.callbacks import Callback
import datetime, time, os, sys
import numpy as np
import h5py
import matplotlib as plt
plt.use('Agg')
import matplotlib.pyplot as pyplot
pyplot.figure
import pickle 
#from pickle import load
from sklearn.metrics import classification_report, confusion_matrix
import subprocess
import pandas as pd

import nvidia_smi as nvs
import io
import pickle
import json


try:
    to_unicode = unicode
except NameError:
    to_unicode = str

modelInfo = {}
modelInfo['Device']  = {} ## Initialize an object to store info on the model and time info

nvs.nvmlInit()

driverVersion = nvs.nvmlSystemGetDriverVersion()
print("Driver Version: {}".format(driverVersion))
modelInfo['Device']['driverVersion']  = driverVersion

# e.g. will print:
#   Driver Version: 352.00
deviceCount = nvs.nvmlDeviceGetCount()
deviceNames = []
for i in range(deviceCount):
    handle = nvs.nvmlDeviceGetHandleByIndex(i)
    dvn = nvs.nvmlDeviceGetName(handle) # store the device name
    print("Device {}: {}".format(i,  dvn))
    deviceNames.append(dvn)
    # e.g. will print:
    #  Device 0 : Tesla K40c
nvs.nvmlShutdown()

modelInfo['Device']['deviceNames']  = deviceNames




### These parameters can be tuned and may affect classification results or accuracy
img_width, img_height = 64, 64
epochs = 1
batch_size = 128


modelInfo['batch_size'] = batch_size
modelInfo['epochs'] = epochs
modelInfo['img_width'] = 64
modelInfo['img_height'] = 64
 

### Define input dirs and output for results which contain the models as well as stats on the run
train_data_dir = '/data/train' 
validation_data_dir = '/data/test' 

resultsDir ="/app/results/"
if not os.path.isdir(resultsDir):
    os.makedirs(resultsDir)

nb_train_samples = 0

for root, dirs, files in os.walk(train_data_dir):
    nb_train_samples += len(files)

nb_validation_samples = 0
for root, dirs, files in os.walk(validation_data_dir):
    nb_validation_samples += len(files)


# Model definition
if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)


# Data augmentation for training
train_datagen = ImageDataGenerator(
    rescale=1. / 255.0,
    shear_range=0.2,
    zoom_range=0.2,
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    fill_mode = 'nearest',
    horizontal_flip=True)

# Only rescaling for validation
valid_datagen = ImageDataGenerator(rescale=1. / 255.0)

# Flows the data directly from the directory structure, resizing where needed
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

validation_generator = valid_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

NumLabels = len(validation_generator.class_indices)

'''
6-conv layers - added on 06/21, Raj
'''
model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same', input_shape=input_shape))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(256, (3, 3), padding='same')) 
model.add(Activation('relu'))
model.add(Conv2D(256, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(1024))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(NumLabels, activation = 'softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


# Captures GPU usage
#subprocess.Popen("timeout 120 nvidia-smi --query-gpu=utilization.gpu,utilization.memory --format=csv -l 1 | sed s/%//g > /app/results/GPU-stats.log",shell=True)



# Timehistory callback to get epoch run times
class TimeHistory(Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)

time_callback = TimeHistory()


# Model fitting and training run
simpsonsModel = model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size,

    callbacks=[time_callback])    


print "Finished running the basic model... trying to save results now.."

# To write the each epoch run time into a json file
now = datetime.datetime.now()
filetime = str(now.year)+str(now.month)+str(now.day)+'_'+str(now.hour)+str(now.minute)+str(now.second)


modelInfo['epochTimeInfo'] = time_callback.times

# epochfilename='SimpsonsEpochRuntime_'+filetime+'.json'
# timefile = open(epochfilename, "a+")
# times = time_callback.times
# print >> timefile, times

modelfilename=resultsDir+'Simpsonsmodel_'+filetime+'.h5'
model.save(modelfilename)

modelInfo['historyData'] =  pd.DataFrame(simpsonsModel.history).to_dict(orient='records')


#pandas.DataFrame(simpsonsModel.history).to_json("/data/trainingdata/simpsons_history_0629.json")


# saving Confusion Matrix and Classification Report to a text file for human vision
target_names = validation_generator.class_indices

Y_pred = model.predict_generator(validation_generator, nb_validation_samples // batch_size+1)
y_pred = np.argmax(Y_pred, axis=1)

cnf_matrix = confusion_matrix(validation_generator.classes, y_pred)
cls_rpt = classification_report(validation_generator.classes, y_pred, target_names=target_names) 


# Serialize confusion matrix and prediction/probabilities matrix stores in json file

modelInfo['confusionMatrix']   =  pd.DataFrame(cnf_matrix).to_dict(orient='records')
modelInfo['prediction_report'] =  pd.DataFrame(y_pred).to_dict(orient='records')

#modelInfo['classification_report'] = 


# df=pd.DataFrame(cnf_matrix)
# df.to_json(rptjson, orient='records', lines=True)

# sysoptfile = 'classificationSystemEnvironment_'+filetime+'.txt'
# import subprocess
# sysopt = (subprocess.check_output("lscpu", shell=True).strip()).decode()
# with open(sysoptfile,"a+") as f:
#     for line in sysopt:
#         f.write(line)








# saving Confusion Matrix and Classification Report to a file
target_names = validation_generator.class_indices
optfile = resultsDir+ 'SimpsonsModeoutput_'+filetime+'.txt'
file = open(optfile, "a+")
Y_pred = model.predict_generator(validation_generator, nb_validation_samples // batch_size+1)
y_pred = np.argmax(Y_pred, axis=1)
ptropt= 'Confusion Matrix' 
print >> file, ptropt
cnf_matrix = confusion_matrix(validation_generator.classes, y_pred)
print >>file, cnf_matrix


ptropt = 'Classification Report'
print >> file, ptropt
cls_rpt = classification_report(validation_generator.classes, y_pred, target_names=target_names) 
print >> file, cls_rpt
file.close()                                         



resultSummaryFile = resultsDir + filetime + ".GutmansTextFile.json"


#Confusion Matrix is shown on a Plot
pyplot.figure(figsize=(8,8))
cnf_matrix =confusion_matrix(validation_generator.classes, y_pred)
#classes = list(chardict.values())
classes = list(target_names)
pyplot.imshow(cnf_matrix, interpolation='nearest')
pyplot.colorbar()
tick_marks = np.arange(len(classes))  
_ = pyplot.xticks(tick_marks, classes, rotation=90)
_ = pyplot.yticks(tick_marks, classes)
plotopt= resultsDir + 'SimpsonsModelImage_'+filetime+'.png'
pyplot.savefig(plotopt)


#To plot GPU usage
# gpu = pd.read_csv("/app/results/GPU-stats.log")   # make sure that 120 seconds have expired before running this cell
# gpuplt=gpu.plot()
# gpuplt=pyplot.show()
# gpuplt='/app/results/SimsonsGPUImage_'+filetime+'.png'
# pyplot.savefig(gpuplt) 



# saving Confusion Matrix and Classification Report to a file
target_names = validation_generator.class_indices
optfile = 'SimsonsModeloutput_'+filetime+'.txt'
file = open(optfile, "a+")
Y_pred = model.predict_generator(validation_generator, nb_validation_samples // batch_size+1)
y_pred = np.argmax(Y_pred, axis=1)
ptropt= 'Confusion Matrix' 
print >> file, ptropt
cnf_matrix = confusion_matrix(validation_generator.classes, y_pred)
print >>file, cnf_matrix

# Serialize confusion matrix and stores in json file
cmfile= resultsDir + 'SimpsonsModelConfusionMatrix_'+filetime+'.json' 
with io.open(cmfile, 'w', encoding='utf8') as outfile:
    str_ = json.dumps(cnf_matrix.tolist(),
                      indent=4, sort_keys=True,
                      separators=(',', ':'), ensure_ascii=False)
    outfile.write(to_unicode(str_))



modelInfo['confusionMatrixV1'] =  cnf_matrix.tolist()

ptropt = 'Classification Report'
print >> file, ptropt
cls_rpt = classification_report(validation_generator.classes, y_pred, target_names=target_names) 
print >> file, cls_rpt
rptjson= resultsDir + 'SimpsonsModelClassificationReport_'+filetime+'.json' 
with io.open(rptjson, 'w', encoding='utf8') as outfile:
    str_ = json.dumps(cnf_matrix.tolist(),
                      indent=4, sort_keys=True,
                      separators=(',', ':'), ensure_ascii=False)
    outfile.write(to_unicode(str_))
file.close()                                         

# sysoptfile = resultsDir + 'SimpsonsSystemEnvironment_'+filetime+'.txt'
# import subprocess
# sysopt = (subprocess.check_output("lscpu", shell=True).strip()).decode()
# with open(sysoptfile,"a+") as f:
#     for line in sysopt:
#         f.write(line)

# from tensorflow.python.client import device_lib
# LOCAL_DEVICES = device_lib.list_local_devices()

# modelInfo['LOCAL_DEVICES'] = LOCAL_DEVICES



with open(resultSummaryFile ,"w")  as fp:
    json.dump(modelInfo,fp,indent=2)
